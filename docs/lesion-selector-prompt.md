# Task: replace size-threshold post-processing with a learned lesion selector

You are working in `D:\dev\mri_3d`, a BraTS 2024 brain tumour segmentation project.
Read `README.md`, `reports/results.md`, `src/mri3d/metrics.py`, `src/mri3d/predict.py`
and `src/mri3d/postprocess.py` before writing any code. Do not skip this — the
existing conventions matter and several of them are easy to get wrong.

**Do not retrain anything. Do not modify `configs/splits.json`. Do not commit
anything under `data/` or `outputs/`.**

---

## 1. Background: what the problem actually is

The trained model segments the main tumour well and then invents roughly one
extra lesion per case somewhere else in the head. Volumetric Dice barely
registers these; lesion-wise Dice (what BraTS ranks on) charges a full zero for
every spurious connected component.

Current test results (`reports/scores_gli_main_test.csv`, 203 cases), split by
whether the case had any false-positive lesion:

| Class | lesion-wise, FP=0 | lesion-wise, FP>0 | volumetric, FP=0 | volumetric, FP>0 |
|---|---|---|---|---|
| SNFH | 0.719 (n=102) | 0.345 (n=101) | **0.860** | **0.842** |
| WT   | 0.735 (n=111) | 0.352 (n=92)  | **0.866** | **0.855** |
| ET   | 0.730 (n=111) | 0.355 (n=43)  | 0.794 | 0.694 |
| RC   | 0.773 (n=128) | 0.286 (n=41)  | 0.775 | 0.569 |
| NETC | 0.614 (n=60)  | 0.251 (n=36)  | 0.524 | 0.412 |

The SNFH and WT rows are the key evidence: **volumetric Dice is essentially
identical whether or not the case has false positives** (0.860 vs 0.842). The
main tumour is segmented just as well. The spurious components are additive
noise elsewhere in the volume, not a symptom of a degraded segmentation. If
FP-heavy cases scored like clean ones, the 4-region mean lesion-wise Dice would
go from ~0.573 to ~0.709.

An earlier attempt to fix this by thresholding on **component size** largely
failed: a single global threshold gained +0.009 on validation and **−0.0006** on
test. Per-class thresholds gained +0.039 on validation but only **+0.0126** on
test. See the "Post-processing" section of `reports/results.md`. Size does not
separate real from spurious components.

**Why it failed structurally:** `predict.py` does `pred = logits.argmax(dim=1)`
and saves a `uint8` mask. Every probability is discarded at that line, so
`postprocess.py` only ever sees geometry. Size was the only available lever.

## 2. The idea to implement

Score each predicted connected component with a small, interpretable model that
uses confidence, geometry and spatial context, then keep or drop it against a
threshold **derived from the scoring function rather than tuned**.

### 2.1 The break-even threshold (derive this in a docstring, it is the core argument)

For a case whose other lesion-wise entries number N and average L, consider one
marginal predicted component c, with p = probability it is real and d = the Dice
it would score if real:

- If c is **real**: its reference lesion is an entry either way. Keeping c scores
  it d instead of 0 → gain `d/N`.
- If c is **spurious**: keeping it adds a new entry scoring 0 → loss `L/(N+1)`.

Expected change from keeping ≈ `p·d/N − (1−p)·L/(N+1)`. Setting this to zero, N
cancels for large N:

```
p* = L / (L + d)
```

With L ≈ 0.72 (mean lesion-wise Dice of clean cases) and d ≈ 0.70 (typical Dice
of a correct detection), **p\* ≈ 0.51**.

Two consequences to state explicitly in the code and the write-up:

1. The optimal rule is a **flat threshold on the posterior**, independent of how
   many components have already been kept. The escalating bar on *raw
   confidence* emerges from the prior — the posterior that the 3rd-ranked
   component is real is much lower than for the 1st — not from the metric.
2. p\* falls when L falls, so the rule correctly becomes more willing to gamble
   on an extra lesion in a case that is already scoring badly.

Compute L and d empirically **from the validation split** and report the values
you used. Do not hardcode 0.51.

### 2.2 Why raw softmax cannot be used as p

A `DiceCELoss`-trained network with `dropout_prob=0.2` is not calibrated, and
Dice loss specifically distorts probabilities because it optimises overlap, not
likelihood. Expect overconfidence. The logistic regression in §5 recalibrates by
construction; produce a reliability diagram to show whether it was needed.

---

## 3. Step 1 — extract per-component data (`predict.py`)

Add a `--save-components` flag to `src/mri3d/predict.py`. While the logits are
still in memory, before they are discarded:

1. `probs = softmax(logits, dim=1)` → `pred = probs.argmax(dim=1)` (unchanged
   behaviour for the mask that gets saved).
2. For each non-background class value v in `paths.GLI_LABELS` / `MEN_LABELS`,
   label the connected components of `pred == v` with `scipy.ndimage.label`.
3. Emit one CSV row per component.

**Do not save probability volumes to disk.** 182×218×182 × 5 classes × fp16 is
~72 MB per case, ~15 GB for the test split. Compute the statistics in memory and
write only the CSV.

### CSV schema — `reports/components_{run}_{split}.csv`

| column | meaning |
|---|---|
| `case_id`, `run`, `split`, `dataset` | provenance |
| `class_name`, `class_value` | e.g. `SNFH`, `2` |
| `comp_id` | component label within (case, class) |
| `n_voxels`, `volume_mm3` | size |
| `centroid_i/j/k` | component centroid in array coords |
| `mean_prob`, `p90_prob`, `max_prob` | softmax prob of `class_value` over the component's voxels |
| `mean_border_prob` | mean prob over the component's 1-voxel dilated boundary (a sharply bounded true lesion differs from a diffuse artefact) |
| `mc_mean_var` | mean MC-dropout variance over the component (null unless `--mc-dropout` used) |
| `dist_to_dominant_mm` | min distance from any component voxel to the dominant mass (see below) |
| `rank_by_volume`, `rank_by_prob` | 1-based rank within (case, class) |
| `n_comps_in_class` | how many components that class had in this case |
| `matched` | bool — matches a scored reference lesion |
| `matched_ref_ids` | which reference lesion(s) |
| `dice_with_matched` | Dice against the union of reference lesions it touches |

Also write `reports/ref_lesions_{run}_{split}.csv`: one row per **reference**
lesion (`case_id`, `class_name`, `ref_id`, `n_voxels`, `detected` bool, `dice`).
This is what gives per-lesion sensitivity, and it is needed to quantify what the
selector costs.

### `dist_to_dominant_mm`

Define the dominant mass as the **largest connected component of whole tumour**,
i.e. of `pred ∈ {1, 2, 3}` (NETC ∪ SNFH ∪ ET — note this excludes RC=4, matching
`GLI_REGIONS` in `metrics.py`). Compute
`scipy.ndimage.distance_transform_edt` on the complement of that mass, with
`sampling=` the voxel spacing, and take the minimum over each component's
voxels. Components inside the dominant mass get 0. GLI is 1 mm isotropic; MEN-RT
is not, so read the spacing from the affine rather than assuming.

### Matching rule — reuse, do not reimplement

The definitions of matched / missed / spurious **must** agree exactly with
`lesion_wise_dice()` in `src/mri3d/metrics.py`:

- reference lesions below `MIN_LESION_VOXELS` (50) are not scored,
- predicted components below 50 voxels are never counted as false positives,
- a predicted component matches if it overlaps ≥1 voxel of a scored reference
  lesion, and a reference lesion is scored against the **union** of all
  predicted components touching it.

Refactor `lesion_wise_dice()` to expose this matching as a reusable function
(e.g. `match_components(pred, ref, min_voxels)` returning the per-component
match table) and have both `lesion_wise_dice()` and the new extraction call it.
**`lesion_wise_dice()` must keep returning identical values** — `tests/test_metrics.py`
must still pass unchanged.

### Acceptance check (do this, it will catch bugs)

Totals of `matched == False` components in the new CSV, per class, must
reconcile exactly with the `lesions_false_*` column totals in the existing
`reports/scores_gli_main_test.csv`:

| class | expected false positives | expected missed |
|---|---|---|
| NETC | 89 | 25 |
| SNFH | 192 | 32 |
| ET | 94 | 29 |
| RC | 87 | 10 |

(TC and WT are derived regions, not separately predicted classes — they are not
part of this reconciliation, and they cannot be filtered independently.)

Write this as an actual test or assertion, not a manual eyeball.

## 4. Step 2 — MC dropout (optional flag, high expected value)

Add `--mc-dropout T` (default 0 = off) to `predict.py`. The model already has
`dropout_prob=0.2`. At inference, keep the model in `eval()` but re-enable only
the dropout modules:

```python
for m in model.modules():
    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout3d)):
        m.train()
```

Run T forward passes, accumulate mean and variance with Welford's algorithm so
you never hold T volumes in memory, and record `mc_mean_var` per component.
Spurious components are expected to show higher disagreement than real ones.
Costs T× inference, no retraining. Use T=8.

## 5. Step 3 — the separation diagnostic (this decides whether to continue)

New module `src/mri3d/select.py`. First job: plot the distributions of each
feature for matched vs unmatched components, from the **validation** CSV:
`dist_to_dominant_mm`, `mean_prob`, `p90_prob`, `mean_border_prob`,
`log(n_voxels)`, `mc_mean_var`, `rank_by_prob`. Overlaid histograms or violins,
one panel per feature, faceted by class. Save to `reports/component_separation.html`
(follow the self-contained-HTML pattern in `heatmap_page.py` — data URIs, opens
from disk without a server) or as PNGs.

Report the AUC of each single feature as a discriminator.

**If no feature separates the two populations, stop and say so.** That is a
legitimate result and it belongs in `reports/results.md` next to the size-sweep
negative result. Do not proceed to fit a model on features that carry no signal.

## 6. Step 4 — the selector

If there is signal:

- Fit **logistic regression** (sklearn, standardized features, `class_weight` if
  the classes are very imbalanced) on **validation components only**.
- Features: the ones above, plus class as a categorical. Prefer one shared model
  with a class indicator over four per-class models unless per-class clearly
  wins on validation — fewer parameters, and the coefficients stay readable.
- Keep the model small and interpretable. Do not reach for gradient boosting
  unless logistic regression demonstrably underfits; the coefficients are part
  of the deliverable.
- Threshold at p\* computed from validation (§2.1).
- Produce a **reliability diagram** (predicted vs observed probability that a
  component is real) for both raw `mean_prob` and the fitted posterior, on
  validation. Save to `reports/reliability_{run}.png`.

Then apply the fitted selector to the **test** split **exactly once**, wire it
into `postprocess.py` as an alternative to `filter_small`, and report:

- lesion-wise and volumetric Dice per class, before → after
- false positives and missed lesions per class, before → after
- the same numbers for the existing per-class size-threshold baseline, so the
  three approaches sit side by side
- per-lesion sensitivity and false-positives-per-case, before → after

Save to `reports/selector_{run}.json` — coefficients, feature means/stds, p\*,
the L and d used to derive it, and all of the above.

### Discipline — non-negotiable

- Fit and threshold on **validation only**. Apply to test **once**.
- Do not iterate on test scores. If you find yourself wanting a second test run
  with different settings, stop and report the first one.
- The previous experiment saw validation overstate the test gain **threefold**.
  Expect the same haircut and say so in the write-up rather than presenting the
  validation number as the result.

## 7. Step 5 — honest reporting of the trade-off

The selector will delete genuinely distant real lesions in multifocal cases, and
lesion-wise Dice **rewards** that trade because there are far more clean cases
than multifocal ones. This must be stated, not hidden.

Using `reports/ref_lesions_{run}_{split}.csv`, quantify both sides:

- how many test cases have a scored reference lesion more than *d* mm from the
  dominant mass,
- how many of those lesions the selector deletes,
- the metric gained overall versus the sensitivity lost on those cases.

Write it into `reports/results.md` as an explicit trade-off, in the voice of the
existing report — which states negative results plainly and does not oversell.
Note also that MEN-RT is a **radiotherapy planning** dataset where a deleted
satellite lesion means under-treatment, so aggressive spatial filtering is
harder to justify there than for GLI.

## 8. Also add detection metrics

The README already claims "connected components of the segmentation *are* the
detections". Make that real — these need no retraining and reuse the CSVs:

- per-lesion sensitivity,
- false positives per case (currently ~0.95 for SNFH, ~0.80 for WT),
- an FROC-style curve: sweep the selector's posterior threshold from 0 to 1 and
  plot sensitivity against FP/case, with p\* marked on it.

The FROC curve is the figure that makes the whole trade-off legible as a point
on a curve rather than a single number.

## 9. Housekeeping while you are in here

- **Add seeding.** There is currently no `torch.manual_seed`, no
  `monai.utils.set_determinism`, and no `--seed` flag anywhere in `src/mri3d/`.
  Add `--seed` (default 0) to `train.py` and `predict.py` and set both torch and
  numpy seeds plus MONAI determinism. For a project whose argument is
  measurement discipline this is the most visible gap in the repo.
- Add `.pytest_cache/` to `.gitignore`.
- Add `dependencies` to `pyproject.toml` (currently empty, so `pip install -e .`
  yields a package that cannot import).
- Extend `tests/` with: the reconciliation assertion from §3, a test that
  `match_components` agrees with `lesion_wise_dice` on synthetic masks, and a
  test that the selector is a no-op at threshold 0.

## 10. Deliverables

1. `predict.py` with `--save-components` and `--mc-dropout`, probabilities used,
   volumes not written to disk.
2. `metrics.py` refactored to expose `match_components`, existing tests passing
   unchanged.
3. `src/mri3d/select.py` — diagnostic plots, fit, apply, FROC.
4. `postprocess.py` able to apply the selector as an alternative to `filter_small`.
5. `reports/components_*.csv`, `reports/ref_lesions_*.csv`,
   `reports/component_separation.html`, `reports/reliability_*.png`,
   `reports/selector_*.json`.
6. `reports/results.md` updated: a new post-processing section with the derived
   threshold, the three-way comparison, the detection metrics, and the
   multifocal trade-off stated explicitly.
7. Seeding, gitignore, pyproject, tests.

## 11. Style notes

Match the existing codebase, which is unusually well documented — read the
module docstrings in `metrics.py`, `model.py` and `heatmap_page.py` for the
register. Docstrings explain **why** a choice was made, including what the
alternative was and why it loses. Keep that. In particular, the p\* derivation
in §2.1 belongs in a docstring in full, because "the threshold was derived from
the scoring function rather than tuned" is the strongest claim this change makes
and a reader should be able to check it.

Work incrementally: get §3 and its reconciliation check green before touching
§5. Show me the separation plots before fitting anything.
