# Results

One SegResNet (4.7M parameters), trained on one RTX 3070 Ti for 8.3 hours. No
ensembling, no test-time augmentation. Scored once on 203 held-out patients
that nothing in training, checkpoint selection, or post-processing tuning ever
touched.

## Glioma (BraTS 2024 post-treatment), test split

Lesion-wise Dice is the metric the challenge ranks on. Volumetric Dice is the
familiar overlap ratio, shown because the difference between the two columns is
the most interesting thing here.

| Region | Lesion-wise | + selector | Volumetric | Cases | BraTS best | BraTS median |
|---|---|---|---|---|---|---|
| NETC | 0.478 | **0.488** | 0.482 | 96 | 0.789 | 0.730 |
| SNFH | 0.533 | **0.618** | 0.851 | 203 | 0.876 | 0.840 |
| ET | 0.625 | **0.650** | 0.766 | 154 | 0.763 | 0.712 |
| RC | 0.655 | **0.696** | 0.725 | 169 | 0.715 | 0.672 |
| TC | 0.605 | **0.628** | 0.739 | 158 | 0.750 | 0.692 |
| WT | 0.561 | **0.642** | 0.861 | 203 | 0.879 | 0.846 |
| **mean** | **0.576** | **0.620** | | | | |

The "+ selector" column is the lesion selector described below, fitted on
validation and applied to this split once. The earlier size-threshold
post-processing reached 0.589 and is kept in the comparison table rather than
in this one.

Challenge figures are from the BraTS 2024 validation-phase leaderboard (188
scored submissions, Synapse table `syn61779417`). They are not strictly
comparable: those are validation-phase scores that teams could resubmit against
repeatedly, from ensembles with far more compute. The column is there for scale,
not for a claim of parity.

Validation Dice peaked at 0.7026 (epoch 80 of 100).

### The gap between the two metrics is the finding

SNFH scores 0.851 volumetric and 0.533 lesion-wise. WT scores 0.861 and 0.561.
A 0.30 spread between two metrics on the same masks is not noise - it is a
specific, diagnosable failure.

The cause is false-positive *lesions*: 192 spurious SNFH components and 162
spurious WT components across 203 cases, roughly one per case. The model finds
the main tumour well and then marks additional tissue elsewhere. Volumetric
Dice hardly registers this, because the extra components are small relative to
the true mass. Lesion-wise Dice scores each connected component separately and
charges a full zero for every invented one.

Reporting only volumetric Dice would have hidden this completely. That is the
argument for the challenge's choice of metric, visible in our own numbers.

## Post-processing, attempt 1: size. A hypothesis that mostly failed

Predicting that removing small connected components would be the single biggest
available win, I swept a minimum-lesion-size threshold. It was not.

**A single global threshold does nothing.** Best on validation was +0.009; on
test it was **-0.0006**. The invented lesions are not specks - the scoring
already ignores components under 50 voxels - so any threshold large enough to
remove them also deletes true lesions. At 100 voxels, false positives fell
796 -> 477 while missed lesions rose 140 -> 340.

**Per-class thresholds help, modestly.** Selected on validation
(NETC 0, SNFH 400, ET 100, RC 400 voxels), they gained +0.039 on validation and
**+0.0126 on test** - about a third of what validation promised. SNFH (+0.042)
and WT (+0.036) improved; ET (-0.011) and TC (-0.012) got slightly worse,
because the ET threshold that helped on validation did not transfer.

The threshold was chosen on validation and applied to test once. Choosing it on
test would have turned a held-out score into a fitted one, and - given the
validation gain overstated the test gain threefold - would have produced a
number that looked better and meant less.

## Post-processing, attempt 2: a learned lesion selector

### Why size was the only lever

`predict.py` did `pred = logits.argmax(dim=1)` and saved a `uint8` mask. Every
probability the network produced was discarded on that line, so the
post-processing stage only ever saw geometry. The size sweep was not a badly
chosen experiment; it was the only experiment the pipeline allowed.

`predict.py --save-components` keeps the evidence instead: it summarises the
softmax over each predicted connected component while the logits are still in
memory and writes one CSV row per component. No probability volume is written -
182x218x182 x 5 classes at fp16 is ~72 MB per case and ~15 GB for one split,
where the per-component statistics are about 1 kB per case.

### The threshold is derived, not swept

For a case whose other lesion-wise entries number N and average L, take one
marginal predicted component with `p` = probability it is real and `d` = the
Dice it scores if it is. If it is real, keeping it turns a 0 entry into `d`, so
the case mean gains `d/N`. If it is spurious, keeping it appends a zero, so the
mean loses `L/(N+1)`. Break-even is where those cancel:

```
  p · d/N  =  (1 − p) · L/(N+1)        expected gain = expected loss
  p · d    =  (1 − p) · L              N/(N+1) → 1 for N of any size
  p · (d + L) = L

  p* = L / (L + d)
```

Two consequences are worth stating because both are easy to get backwards.
**N cancels**, so the optimal rule is a flat threshold on the posterior - the
familiar intuition that the third lesion should need more evidence than the
first is real, but it lives in the prior, not in the metric, and a rule that
escalated the threshold as well would count it twice. And **p\* falls as L
falls**, so a case already scoring badly correctly becomes more willing to
gamble on an extra lesion.

Measured on validation: `L` = 0.6191 (mean Dice over the 1152 scored reference
lesions - what an entry is worth when it is not a false positive's zero) and
`d` = 0.4915 (mean Dice of matched components that are *not* the largest in
their case and class, which is the population the threshold actually decides
on). That gives **p\* = 0.5574**. Estimating `d` over every matched component
instead, including the dominant mass that is never near the threshold, would
give 0.6695 and p\* = 0.4804; per-class values range from 0.501 (NETC) to
0.631 (RC). Nothing here was tuned against a score.

### Does anything separate real from spurious components?

Before fitting anything: AUC for discriminating matched from unmatched
components, validation split, 1575 components of at least 50 voxels of which
1027 are real. Values below 0.5 separate just as strongly in the opposite
direction. Panels per feature and class are in
[`component_separation.html`](component_separation.html).

| feature | ET | NETC | RC | SNFH | all |
|---|---|---|---|---|---|
| `p90_prob` | 0.887 | 0.680 | 0.911 | 0.872 | **0.860** |
| `max_prob` | 0.885 | 0.701 | 0.906 | 0.875 | **0.861** |
| `mean_prob` | 0.886 | 0.666 | 0.903 | 0.860 | **0.855** |
| `log_n_voxels` | 0.833 | 0.730 | 0.893 | 0.835 | **0.823** |
| `dist_to_dominant_mm` | 0.422 | 0.506 | 0.402 | 0.241 | **0.333** |
| `mean_border_prob` | 0.255 | 0.445 | 0.333 | 0.238 | **0.319** |
| `rank_by_prob` | 0.335 | 0.451 | 0.232 | 0.214 | **0.282** |
| `mc_mean_var` | 0.208 | 0.423 | 0.215 | 0.192 | **0.225** |

Everything carries signal, and one row deserves a second look: `log_n_voxels`
reaches 0.823, yet the size *threshold* gained almost nothing on test. Size
ranks components well **within** a case and transfers badly **across** cases,
which is exactly the failure a per-component model with other features can
route around. The size experiment was not measuring the wrong thing; it was
using the right thing in the only way a global cut point allows.

MC dropout earns its place. Eight stochastic forward passes - the model already
trains with `dropout_prob=0.2`, so this costs 8x inference and no retraining -
give `mc_mean_var` an AUC of 0.225, i.e. 0.775 read the other way, and it
survives as the fourth-largest coefficient after conditioning on everything
else.

### The raw softmax is not a probability

`mean_prob` ranks well (AUC 0.855) and is useless as a probability. Its
expected calibration error on validation is **0.179**: a `DiceCELoss`-trained
network with dropout has no reason to be calibrated, and the Dice term
specifically optimises overlap rather than likelihood. The result is
overconfidence, piled up against 1.0:

| `mean_prob` | components | actually real |
|---|---|---|
| 0.5 - 0.6 | 89 | 19% |
| 0.6 - 0.7 | 255 | 27% |
| 0.7 - 0.8 | 344 | 49% |
| 0.8 - 0.9 | 426 | 77% |
| 0.9 - 1.0 | 455 | 98% |

The median is 0.825, and only 6 of 1575 components fall below 0.5. Thresholding
this at the derived p\* of 0.557 would have deleted those 6 and changed nothing
else. The raw confidence that actually means "even odds" is about 0.80 — and
there is no way to know that without measuring it, which is the whole reason
the posterior is fitted rather than read off the network.

The fitted posterior has ECE **0.031** on out-of-fold predictions grouped by
case. That recalibration, not extra discrimination, is most of what the
selector contributes: out-of-fold AUC only moves 0.855 -> 0.869. See
[`reliability_gli_main.png`](reliability_gli_main.png).

### The selector

Logistic regression, 12 coefficients, fitted on validation components only.
Standardised coefficients, in logit units per standard deviation:

| feature | coef | reads as |
|---|---|---|
| `log_n_voxels` | **+1.18** | bigger is realer |
| `mean_prob` | **+0.84** | more confident is realer |
| `log1p_dist` | **-0.56** | further from the dominant mass is less real |
| `log_mc_var` | **-0.27** | more MC disagreement is less real |
| `is_SNFH` | +0.26 | SNFH components are likelier real at equal evidence |
| `rank_by_prob` | +0.15 | |
| `mean_border_prob` | -0.12 | sharp edges are realer than diffuse ones |
| `max_prob` | -0.12 | |
| `is_NETC`, `n_comps_in_class`, `is_RC`, `p90_prob` | < 0.03 | |

`class_weight="balanced"` was considered and rejected. It is the usual reflex
for imbalanced data, and it would have broken the whole argument: reweighting
rescales the prior, and thresholding at p\* requires a posterior on the real
prior. Gradient boosting was not tried, because the coefficients above are part
of what this change is for and logistic regression does not appear to underfit.

### Results, three ways

Test split, 203 patients, scored once. "none" is the raw model, "size" is the
per-class thresholds from attempt 1, "selector" is the above at p\* = 0.5574.

| Region | lesion: none | size | **selector** | volumetric: none | selector |
|---|---|---|---|---|---|
| NETC | 0.4777 | 0.4777 | **0.4875** | 0.4821 | 0.4720 |
| SNFH | 0.5329 | 0.5749 | **0.6177** | 0.8514 | 0.8523 |
| ET | 0.6253 | 0.6139 | **0.6500** | 0.7663 | 0.7549 |
| RC | 0.6546 | 0.6751 | **0.6960** | 0.7251 | 0.7120 |
| TC | 0.6050 | 0.5933 | **0.6282** | 0.7391 | 0.7279 |
| WT | 0.5613 | 0.5973 | **0.6419** | 0.8609 | 0.8617 |
| **mean** | **0.5761** | **0.5887** | **0.6202** | | |

**+0.0441 on test, against +0.0126 for the size baseline** - three and a half
times the gain, and unlike the size thresholds it improves every region rather
than trading ET and TC away. Volumetric Dice is roughly unchanged: SNFH and WT
actually rise slightly, the focal classes give up 0.010-0.013. That asymmetry
is the point. The deleted components are small relative to the true mass, so
they barely register volumetrically, which is the same fact that made the
lesion-wise/volumetric gap diagnosable in the first place.

Validation promised more, as usual:

| | validation | test | shortfall |
|---|---|---|---|
| size thresholds | +0.0393 | +0.0126 | 68% |
| **selector** | **+0.0675** | **+0.0441** | **35%** |

The haircut is real and was expected - it is stated here rather than reported
as +0.0675. It is roughly half the size experiment's, which is consistent with
the two separate checks made on validation: out-of-fold posteriors grouped by
case cost only 0.009 sensitivity and 0.084 FP/case against in-sample ones, so
most of what validation overstates is distribution shift between splits, not
the model having memorised its own components.

### Detection metrics

Connected components of the segmentation *are* the detections, which the README
already claimed; these make it true. Counts are over all 203 cases and all four
predicted sub-regions.

| | before | after |
|---|---|---|
| per-lesion sensitivity | 0.9140 | 0.8306 |
| false positives per case | 2.276 | 0.591 |
| false-positive lesions | 462 | 120 |
| detected reference lesions (of 1116) | 1020 | 927 |

Per class, sensitivity and FP/case before -> after:

| class | reference lesions | sensitivity | FP/case | false positives | missed lesions |
|---|---|---|---|---|---|
| NETC | 181 | 0.862 -> 0.762 | 0.438 -> 0.207 | 89 -> 42 | 25 -> 43 |
| SNFH | 424 | 0.924 -> 0.830 | 0.946 -> 0.138 | 192 -> 28 | 32 -> 72 |
| ET | 319 | 0.909 -> 0.812 | 0.463 -> 0.143 | 94 -> 29 | 29 -> 60 |
| RC | 192 | 0.948 -> 0.927 | 0.429 -> 0.103 | 87 -> 21 | 10 -> 14 |

NETC is the one class where the trade is questionable: it gives up 10 points of
sensitivity to halve its false positives, and its lesion-wise Dice moves only
+0.0098 as a result. It is also the class where the raw confidence separates
worst (AUC 0.666 against 0.86-0.90 elsewhere), so the selector has least to go
on. A per-class p\* would have set NETC's bar at 0.501 rather than 0.557; that
was measured and deliberately not used, because four thresholds chosen per
class is the kind of freedom that stops a derived number from being derived.

The selector drops 531 of 1436 decidable test components. A 3.9x reduction in
false positives for 8 points of sensitivity is a good trade *for this metric*,
and [`froc_gli_main.png`](froc_gli_main.png) is the honest way to see it: the
whole curve, with the derived p\* marked. It lands close to the knee on both
splits without having been tuned to either, and the validation and test curves
nearly coincide - the selector generalises, the operating point is simply worth
slightly less on test.

### How much is left on the table

A perfect selector - one that reads the labels and keeps exactly the matched
components - is the ceiling for this approach. It is not a model, but it is the
number that decides whether more feature engineering here is worth doing.

| Region | none | selector | perfect selection |
|---|---|---|---|
| NETC | 0.4777 | 0.4875 | 0.6057 |
| SNFH | 0.5329 | 0.6177 | 0.6987 |
| ET | 0.6253 | 0.6500 | 0.7186 |
| RC | 0.6546 | 0.6960 | 0.7416 |
| TC | 0.6050 | 0.6282 | 0.7009 |
| WT | 0.5613 | 0.6419 | 0.7164 |
| **mean** | **0.5761** | **0.6202** | **0.6970** |

The selector captured **36% of the available headroom** (+0.0441 of +0.1209),
leaving 0.077 reachable without retraining anything. The gap is widest exactly
where the features are weakest: NETC has both the largest remaining margin
(0.118) and the worst single-feature AUC (0.666 against 0.86-0.90 elsewhere).
The binding constraint is the component-level AUC of 0.869, and the most
obvious unused evidence is that the selector never looks at the MRI - it sees
only the prediction, so it cannot tell a genuine FLAIR hyperintensity from a
confident blob in normal-appearing white matter.

Separately, 96 of 1116 reference lesions (8.6%) are never detected at all, which
caps per-lesion sensitivity at 0.914 no matter how good selection gets. Those
misses are small: 21% of lesions between 50 and 100 voxels are missed against
0% above 10k, and the median missed lesion is 115 voxels against 1865 for a
detected one. Raising that ceiling needs a better segmentation model, not
better post-processing.

### The failures are two different problems, counted as one

`lesion_wise_dice` charges a full zero for a predicted component that matches
no reference lesion, and never asks why it is unmatched. Asking turns out to
matter more than anything else measured here. `mri3d.failure_anatomy` takes
every false positive on the test split and reports what the reference says is
actually underneath it:

| predicted class | n | the reference calls that tissue... | mean tumour-voxel fraction |
|---|---|---|---|
| **NETC** | 89 | SNFH 37%, RC 31%, ET 19%, NETC 9%, **background 3%** | **0.96** |
| ET | 94 | SNFH 36%, **background 33%**, RC 23%, ET 4%, NETC 3% | 0.64 |
| RC | 87 | **background 48%**, SNFH 30%, NETC 22% | 0.51 |
| SNFH | 192 | **background 82%**, RC 8%, ET 4%, SNFH 3%, NETC 3% | 0.19 |

**229 of the 462 false-positive lesions sit mostly on real tumour.** They are
not invented. The model found the tumour and named the sub-region wrong. For
NETC this is essentially the whole story: 97% of its "spurious" components are
on tissue the reference calls tumour, and only three of eighty-nine are on
background. The same reading in reverse, for the lesions we never detect:

| reference class | n | what was predicted there |
|---|---|---|
| NETC | 25 | ET 36%, SNFH 24%, RC 24%, **nothing 16%** |
| RC | 10 | NETC 50%, **nothing 30%**, SNFH 10%, ET 10% |
| SNFH | 32 | **nothing 97%**, ET 3% |
| ET | 29 | **nothing 86%**, SNFH 7%, RC 7% |

A third of the "missed" lesions were found and misnamed. And lesion-wise Dice
charges for that twice - once for the spurious component, once for the
reference lesion it left unmatched. Validation says the same thing: 97% of NETC
false positives on real tumour, 89% of SNFH false positives on background.

### Attributing the deficit

Three oracles, each fixing exactly one thing. `relabel` renames every
false-positive component that sits mostly on reference tumour to that
sub-region and changes nothing else; `delete` removes it instead.

| region | none | relabel | delete | both |
|---|---|---|---|---|
| NETC | 0.4777 | **0.6391** | 0.6057 | 0.6473 |
| SNFH | 0.5329 | 0.5457 | **0.6987** | 0.6921 |
| ET | 0.6253 | 0.6669 | **0.7186** | 0.7042 |
| RC | 0.6546 | 0.7256 | 0.7416 | **0.7911** |
| TC | 0.6050 | 0.6716 | 0.7009 | **0.7080** |
| WT | 0.5613 | 0.5743 | 0.7164 | **0.7198** |
| **mean** | **0.5761** | **0.6372** | **0.6970** | **0.7104** |

Renaming alone is worth **+0.0611**, and for NETC it beats deleting outright
(+0.161 against +0.128). The two fixes are largely independent - together they
reach 0.7104 against 0.6970 for deletion alone.

Validation agrees on every claim that matters, on a cohort where the mix is
different - 38% of its false positives are renameable against 50% on test, so
it has proportionally more genuinely invented lesions:

| split | none | relabel | delete | both | renameable |
|---|---|---|---|---|---|
| validation | 0.5598 | 0.6152 | 0.7135 | 0.7176 | 207/548 (38%) |
| test | 0.5761 | 0.6372 | 0.6970 | 0.7104 | 229/462 (50%) |

and NETC behaves the same way on both, in both metrics:

| | lesion-wise: none → relabel → delete | volumetric: none → relabel → delete |
|---|---|---|
| validation | 0.5438 → **0.6883** → 0.6681 | 0.5649 → **0.6145** → 0.5774 |
| test | 0.4777 → **0.6391** → 0.6057 | 0.4821 → **0.5856** → 0.5254 |

The asymmetry that matters is in the volumetric column, which deletion cannot
win by construction:

| region | none | relabel | delete |
|---|---|---|---|
| NETC | 0.4821 | **0.5856** | 0.5254 |
| RC | 0.7251 | **0.7738** | 0.7340 |
| WT | 0.8609 | 0.8704 | 0.8687 |

**Relabelling improves both metrics; deleting trades one for the other.** The
selector as built can only delete, so none of the +0.0611 is available to it,
and on NETC it is applying the wrong operation - which is exactly why NETC
gains least from it (+0.0098) and is the one class whose volumetric Dice the
selector makes worse.

### What an invented lesion looks like

For the classes where the false positives really are invented, they have a
consistent signature. Measured on validation, comparing matched against
unmatched components:

| class | depth from brain surface (real → spurious) | FLAIR contrast (real → spurious) |
|---|---|---|
| SNFH | 34.5 → **19.3** mm | +1.21 → **+0.89** |
| ET | 25.9 → **15.6** mm | +0.48 → **+0.25** |
| RC | 23.3 → **13.6** mm | −1.65 → −0.88 |
| NETC | 24.2 → 24.4 mm | +0.06 → +0.06 |

Invented lesions are shallow and low-contrast - the partial-volume rind near
the cortical surface. **NETC is the single exception, and it is exactly the
class whose false positives are real tumour**, so its "spurious" components are
as deep and as contrasted as its genuine ones. Two independent measurements,
the same conclusion.

They are also concentrated: 38 of 203 test cases produce no false positive at
all, and the worst 20% of cases hold 52% of them.

### Letting the selector look at the MRI: it did not help

The obvious next move, given that gap, was that the selector never sees the
scan. `mri3d.image_features` adds fourteen features that read the images
directly: per-modality intensity standardised over the brain, contrast against
a 3-voxel shell around each component, contrast against the mirrored location
in the other hemisphere, `z_t1c - z_t1n` (the definition of enhancement),
`z_t2f - z_t2w` (which separates a CSF-filled cavity from oedema), and depth
into the brain. No retraining, one pass over the images per split.

Out-of-fold on validation, grouped by case:

| feature set | all | ET | NETC | RC | SNFH | parameters |
|---|---|---|---|---|---|---|
| prediction only | **0.8685** | 0.8717 | 0.7084 | 0.9136 | 0.8763 | 12 |
| + the 4 strongest image features | 0.8706 | 0.8753 | 0.7099 | 0.9144 | 0.8784 | 16 |
| + all 14 | 0.8682 | 0.8702 | 0.6983 | 0.9180 | 0.8754 | 26 |
| **image features only** | 0.7280 | 0.6820 | 0.5356 | 0.8104 | 0.7583 | 17 |
| raw `mean_prob` alone | 0.8549 | | | | | 1 |

**+0.002 AUC at best, and the full set is worse than not bothering.** NETC, the
class with the most headroom, moves 0.7084 -> 0.7099. Calibration degrades
(ECE 0.031 -> 0.037). Nothing here justifies a second application to test.

The features are not broken - individually they behave exactly as the physics
says they should. `z_t2f_contrast` is the best single image feature for SNFH
(AUC 0.674: oedema is FLAIR-bright against its surroundings), and
`flair_minus_t2` reads 0.331 for RC, i.e. 0.669 inverted, because a resection
cavity is FLAIR-dark and T2-bright. On their own they reach 0.728, which is
real signal.

They are **redundant**. The segmentation network already consumed those
intensities, through a receptive field far larger and a function far richer
than a mean z-score, and `mean_prob` alone scores 0.855. Asking a logistic
regression to re-derive radiology from four summary statistics cannot add to
what a trained 3D CNN extracted from the same voxels.

That is worth stating as a general result, because the intuition that "the
model should look at the image" is a strong one: **what a post-hoc selector
adds is the information the segmentation network could not see, not a second
opinion on what it could.** The three features carrying the most weight are
exactly of that kind - component size, distance from the dominant mass, and
MC-dropout disagreement are all global or ensemble properties that no single
forward pass through a patch-based CNN has access to. Intensity is not.

The features remain in the repository as `mri3d.image_features` and are opt-in
behind `mri3d.select --fit --image-features`, so the result can be rechecked
rather than taken on trust.

### What it costs: multifocal cases

Lesion-wise Dice **rewards** deleting a genuinely distant real lesion, because
clean cases far outnumber multifocal ones: the zero saved across a hundred
single-focus cases outweighs the zero created on ten multifocal ones. That is a
property of the metric, not evidence that the deletion was correct, so both
sides are given separately here.

Test split, reference lesions by distance from the predicted dominant mass:

| distance | scored lesions | cases with one | detections deleted | cases hurt | those cases | all other cases |
|---|---|---|---|---|---|---|
| > 10 mm | 85 | 52 | 25 | 22 | **+0.0250** | +0.0633 |
| > 20 mm | 46 | 28 | 11 | 9 | **+0.0209** | +0.0587 |
| > 40 mm | 24 | 17 | 5 | 5 | **-0.0154** | +0.0598 |

Cases carrying a reference lesion more than 40 mm from the main tumour are
**made worse** by the selector, by -0.0154 on average, while everything else
gains +0.0598. Seventeen test cases are in that group and five of them lose a
real detection. In aggregate the metric approves; for those five patients the
model now misses a lesion it previously found.

This matters more for one dataset than the other. MEN-RT is a **radiotherapy
planning** cohort, where a deleted satellite lesion means a volume that does
not get irradiated. Spatial filtering that is defensible for a glioma
segmentation benchmark is much harder to justify there, and the selector has
not been applied to MEN-RT for that reason.

### A reproducibility bug found while adding seeding

There was no `torch.manual_seed`, no `monai.utils.set_determinism` and no
`--seed` anywhere in `src/mri3d/` until now, which was the most visible gap in
a project whose argument is measurement discipline. Adding `mri3d.seeding`
briefly made things worse: `set_determinism` turns `cudnn.benchmark` off, the
first version turned it back on to recover training throughput, and cuDNN's
autotuner picks its convolution algorithm by timing candidates against the
workspace that happens to be free.

Measured on this card, the same seeded sliding-window inference run with 3 GB
of VRAM already occupied selected a different algorithm and produced different
numbers - softmax probabilities agreeing to about four decimal places,
volumetric Dice moving by up to 0.0095 on a case, and two per-class
false-positive counts shifting by one as components crossed the 50-voxel floor.
Two runs under equal memory conditions agreed exactly, which is why it was not
obvious. With autotuning off the experiment is bit-identical at 0 GB and 3 GB
occupied, and it reproduces the original unseeded predictions exactly: the
regenerated `scores_gli_main_{val,test}.csv` are byte-for-byte the previously
committed files, and re-running the size sweep reproduced
`postprocess_sweep_gli_main.json` byte-for-byte too. So the "none" and "size"
columns above are the numbers this repository has always reported, not a
re-measurement that happens to be close.

`mri3d.predict` now seeds with `benchmark=False`; training keeps it on, where
the throughput is worth more than bit-reproducibility of a checkpoint. The
general lesson is narrower than "seed everything": a seed fixes the random
draws, and on a GPU the random draws were never the part that was drifting.

### Test-time augmentation: real on its own, redundant with the selector

`predict.py --tta` averages softmax probabilities over all 8 combinations of
flipping the three spatial axes - the flips the model saw in training - and
undoes each flip before averaging (`tests/test_tta.py`). Same checkpoint,
nothing retrained. It was run as a separate run name, `gli_main_tta`, through
the whole pipeline: predict val and test, size sweep on val, selector refit on
val, test scored once.

| Region | none | + TTA | selector | TTA + selector |
|---|---|---|---|---|
| NETC | 0.478 | 0.496 | 0.488 | 0.480 |
| SNFH | 0.533 | 0.552 | 0.618 | 0.623 |
| ET | 0.625 | 0.644 | 0.650 | 0.647 |
| RC | 0.655 | 0.675 | 0.696 | 0.702 |
| TC | 0.605 | 0.625 | 0.628 | 0.627 |
| WT | 0.561 | 0.577 | 0.642 | 0.644 |
| **mean** | **0.576** | **0.595** | **0.620** | **0.621** |

On its own TTA is worth **+0.019**, on every region - squarely in the range it
usually gives. Stacked on the selector it is worth **+0.0006**, which is noise.
The reason is in the false-positive counts: TTA brings invented lesions down
from 2.28 to 1.95 per case before any filtering, and those are lesions the
selector was already deleting. Two methods, one error. What survives the
overlap is detection: at the same ~0.6 false positives per case, per-lesion
sensitivity after the selector is 0.841 with TTA against 0.831 without. The
refitted selector derived almost the same threshold (p\* = 0.557 both times).

The headline stays on the single-pass model: an 8x inference cost for +0.0006
is not a trade worth making. Note that the test split was scored for both
variants and the better one was *not* adopted on that basis.

## Meningioma (BraTS 2024 MEN-RT), test split

75 held-out patients, single T1c modality, one binary gross-tumour-volume mask.

| | 300 epochs (`men_300`) | 120 epochs (`men_long`) | BraTS best | BraTS median |
|---|---|---|---|---|
| Lesion-wise Dice | **0.589** (median 0.732) | 0.509 (median 0.462) | 0.849 | 0.794 |
| Volumetric Dice | **0.660** (median 0.816) | 0.626 (median 0.786) | | |
| False lesions per case | 0.53 | 0.57 | | |
| Cases with any false lesion | 16 | 27 | | |

Validation Dice 0.7317 at epoch 270 of 300 (`men_long`: 0.6695 at epoch 120).

`men_300` is the same model, data, patch size and validation schedule as
`men_long`, with only the epoch count changed - item 1 of the list at the end
of this report, taken. The test split has now been scored twice, once per run;
both numbers are in the table rather than only the better one.

### More training cleaned up the good cases, not the failures

| Volumetric Dice | `men_300` | `men_long` |
|---|---|---|
| 0.9 - 1.0 | 16 | 14 |
| 0.8 - 0.9 | 25 | 19 |
| 0.5 - 0.8 | 16 | 20 |
| 0.2 - 0.5 | 5 | 9 |
| **0.0 - 0.2** | **13** | **13** |

The distribution is still bimodal, and the gain is all on one side of it. 41 of
75 cases now score above 0.8 (was 33), and the lesion-wise *median* jumped
0.462 -> 0.732, because the number of cases carrying at least one invented
lesion fell from 27 to 16. Per case, 34 improved by more than 0.01 lesion-wise,
14 got worse, 27 were unchanged.

The near-total misses did not move: still 13. Three of the old ones were
fixed, three new ones appeared, and **10 cases fail under both runs**. Longer
training is not going to reach them. Whatever those 10 have in common is now
the clearest target in the project, and they are visible in the prediction
review page, which samples across the score range.

### Preprocessing

Resampling to 1 mm isotropic once (`mri3d.preprocess`) rather than per epoch
halved epoch time, 317s -> 166s, and removed the memory pressure that had
killed an earlier run. Training 40 epochs reached 0.456; 120 epochs reached
0.670, so the first schedule was simply too short - worth +0.21 Dice.

## What would improve these numbers

In rough order of expected value:

1. ~~**More training.**~~ Done for meningioma: 120 -> 300 epochs moved
   lesion-wise Dice 0.509 -> 0.589 (see the meningioma section). Glioma
   converged at epoch 80, so it is not expected to gain the same way.
2. **A relabeller, not just a deleter.** The largest measured, un-taken win:
   renaming the 229 false-positive components that sit on real tumour is worth
   +0.0611 lesion-wise and improves volumetric Dice at the same time, and it is
   entirely outside what the current selector can express. The evidence for
   which name is right is already in the network - it is the runner-up class in
   the softmax - but `predict.py --save-components` records the probability of
   the *assigned* class only, so the margin to second place is thrown away. It
   is a small change to keep all five, and then the same
   fit-on-validation-apply-once discipline applies.
3. **Flip disagreement as a selector feature.** Flip-*averaging* is done (see
   "Test-time augmentation" above): +0.019 on its own, +0.0006 once the
   selector runs, because both remove the same invented lesions. What has not
   been tried is the *disagreement* across the eight flips as a per-component
   feature: it probes an invariance the model was actually trained for
   (`RandFlipd` on all three axes), where MC dropout only perturbs weights. It
   is the kind of information the image features turned out not to be,
   because a single forward pass cannot contain it.
4. **Attack the false positives at the source** rather than by filtering. The
   selector is still post-hoc: it recovers +0.0441 of the +0.121 that perfect
   component selection would give, and it pays for it with 8 points of
   per-lesion sensitivity. Deep supervision, or a loss term that penalises
   spurious components, would address the cause instead of the symptom.
5. **Ensembling** across folds - the standard reason challenge entries score
   where they do, and disagreement between independently trained models is a
   stronger selector feature than either MC dropout or flip consistency.
6. **NETC is the weakest class** (0.478) and the rarest (42% of cases, 0.047%
   of voxels). It is weakest for the selector too - out-of-fold AUC 0.708
   against 0.88-0.91 for the others - and it carries the largest remaining
   margin to perfect selection (0.118). Class-balanced sampling already rescued
   it once during training, from 0.487 to 0.675 on validation; more aggressive
   oversampling may help further.
7. **Diagnose the meningioma failures.** Not a glioma item, but the largest
   single number available anywhere in this project. MEN-RT has only 0.53
   false-positive lesions per case, so the selector has little to delete
   there, and 13 of 75 test cases still score below 0.2 volumetric after
   300 epochs - 10 of them the same cases that failed after 120. (Figures
   below are from the 120-epoch analysis.) They skew small (median GTV
   2795 voxels against 11138 for the rest) but four of them exceed 10000
   voxels, so "too small to see" is not the whole explanation.
   `reports/lesion_3d_*.html` is the tool for looking at them.
