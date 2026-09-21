# Brain tumour segmentation on BraTS 2024

3D brain tumour segmentation and lesion detection on multi-parametric MRI, trained on a
single 8 GB consumer GPU (RTX 3070 Ti). Built on the BraTS 2024 challenge cohorts:
1350 labelled post-treatment glioma cases and 500 meningioma radiotherapy cases.

**The imaging data is not in this repository** and cannot be — it is CC-BY-NC and
access-controlled through Synapse. Everything here regenerates from it.

## Results

Each model scored once on a held-out, patient-level split. Full tables and
discussion in [`reports/results.md`](reports/results.md).

**Glioma** — 203 patients, four tumour sub-regions:

| Region | Lesion-wise Dice | Volumetric Dice |
|---|---|---|
| NETC | 0.478 | 0.482 |
| SNFH | 0.575 | 0.851 |
| ET | 0.614 | 0.766 |
| RC | 0.675 | 0.725 |
| **mean** | **0.589** | |

**Meningioma** — 75 patients, binary gross tumour volume: lesion-wise 0.509,
volumetric 0.626 (median 0.786).

Two things in those numbers are worth more than the numbers themselves:

- **The 0.30 gap between the glioma columns.** SNFH scores 0.851 volumetric and
  0.575 lesion-wise, because the model invents roughly one spurious lesion per
  case. Volumetric Dice barely notices; lesion-wise charges a full zero for each.
  Reporting only the familiar metric would have hidden the defect entirely.
- **The meningioma mean describes no actual case.** 33 of 75 cases score above
  0.8 and 13 score below 0.2 — the model is usually good and occasionally blind.
  The median, 0.786, is the honest summary, and those 13 failures are the real
  target for the next iteration.

## What's here

| | |
|---|---|
| **Results** | `reports/results.md` — test scores, next to the challenge leaderboard |
| **Tumour atlas** | `reports/tumour_atlas.html` — where tumours occur across all 1350 cases, in three planes |
| **Prediction review** | `reports/predictions_*.html` — what the model called a tumour, vs. what a radiologist drew |
| **Training curves** | `reports/curves_*.html` — loss and per-class Dice per epoch |
| **Leakage audit** | `reports/duplicates.csv` — found 3 pairs of identical scans under different patient IDs |
| **Case index** | `reports/case_index.csv` — per-case class volumes, tumour centroid, bbox, geometry |
| **Design note** | `docs/patch-vs-full-volume.md` — measured, not assumed |

## Pipeline

```bash
python -m mri3d.scan --which all        # one pass over ~45 GB: stats, heatmaps, fingerprints
python -m mri3d.duplicates              # leakage audit -> reports/duplicates.csv
python -m mri3d.splits                  # patient-level train/val/test -> configs/splits.json
python -m mri3d.template                # mean brain for the atlas backdrop
python -m mri3d.heatmap_page            # -> reports/tumour_atlas.html

python -m mri3d.train --overfit 5 --epochs 250    # correctness gate: Dice must approach 1.0
python -m mri3d.train --epochs 300                # the real run
python -m mri3d.predict --run <run> --split test --save-masks
python -m mri3d.case_viewer --run <run> --split test
```

## Method

- **3D SegResNet** (residual encoder-decoder CNN). Transformer variants don't beat a
  well-tuned CNN at this data scale and don't fit in 8 GB.
- **Dice + cross-entropy loss.** Cross entropy alone is optimised by predicting background
  everywhere — tumour is ~1% of voxels, so that scores 99% "accuracy". Dice is a per-class
  ratio, so a small tumour counts as much as the huge background.
- **Class-balanced patch sampling.** Uniform random patches would rarely contain the rare
  classes; NETC appears in only 42% of cases.
- **Softmax over mutually-exclusive classes**, because BraTS 2024 scores the four sub-regions
  directly. Pre-2024 BraTS code assumes overlapping regions (WT ⊃ TC ⊃ ET) and sigmoid outputs
  — that does not apply here.
- **Lesion-wise Dice** as the headline metric, matching the challenge: each connected tumour
  is scored separately, so a missed satellite lesion costs a full zero. This is also the
  detection metric — connected components of the segmentation *are* the detections.

## Layout

```
mri_3d/
├── src/mri3d/                      the package (scan, splits, duplicates, data, model, train, …)
├── reports/                        committed outputs: atlas, prediction review, audits, scores
├── configs/splits.json             patient-level split manifest
├── docs/                           design notes and BibTeX citations
├── tools/brats2024_mlcubes/        official MLCube prep/eval configs
├── outputs/                        checkpoints, predictions, large arrays (gitignored)
└── data/                           the imagery (gitignored, never committed)
    ├── raw/                        original Synapse downloads
    ├── brats2024_gli/train|val/    glioma: 1350 labelled, 188 unlabelled
    ├── brats2024_men_rt/train/     meningioma: 500 labelled
    ├── samples/                    3-case "fastlane" sets for smoke tests
    └── metadata/                   clinical spreadsheets, Synapse manifest
```

## Datasets

### BraTS 2024 MEN-RT (meningioma, radiotherapy planning) — `data/brats2024_men_rt/train/`
- 500 cases, one folder per case: `BraTS-MEN-RT-XXXX-1/`
  - `*_t1c.nii.gz` — contrast-enhanced T1 (single modality, native space, not skull-stripped)
  - `*_gtv.nii.gz` — binary gross tumour volume mask (0 = background, 1 = GTV)
- ⚠️ **Unlike GLI, this data is raw**: shapes vary (256×256×124, 512×512×136, …) and voxels are
  anisotropic and inconsistent (1.02×1.02×1.5 mm, 0.37×0.37×1.1 mm, …). You must resample to a
  common spacing yourself; a model trained on GLI geometry will not transfer without it.
- Case `BraTS-MEN-RT-0402-1` was replaced with the organisers' corrected release (Feb 2025).
  The original training-zip version is still inside `data/raw/BraTS2024-MEN-RT-TrainingData.zip`.
- Clinical data: `data/metadata/Meningioma radiotherapy supplementary clinical data ... v2.xlsx`

### BraTS 2024 GLI (post-treatment adult glioma) — `data/brats2024_gli/`
- Case folders `BraTS-GLI-XXXXX-YYY/` with 4 co-registered, skull-stripped 1 mm modalities:
  `-t1n` (T1 native), `-t1c` (T1 contrast), `-t2w` (T2), `-t2f` (T2-FLAIR), plus `-seg` in training.
- Label map (2024): 0 background, 1 NETC (non-enhancing tumour core), 2 SNFH (surrounding
  FLAIR hyperintensity / oedema), 3 ET (enhancing tumour), 4 RC (resection cavity).
  Verified on a 150-case sample: **not every case contains every class** (~30% have all four,
  ~27% lack NETC, label 2/SNFH is present in all). Per-case Dice must handle empty classes
  (BraTS scores an empty prediction on an empty reference as 1, not 0).
- Geometry is uniform: **182×218×182 at 1 mm isotropic** (SRI24 space), already co-registered,
  resampled and skull-stripped. Note this is *not* the 240×240×155 of BraTS 2021 and earlier,
  so older BraTS code with hardcoded shapes or crops will need adjusting.
- `train/` has 1350 cases with `-seg` labels (zip folder was `training_data1_v2`).
- `val/` has 188 cases **without** segmentations (was for leaderboard submission only).
- Case IDs are `BraTS-GLI-<patient>-<timepoint>`: 1350 cases come from only 613 patients, so
  split train/val/test **by patient ID**, not by case, to avoid leakage.
- Source: Synapse BraTS 2024 data folder `syn64952546` → BraTS-GLI (`syn59059776`).
  Not yet downloaded: `BraTS2024-BraTS-GLI-AdditionalTrainingData.zip` (`syn64314352`).

### Samples — `data/samples/`
3 labelled cases each for GLI and MEN-RT (`data/` + `labels/` + `paths.yaml`), and 3 BraTS-Path
histopathology PNGs with `labels.csv`. Useful for pipeline smoke tests.

## Environment

Python 3.12 in `.venv/`, managed by [uv](https://docs.astral.sh/uv/). GPU: RTX 3070 Ti, **8 GB**.

```powershell
.venv\Scripts\activate          # or call .venv\Scripts\python.exe directly
uv pip install -r requirements.lock.txt --torch-backend=cu128
```

Key packages: torch 2.11.0+cu128, monai 1.6.0, nibabel 5.4.2, numpy 2.5.3, jupyterlab.
Exact versions are pinned in `requirements.lock.txt`.

**8 GB VRAM is the binding constraint.** A full 182×218×182 volume × 4 modalities will not fit
through a 3D U-Net at a useful width, so train on random patches (e.g. 128³ or 96³) and run
sliding-window inference over the full volume. Use AMP (bfloat16 works on sm_86) and expect
batch size 1–2.

## Citations
If you publish anything, cite per `docs/citations/` (`brats2024_all.bib` covers the core set).
