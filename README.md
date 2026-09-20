# mri_3d — 3D brain MRI tumour segmentation / classification

Deep-learning project on the BraTS 2024 challenge data (multi-parametric 3D brain MRI, NIfTI `.nii.gz`).

## Layout

```
mri_3d/
├── data/
│   ├── raw/                        Original downloads, untouched (safe to delete once happy)
│   ├── brats2024_men_rt/train/     Meningioma radiotherapy — 500 cases, labelled
│   ├── brats2024_gli/
│   │   ├── train/                  Glioma training — 1350 cases, labelled
│   │   └── val/                    Glioma validation — 188 cases, NO labels
│   ├── samples/                    Tiny "fastlane" sets (3 cases each, labelled) for smoke tests
│   │   ├── gli/  men_rt/  path/
│   └── metadata/                   Clinical/demographic spreadsheets + Synapse manifest
├── docs/citations/                 BibTeX to cite for each BraTS sub-challenge
├── tools/brats2024_mlcubes/        Official MLCube prep/eval configs (Docker-based eval)
├── src/                            Python package code (datasets, models, training)
├── notebooks/                      Exploration / visualisation
├── configs/                        Experiment configs
└── outputs/                        Checkpoints, logs, predictions (gitignore this)
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
