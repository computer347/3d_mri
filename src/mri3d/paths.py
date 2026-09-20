"""Canonical locations for data and derived artefacts.

Everything downstream imports paths from here, so moving the data means editing
one file (or setting MRI3D_DATA).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MRI3D_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("MRI3D_DATA", ROOT / "data"))

GLI_TRAIN = DATA / "brats2024_gli" / "train"
GLI_VAL = DATA / "brats2024_gli" / "val"
MEN_TRAIN = DATA / "brats2024_men_rt" / "train"
# Written by mri3d.preprocess: the same cases resampled to 1 mm isotropic.
# Preferred automatically when present (see mri3d.index).
MEN_TRAIN_1MM = DATA / "brats2024_men_rt" / "train_1mm"

OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

# Derived artefacts small enough to commit (they contain no patient imagery).
CASE_INDEX = REPORTS / "case_index.csv"
SPLITS = CONFIGS / "splits.json"
EMBEDDINGS = OUTPUTS / "embeddings.npz"
HEATMAPS = OUTPUTS / "heatmaps.npz"
DUPLICATES = REPORTS / "duplicates.csv"

# BraTS 2024 post-treatment glioma label map.
GLI_LABELS = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
GLI_CLASSES = ["NETC", "SNFH", "ET", "RC"]
# Meningioma radiotherapy: a single binary gross tumour volume.
MEN_LABELS = {0: "background", 1: "GTV"}


def ensure_dirs() -> None:
    for d in (OUTPUTS, REPORTS, CONFIGS):
        d.mkdir(parents=True, exist_ok=True)
