"""Metric behaviour, including the conventions that are easy to get wrong.

    .venv/Scripts/python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np

from mri3d.metrics import GLI_REGIONS, dice, lesion_wise_dice, score_case

LABELS = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}


def test_empty_reference_and_empty_prediction_scores_one():
    """BraTS convention: correctly predicting absence is a perfect score."""
    z = np.zeros((8, 8, 8), dtype=np.uint8)
    assert dice(z, z) == 1.0


def test_empty_reference_with_prediction_scores_zero():
    z = np.zeros((8, 8, 8), dtype=np.uint8)
    p = z.copy()
    p[2:4, 2:4, 2:4] = 1
    assert dice(p, z) == 0.0


def test_composite_regions_exclude_resection_cavity():
    """TC = ET+NETC, WT = ET+NETC+SNFH. Neither includes RC (label 4).

    Pinned against the challenge paper's own wording; folding RC into either
    composite would inflate scores on a cohort where 85% of cases have one.
    """
    assert GLI_REGIONS["TC"] == (1, 3)
    assert GLI_REGIONS["WT"] == (1, 2, 3)
    assert 4 not in GLI_REGIONS["TC"] and 4 not in GLI_REGIONS["WT"]


def test_composites_are_built_from_the_right_labels():
    ref = np.zeros((10, 10, 10), dtype=np.uint8)
    ref[0:2] = 1   # NETC
    ref[2:4] = 2   # SNFH
    ref[4:6] = 3   # ET
    ref[6:8] = 4   # RC
    scores = score_case(ref.copy(), ref, LABELS)
    # A perfect prediction scores 1.0 everywhere, composites included.
    for key in ("dice_TC", "dice_WT", "dice_NETC", "dice_RC"):
        assert scores[key] == 1.0

    # Predicting the cavity as tumour must not be rewarded by the composites.
    pred = ref.copy()
    pred[6:8] = 3  # call the RC "enhancing tumour"
    s = score_case(pred, ref, LABELS)
    assert s["dice_RC"] == 0.0
    assert s["dice_WT"] < 1.0, "WT must be penalised for swallowing the cavity"


def test_lesion_wise_penalises_a_missed_second_lesion():
    """The point of lesion-wise scoring: one big hit cannot hide one total miss."""
    ref = np.zeros((40, 40, 40), dtype=np.uint8)
    ref[2:12, 2:12, 2:12] = 1        # large lesion
    ref[30:36, 30:36, 30:36] = 1     # smaller, separate lesion
    pred = np.zeros_like(ref)
    pred[2:12, 2:12, 2:12] = 1       # only the large one found

    volumetric = dice(pred > 0, ref > 0)
    lesionwise, counts = lesion_wise_dice(pred > 0, ref > 0)
    assert volumetric > 0.8, "volumetric Dice barely notices the miss"
    assert lesionwise <= 0.55, "lesion-wise must charge a zero for the missed lesion"
    assert counts["fn"] == 1


def test_lesion_wise_penalises_invented_lesions():
    ref = np.zeros((40, 40, 40), dtype=np.uint8)
    ref[2:12, 2:12, 2:12] = 1
    pred = ref.copy()
    pred[30:36, 30:36, 30:36] = 1    # a lesion that is not there
    lesionwise, counts = lesion_wise_dice(pred > 0, ref > 0)
    assert counts["fp"] == 1
    assert lesionwise < 0.6
