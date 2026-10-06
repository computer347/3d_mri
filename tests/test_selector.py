"""The lesion selector's edges: the no-op, the total wipe, and the safety check.

The threshold in §2.1 is derived rather than tuned, so the two ends of its
range are the cheapest honest test of the plumbing: at p* = 0 nothing may be
deleted, and at p* > 1 everything the metric can see must be. Anything between
those is a measurement, not an assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from mri3d.metrics import MIN_LESION_VOXELS
from mri3d.select import (DECIDABLE_MIN_VOXELS, break_even_threshold,
                          filter_by_selector, keep_table)

LABELS = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}


def _mask_and_rows():
    """One SNFH mask with three components: large, medium, and below the floor."""
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[2:12, 2:12, 2:12] = 2       # 1000 voxels
    mask[20:26, 20:26, 20:26] = 2    # 216 voxels
    mask[35:38, 35:38, 35:38] = 2    # 27 voxels: under MIN_LESION_VOXELS
    rows = [
        {"case_id": "C", "class_name": "SNFH", "comp_id": 1, "n_voxels": 1000,
         "mean_prob": 0.95, "p90_prob": 0.99, "max_prob": 1.0, "mean_border_prob": 0.2,
         "dist_to_dominant_mm": 0.0, "mc_mean_var": 1e-4, "rank_by_prob": 1,
         "n_comps_in_class": 3},
        {"case_id": "C", "class_name": "SNFH", "comp_id": 2, "n_voxels": 216,
         "mean_prob": 0.60, "p90_prob": 0.80, "max_prob": 0.9, "mean_border_prob": 0.4,
         "dist_to_dominant_mm": 30.0, "mc_mean_var": 1e-2, "rank_by_prob": 2,
         "n_comps_in_class": 3},
        {"case_id": "C", "class_name": "SNFH", "comp_id": 3, "n_voxels": 27,
         "mean_prob": 0.55, "p90_prob": 0.70, "max_prob": 0.8, "mean_border_prob": 0.45,
         "dist_to_dominant_mm": 44.0, "mc_mean_var": 2e-2, "rank_by_prob": 3,
         "n_comps_in_class": 3},
    ]
    return mask, rows


def _model(features, classes=("SNFH",)):
    """A selector with arbitrary but non-degenerate coefficients."""
    n = len(features) + max(0, len(classes) - 1)
    return {"features": list(features), "classes": list(classes),
            "impute": {f: 0.0 for f in features},
            "mean": [0.0] * n, "std": [1.0] * n,
            "coef": [0.3] * n, "intercept": -0.5}


FEATURES = ["log_n_voxels", "mean_prob", "mean_border_prob", "log1p_dist", "rank_by_prob"]


def test_selector_is_a_no_op_at_threshold_zero():
    mask, rows = _mask_and_rows()
    keep = keep_table(_model(FEATURES), rows, 0.0)
    assert all(keep.values())
    out = filter_by_selector(mask, "C", LABELS, keep)
    assert np.array_equal(out, mask)


def test_above_one_deletes_everything_the_metric_can_see():
    """A posterior is never >= 1.0001, so only the sub-floor component survives."""
    mask, rows = _mask_and_rows()
    keep = keep_table(_model(FEATURES), rows, 1.0001)
    assert keep[("C", "SNFH", 1)] is False
    assert keep[("C", "SNFH", 2)] is False
    # Under MIN_LESION_VOXELS the metric does not charge for it, so the selector
    # is not allowed to gamble volumetric Dice on removing it.
    assert keep[("C", "SNFH", 3)] is True
    out = filter_by_selector(mask, "C", LABELS, keep)
    assert (out == 2).sum() == 27


def test_components_absent_from_the_table_are_kept():
    """Missing is not a vote to delete: an unknown component stays."""
    mask, _ = _mask_and_rows()
    assert np.array_equal(filter_by_selector(mask, "C", LABELS, {}), mask)


def test_a_size_mismatch_raises_rather_than_deleting_the_wrong_blob():
    """The CSV is written at inference, the mask is read back later from disk.

    They are tied together only by ndimage.label being deterministic. If the two
    ever come from different inference runs the component ids would still line
    up numerically while pointing at different tissue, which is exactly the kind
    of silent error this project is supposed to catch.
    """
    mask, rows = _mask_and_rows()
    keep = keep_table(_model(FEATURES), rows, 0.5)
    sizes = {("C", "SNFH", 1): 999}  # deliberately wrong
    with pytest.raises(RuntimeError, match="component mismatch"):
        filter_by_selector(mask, "C", LABELS, keep, sizes)


def test_decidable_floor_matches_the_metric():
    assert DECIDABLE_MIN_VOXELS == MIN_LESION_VOXELS


def test_break_even_threshold_matches_the_derivation():
    # p* = L / (L + d): equal stakes means a coin flip, and a case already
    # scoring badly (low L) should be more willing to gamble on a lesion.
    assert break_even_threshold(0.7, 0.7) == pytest.approx(0.5)
    assert break_even_threshold(0.72, 0.70) == pytest.approx(0.5070, abs=1e-4)
    assert break_even_threshold(0.3, 0.7) < break_even_threshold(0.9, 0.7)
