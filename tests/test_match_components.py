"""``match_components`` must agree with ``lesion_wise_dice``, and with the CSV.

Two levels of check. The synthetic tests pin that the refactored matching is
the *same* matching the metric performs - if those drift, the selector would be
optimising a target the leaderboard does not score. The reconciliation test
pins that the per-component CSV written during inference reproduces the
false-positive and missed-lesion totals already reported in
``reports/scores_gli_main_test.csv``, which is the cheapest way to catch an
off-by-one in the extraction.

    .venv/Scripts/python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from mri3d.metrics import (MIN_LESION_VOXELS, dice, lesion_wise_dice,
                           match_components)

REPORTS = Path(__file__).resolve().parents[1] / "reports"
RUN, SPLIT = "gli_main", "test"

# Pinned from reports/scores_gli_main_test.csv. They are also recomputed from
# that CSV below, so a change to the predictions fails loudly here rather than
# silently shifting the baseline the post-processing sections compare against.
#
# An intermediate version of mri3d.seeding set cudnn.benchmark=True and these
# came out as SNFH 191, RC 89 instead - autotuning had selected different
# convolution algorithms, moving enough boundary voxels for two components to
# cross the 50-voxel floor. Seeding with benchmark off reproduces the original
# run exactly. That near-miss is the reason this test pins absolute numbers and
# not just internal agreement.
EXPECTED_FALSE = {"NETC": 89, "SNFH": 192, "ET": 94, "RC": 87}
EXPECTED_MISSED = {"NETC": 25, "SNFH": 32, "ET": 29, "RC": 10}


def _blobs(seed: int, quantile: float, shape=(28, 28, 28)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = ndimage.gaussian_filter(rng.normal(size=shape), sigma=1.6)
    return x > np.quantile(x, quantile)


@pytest.mark.parametrize("seed", range(12))
def test_matching_reproduces_lesion_wise_dice(seed):
    """Rebuild the metric from the match table; it must come out identical."""
    pred = _blobs(seed, 0.90)
    ref = _blobs(seed + 100, 0.88)
    for min_voxels in (1, 10, MIN_LESION_VOXELS):
        m = match_components(pred, ref, min_voxels)
        scores = [r.dice for r in m.scored_refs]
        scores += [0.0] * sum(1 for c in m.comps if c.false_positive)
        expected, counts = lesion_wise_dice(pred, ref, min_voxels)
        got = float(np.mean(scores)) if scores else 1.0
        assert got == pytest.approx(expected, abs=1e-12)
        assert counts["fp"] == sum(1 for c in m.comps if c.false_positive)
        assert counts["fn"] == sum(1 for r in m.scored_refs if not r.detected)


def test_a_reference_lesion_is_scored_against_the_union_of_its_matches():
    """Two predicted blobs touching one reference lesion count once, together."""
    ref = np.zeros((30, 30, 30), dtype=bool)
    ref[5:15, 5:15, 5:15] = True
    pred = np.zeros_like(ref)
    pred[5:10, 5:15, 5:15] = True     # one half
    pred[12:15, 5:15, 5:15] = True    # the other, separated by a gap
    m = match_components(pred, ref)
    assert len(m.comps) == 2
    assert all(c.matched for c in m.comps)
    assert len(m.scored_refs) == 1
    assert m.scored_refs[0].dice == pytest.approx(dice(pred, ref))
    # Neither half is charged as a false positive, and the lesion is one entry.
    assert lesion_wise_dice(pred, ref)[1] == {"tp": 1, "fn": 0, "fp": 0}


def test_small_reference_lesions_cannot_rescue_a_prediction():
    """A component touching only an unscored speck is still a false positive."""
    ref = np.zeros((30, 30, 30), dtype=bool)
    ref[2:12, 2:12, 2:12] = True           # scored
    ref[25:27, 25:27, 25:27] = True        # 8 voxels: below MIN_LESION_VOXELS
    pred = np.zeros_like(ref)
    pred[2:12, 2:12, 2:12] = True
    pred[24:28, 24:28, 24:28] = True       # 64 voxels, overlaps only the speck
    m = match_components(pred, ref)
    speck_hit = [c for c in m.comps if c.n_voxels == 64][0]
    assert not speck_hit.matched
    assert speck_hit.false_positive
    assert lesion_wise_dice(pred, ref)[1]["fp"] == 1


def test_an_unmatched_component_below_the_floor_is_not_a_false_positive():
    ref = np.zeros((30, 30, 30), dtype=bool)
    ref[2:12, 2:12, 2:12] = True
    pred = ref.copy()
    pred[25:28, 25:28, 25:28] = True       # 27 voxels, invented but too small to count
    m = match_components(pred, ref)
    tiny = [c for c in m.comps if c.n_voxels == 27][0]
    assert not tiny.matched and not tiny.false_positive
    assert lesion_wise_dice(pred, ref)[1]["fp"] == 0


def _load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.skipif(not (REPORTS / f"components_{RUN}_{SPLIT}.csv").exists(),
                    reason="run: python -m mri3d.predict --run gli_main --split test "
                           "--save-components")
def test_components_csv_reconciles_with_the_scores_csv():
    """Per-class false positives and missed lesions must match, exactly.

    ``score_case`` and the component extraction reach these numbers by
    completely different routes - one sums per-case metric counters, the other
    tabulates individual components - so agreement to the unit is a real check
    that the extraction inherited the metric's conventions rather than
    approximating them.
    """
    comps = _load(REPORTS / f"components_{RUN}_{SPLIT}.csv")
    refs = _load(REPORTS / f"ref_lesions_{RUN}_{SPLIT}.csv")
    scores = _load(REPORTS / f"scores_{RUN}_{SPLIT}.csv")

    false_from_comps = Counter(
        c["class_name"] for c in comps
        if c["matched"] == "False" and int(c["n_voxels"]) >= MIN_LESION_VOXELS)
    missed_from_refs = Counter(r["class_name"] for r in refs if r["detected"] == "False")

    for name in EXPECTED_FALSE:
        from_scores = sum(int(s[f"lesions_false_{name}"]) for s in scores)
        assert false_from_comps[name] == from_scores, f"{name}: components CSV disagrees"
        assert from_scores == EXPECTED_FALSE[name], f"{name}: predictions changed"

        from_scores = sum(int(s[f"lesions_missed_{name}"]) for s in scores)
        assert missed_from_refs[name] == from_scores, f"{name}: ref-lesions CSV disagrees"
        assert from_scores == EXPECTED_MISSED[name], f"{name}: predictions changed"
