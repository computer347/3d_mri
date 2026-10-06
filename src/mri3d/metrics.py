"""Scoring, in the two forms that matter here.

**Volumetric Dice** is the usual overlap ratio, with BraTS's empty-class
convention: if a case genuinely has no ET and the model predicts no ET, that
scores 1.0, not 0.0. Getting this wrong quietly deflates every average, and
~58% of cases have no NETC at all.

**Lesion-wise Dice** is what the challenge actually ranks on, and it is the
detection-flavoured metric. Each connected component of the reference is scored
separately, so a missed satellite lesion costs a full zero instead of being
hidden by one large, well-segmented mass; predicted components that match no
reference lesion are counted as zeros too, which is what stops a model from
buying overlap with scattered false positives.

The matching that underpins lesion-wise Dice is exposed separately as
``match_components``. Anything that wants to reason about individual predicted
components - the lesion selector in ``mri3d.select``, the per-component feature
extraction in ``mri3d.predict`` - must use the *same* notion of matched,
missed and spurious as the metric it is trying to move, or it will optimise a
target the leaderboard does not score. ``lesion_wise_dice`` is now a thin
reduction over ``match_components``; ``tests/test_metrics.py`` pins that the
returned numbers did not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# Lesions below this many voxels are ignored, matching the BraTS convention of
# not scoring specks that annotators would not have drawn consistently.
MIN_LESION_VOXELS = 50


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = pred.sum(), ref.sum()
    if p == 0 and r == 0:
        return 1.0  # correctly predicted absence
    if p == 0 or r == 0:
        return 0.0
    return float(2 * np.logical_and(pred, ref).sum() / (p + r))


@dataclass
class PredComponent:
    """One connected component of the *prediction*."""

    comp_id: int
    n_voxels: int
    #: Overlaps ``>= 1`` voxel of a scored reference lesion. Note that a
    #: component is matched regardless of its own size - size only decides
    #: whether an *un*matched component is charged as a false positive.
    matched: bool
    #: Which scored reference lesions it touches (label ids in ``ref_lab``).
    matched_ref_ids: list[int] = field(default_factory=list)
    #: Dice of this component against the union of the scored reference
    #: lesions it touches. 0.0 when it touches none.
    dice_with_matched: float = 0.0
    #: Charged as a false positive by ``lesion_wise_dice``: unmatched *and*
    #: at least ``min_voxels`` large.
    false_positive: bool = False


@dataclass
class RefLesion:
    """One connected component of the *reference*."""

    ref_id: int
    n_voxels: int
    #: Below ``min_voxels``: not scored at all, in either direction.
    scored: bool
    #: Any predicted component touches it (size of that component irrelevant).
    detected: bool
    #: The lesion-wise Dice entry this lesion contributes; 0.0 when missed.
    dice: float
    #: Which predicted components touch it.
    matched_comp_ids: list[int] = field(default_factory=list)


@dataclass
class Matching:
    pred_lab: np.ndarray
    ref_lab: np.ndarray
    comps: list[PredComponent]
    refs: list[RefLesion]

    @property
    def scored_refs(self) -> list[RefLesion]:
        return [r for r in self.refs if r.scored]


def match_components(pred: np.ndarray, ref: np.ndarray,
                     min_voxels: int = MIN_LESION_VOXELS) -> Matching:
    """Connected-component matching, exactly as ``lesion_wise_dice`` scores it.

    Three conventions, all of them load-bearing:

    1. Reference lesions below ``min_voxels`` are dropped entirely. They are
       neither scored nor able to rescue a predicted component that touches
       only them.
    2. A predicted component matches if it overlaps **at least one voxel** of a
       scored reference lesion. There is no overlap-fraction threshold; BraTS
       lesion-wise scoring is a detection test, not an IoU test.
    3. A reference lesion is scored against the **union** of every predicted
       component touching it, so a single prediction bridging two reference
       lesions is not counted twice and is not penalised twice.

    Implementation note: the obvious loop (``for i in range(n_ref): ref_lab == i``)
    is O(n_lesions x volume) and was measurably slow once this ran over 406
    cases x 4 classes. Everything below is derived from one pass over the
    voxels where both labellings are non-zero, giving the full pred x ref
    overlap table; sizes come from ``bincount``. The results are exact, not
    approximate - only the arithmetic order changed.
    """
    ref_lab, n_ref = ndimage.label(ref)
    pred_lab, n_pred = ndimage.label(pred)

    pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)
    ref_sizes = np.bincount(ref_lab.ravel(), minlength=n_ref + 1)

    # Sparse overlap table: overlap[(j, i)] = |pred component j n ref lesion i|.
    overlap: dict[tuple[int, int], int] = {}
    if n_pred and n_ref:
        both = (pred_lab > 0) & (ref_lab > 0)
        if both.any():
            key = pred_lab[both].astype(np.int64) * (n_ref + 1) + ref_lab[both].astype(np.int64)
            uniq, counts = np.unique(key, return_counts=True)
            for k, c in zip(uniq.tolist(), counts.tolist()):
                overlap[(k // (n_ref + 1), k % (n_ref + 1))] = c

    by_ref: dict[int, list[int]] = {}
    by_comp: dict[int, list[int]] = {}
    for (j, i) in overlap:
        by_ref.setdefault(i, []).append(j)
        by_comp.setdefault(j, []).append(i)

    scored_ref = {i for i in range(1, n_ref + 1) if ref_sizes[i] >= min_voxels}

    refs: list[RefLesion] = []
    matched_pred: set[int] = set()
    for i in range(1, n_ref + 1):
        hits = sorted(by_ref.get(i, []))
        if i not in scored_ref:
            refs.append(RefLesion(i, int(ref_sizes[i]), scored=False,
                                  detected=bool(hits), dice=0.0, matched_comp_ids=hits))
            continue
        if not hits:
            refs.append(RefLesion(i, int(ref_sizes[i]), scored=True, detected=False, dice=0.0))
            continue
        inter = sum(overlap[(j, i)] for j in hits)
        union_pred = sum(int(pred_sizes[j]) for j in hits)
        d = float(2 * inter / (union_pred + int(ref_sizes[i])))
        refs.append(RefLesion(i, int(ref_sizes[i]), scored=True, detected=True,
                              dice=d, matched_comp_ids=hits))
        matched_pred.update(hits)

    comps: list[PredComponent] = []
    for j in range(1, n_pred + 1):
        touched = sorted(i for i in by_comp.get(j, []) if i in scored_ref)
        matched = bool(touched)
        d = 0.0
        if matched:
            inter = sum(overlap[(j, i)] for i in touched)
            d = float(2 * inter / (int(pred_sizes[j]) + sum(int(ref_sizes[i]) for i in touched)))
        comps.append(PredComponent(
            comp_id=j, n_voxels=int(pred_sizes[j]), matched=matched,
            matched_ref_ids=touched, dice_with_matched=d,
            false_positive=(not matched) and int(pred_sizes[j]) >= min_voxels))

    return Matching(pred_lab=pred_lab, ref_lab=ref_lab, comps=comps, refs=refs)


def lesion_wise_dice(pred: np.ndarray, ref: np.ndarray,
                     min_voxels: int = MIN_LESION_VOXELS) -> tuple[float, dict]:
    """Mean Dice over reference lesions, with unmatched predictions as zeros."""
    if pred.sum() == 0 and ref.sum() == 0:
        return 1.0, {"tp": 0, "fn": 0, "fp": 0}

    m = match_components(pred, ref, min_voxels)
    scores = [r.dice for r in m.scored_refs]
    fn = sum(1 for r in m.scored_refs if not r.detected)
    fp = sum(1 for c in m.comps if c.false_positive)
    scores += [0.0] * fp  # spurious lesions each contribute a full zero

    if not scores:
        return 1.0, {"tp": 0, "fn": 0, "fp": 0}
    return float(np.mean(scores)), {"tp": len(scores) - fn - fp, "fn": fn, "fp": fp}


# Composite regions, scored by the challenge alongside the four sub-regions.
# Definitions quoted from the BraTS 2024 post-treatment glioma paper
# (arXiv:2405.18368):
#   "Tumor core (ET plus NETC) describes what is typically resected during a
#    surgical procedure."
#   "Whole tumor (ET plus SNFH plus NETC) defines the whole extent of the
#    tumor, including the tumor core, infiltrating tumor, peritumoral edema and
#    treatment-related changes."
# Note that NEITHER includes the resection cavity - RC is scored only on its
# own. Assuming otherwise would silently inflate both composites on this
# cohort, where 85% of cases contain a cavity.
GLI_REGIONS = {
    "TC": (1, 3),        # NETC + ET
    "WT": (1, 2, 3),     # NETC + SNFH + ET
}


def score_case(pred: np.ndarray, ref: np.ndarray, class_values: dict[int, str],
               regions: dict[str, tuple] | None = None) -> dict:
    """Volumetric and lesion-wise Dice per class, plus the composite regions.

    Reported the same way the challenge leaderboard reports it, so numbers here
    are directly comparable to it.
    """
    out: dict[str, float] = {}
    targets: list[tuple[str, tuple]] = [
        (name, (value,)) for value, name in class_values.items() if value != 0
    ]
    if regions is None and set(class_values) >= {1, 2, 3}:
        regions = GLI_REGIONS
    targets += list((regions or {}).items())

    for name, values in targets:
        p = np.isin(pred, values)
        r = np.isin(ref, values)
        out[f"dice_{name}"] = dice(p, r)
        lw, counts = lesion_wise_dice(p, r)
        out[f"lesion_dice_{name}"] = lw
        out[f"lesions_missed_{name}"] = counts["fn"]
        out[f"lesions_false_{name}"] = counts["fp"]
        out[f"present_{name}"] = bool(r.any())
    return out
