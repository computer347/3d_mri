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
"""

from __future__ import annotations

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


def lesion_wise_dice(pred: np.ndarray, ref: np.ndarray,
                     min_voxels: int = MIN_LESION_VOXELS) -> tuple[float, dict]:
    """Mean Dice over reference lesions, with unmatched predictions as zeros."""
    if pred.sum() == 0 and ref.sum() == 0:
        return 1.0, {"tp": 0, "fn": 0, "fp": 0}

    ref_lab, n_ref = ndimage.label(ref)
    pred_lab, n_pred = ndimage.label(pred)

    scores: list[float] = []
    matched_pred: set[int] = set()
    fn = 0
    for i in range(1, n_ref + 1):
        lesion = ref_lab == i
        if lesion.sum() < min_voxels:
            continue
        hits = np.unique(pred_lab[lesion])
        hits = hits[hits > 0]
        if hits.size == 0:
            scores.append(0.0)  # missed entirely
            fn += 1
            continue
        # Score the lesion against every predicted component touching it, so a
        # prediction that bridges two reference lesions is not double-counted.
        comp = np.isin(pred_lab, hits)
        scores.append(dice(comp, lesion))
        matched_pred.update(int(h) for h in hits)

    fp = 0
    for j in range(1, n_pred + 1):
        if j in matched_pred:
            continue
        if (pred_lab == j).sum() < min_voxels:
            continue
        scores.append(0.0)  # spurious lesion
        fp += 1

    if not scores:
        return 1.0, {"tp": 0, "fn": 0, "fp": 0}
    return float(np.mean(scores)), {"tp": len(scores) - fn - fp, "fn": fn, "fp": fp}


def score_case(pred: np.ndarray, ref: np.ndarray, class_values: dict[int, str]) -> dict:
    """Per-class volumetric and lesion-wise Dice for one case."""
    out: dict[str, float] = {}
    for value, name in class_values.items():
        if value == 0:
            continue
        p, r = pred == value, ref == value
        out[f"dice_{name}"] = dice(p, r)
        lw, counts = lesion_wise_dice(p, r)
        out[f"lesion_dice_{name}"] = lw
        out[f"lesions_missed_{name}"] = counts["fn"]
        out[f"lesions_false_{name}"] = counts["fp"]
        out[f"present_{name}"] = bool(r.any())
    return out
