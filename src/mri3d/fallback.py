"""Never return an empty mask when the task guarantees a target.

Every MEN-RT case is a radiotherapy plan, so every case has a gross tumour
volume. An empty prediction is therefore always wrong and always scores zero -
and six of the ten meningioma test cases that fail under both the 120- and the
300-epoch model fail exactly this way: small tumours (median 2.8 cc against
6.6 cc for the rest) where no voxel's foreground probability reached the argmax.

The rule: if the argmax mask has no foreground at all, take the voxel with the
highest foreground probability and keep the connected region around it where
foreground probability is at least half of that peak. "Half of the peak" is the
full-width-at-half-maximum convention, fixed in advance rather than swept, so it
adds no parameter for the test split to be fitted to. A case the model already
segments is never touched.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

HALF_MAX = 0.5


def fill_if_empty(pred: np.ndarray, probs: np.ndarray, frac: float = HALF_MAX) -> tuple[np.ndarray, bool]:
    """Return ``(pred, changed)``. ``probs`` is (classes, X, Y, Z) softmax, class 0 background."""
    if (pred > 0).any():
        return pred, False
    fg = 1.0 - probs[0]
    peak_idx = np.unravel_index(int(np.argmax(fg)), fg.shape)
    peak = float(fg[peak_idx])
    if peak <= 0:
        return pred, False
    lab, _ = ndimage.label(fg >= frac * peak)
    region = lab == lab[peak_idx]
    # The most likely foreground class at the peak names the region; for
    # MEN-RT there is only one.
    cls = int(np.argmax(probs[1:, peak_idx[0], peak_idx[1], peak_idx[2]])) + 1
    out = pred.copy()
    out[region] = cls
    return out, True
