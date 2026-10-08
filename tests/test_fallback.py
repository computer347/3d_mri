"""The empty-mask fallback: only ever fires on an empty mask, and stays local."""

from __future__ import annotations

import numpy as np

from mri3d.fallback import fill_if_empty


def _probs(fg: np.ndarray) -> np.ndarray:
    return np.stack([1.0 - fg, fg]).astype(np.float32)


def test_a_non_empty_prediction_is_never_touched():
    fg = np.zeros((10, 10, 10), np.float32)
    fg[2:4, 2:4, 2:4] = 0.9
    pred = (fg > 0.5).astype(np.uint8)
    out, changed = fill_if_empty(pred, _probs(fg))
    assert not changed and out is pred


def test_empty_prediction_gets_the_half_max_region_around_the_peak_only():
    fg = np.zeros((20, 20, 20), np.float32)
    fg[4:7, 4:7, 4:7] = 0.30          # the faint tumour; peak 0.40 at its centre
    fg[5, 5, 5] = 0.40
    fg[14:16, 14:16, 14:16] = 0.10    # a separate, weaker blob below half max
    pred = np.zeros(fg.shape, np.uint8)
    out, changed = fill_if_empty(pred, _probs(fg))
    assert changed
    assert out[4:7, 4:7, 4:7].all()
    assert out.sum() == 27
    assert not out[14:16, 14:16, 14:16].any()


def test_disconnected_region_above_half_max_is_not_added():
    fg = np.zeros((20, 20, 20), np.float32)
    fg[2:4, 2:4, 2:4] = 0.40
    fg[15:17, 15:17, 15:17] = 0.35   # above half max, but not connected to the peak
    out, changed = fill_if_empty(np.zeros(fg.shape, np.uint8), _probs(fg))
    assert changed
    assert out[2:4, 2:4, 2:4].all()
    assert not out[15:17, 15:17, 15:17].any()
