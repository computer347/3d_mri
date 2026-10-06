"""Size filtering: the per-class mapping, and the edge cases around "no threshold"."""

from __future__ import annotations

import numpy as np

from mri3d.postprocess import filter_small


def _mask():
    m = np.zeros((30, 30, 30), dtype=np.uint8)
    m[2:12, 2:12, 2:12] = 2     # SNFH, 1000 voxels
    m[20:25, 20:25, 20:25] = 2  # SNFH, 125 voxels
    m[2:8, 20:26, 2:8] = 1      # NETC, 216 voxels
    return m


def test_zero_threshold_is_a_no_op():
    m = _mask()
    assert np.array_equal(filter_small(m, 0), m)


def test_empty_mapping_is_a_no_op():
    """An empty dict means no class has a threshold, not "compare against a dict".

    This used to raise TypeError: the mapping was tested for truthiness rather
    than for None, so `{}` fell through to the scalar branch.
    """
    m = _mask()
    assert np.array_equal(filter_small(m, {}), m)


def test_a_mapping_without_an_entry_leaves_that_class_alone():
    m = _mask()
    out = filter_small(m, {2: 500})     # nothing said about NETC
    assert (out == 1).sum() == 216, "NETC must be untouched"
    assert (out == 2).sum() == 1000, "only the 125-voxel SNFH blob goes"


def test_scalar_threshold_applies_to_every_class():
    m = _mask()
    out = filter_small(m, 500)
    assert (out == 1).sum() == 0
    assert (out == 2).sum() == 1000


def test_filtering_never_adds_voxels():
    m = _mask()
    for t in (0, {}, 50, 200, {1: 300, 2: 200}):
        out = filter_small(m, t)
        assert np.all((out != 0) <= (m != 0))
        assert np.all(out[out != 0] == m[out != 0]), "labels must not change, only vanish"
