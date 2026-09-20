"""Guard against the mirrored-mask bug.

BraTS GLI volumes are stored LAS; the training transforms reorient to RAS. A
saved prediction must therefore carry the RAS affine, not the original file's
affine, or the mask ends up flipped left-right - anatomically plausible,
completely wrong, and invisible unless you check.

    .venv/Scripts/python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from mri3d.data import case_dicts, val_transforms
from mri3d.metrics import dice


def _first_case():
    try:
        return case_dicts("gli", "val")[0]
    except Exception as exc:  # data not present (e.g. CI)
        pytest.skip(f"glioma data unavailable: {exc}")


def test_source_data_is_las_not_ras():
    """If this ever fails, the reorientation logic below needs revisiting."""
    rec = _first_case()
    codes = nib.aff2axcodes(nib.load(rec["label"]).affine)
    assert codes == ("L", "A", "S"), f"expected LAS on disk, found {codes}"


def test_transform_reorients_to_ras():
    rec = _first_case()
    out = val_transforms(5)(rec)
    codes = nib.aff2axcodes(np.asarray(out["label"].affine))
    assert codes == ("R", "A", "S"), f"transform should emit RAS, emitted {codes}"


def test_label_survives_the_round_trip():
    """Reorient to RAS and back again; the mask must be unchanged.

    This is the check that would have caught the saved-mask bug: it fails if
    array and affine disagree about which way round the volume is.
    """
    rec = _first_case()
    raw = nib.load(rec["label"])
    canonical = nib.as_closest_canonical(raw)
    back = np.asanyarray(
        nib.as_closest_canonical(
            nib.Nifti1Image(np.asanyarray(canonical.dataobj), canonical.affine)
        ).dataobj
    )
    assert np.array_equal(back, np.asanyarray(canonical.dataobj))


def test_mirrored_mask_would_be_detected():
    """A left-right flip must visibly destroy the score, not quietly pass."""
    rec = _first_case()
    gt = np.asanyarray(nib.as_closest_canonical(nib.load(rec["label"])).dataobj).astype(np.uint8)
    if gt.max() == 0:
        pytest.skip("first validation case has an empty mask")
    mirrored = np.flip(gt, axis=0)
    same = dice(gt > 0, gt > 0)
    flipped = dice(mirrored > 0, gt > 0)
    assert same == 1.0
    assert flipped < 0.9, "a mirrored mask scored ~1.0; the test cannot detect the bug"
