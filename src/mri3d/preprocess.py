"""Resample MEN-RT to a common grid, once, to disk.

The meningioma scans arrive at whatever the scanner produced: matrices from
256x256x124 to 704x704x168, voxels from 0.37 to 1.5 mm. Training needs a common
spacing, and doing that inside the data loader means resampling a 142 MB volume
on every epoch, in every worker - which is what exhausted system memory on the
first attempt at this run.

Doing it once costs ~20 minutes and makes every later epoch both faster and
cheap in memory. This is what nnU-Net does, and for the same reason.

Output mirrors the input layout, so case IDs - and therefore the existing
splits - stay valid:

    data/brats2024_men_rt/train_1mm/BraTS-MEN-RT-0100-1/
        BraTS-MEN-RT-0100-1_t1c.nii.gz
        BraTS-MEN-RT-0100-1_gtv.nii.gz

``mri3d.index`` picks this directory up automatically once it exists.

    python -m mri3d.preprocess --workers 4
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Spacingd

from . import paths
from .index import load_cases

SPACING = (1.0, 1.0, 1.0)


def _process(args: tuple) -> tuple[str, str]:
    case_id, t1c_path, gtv_path, out_dir = args
    out = Path(out_dir) / case_id
    t1c_out = out / f"{case_id}_t1c.nii.gz"
    gtv_out = out / f"{case_id}_gtv.nii.gz"
    if t1c_out.exists() and gtv_out.exists():
        return case_id, "skipped (already done)"

    tf = Compose([
        LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=SPACING, mode=("bilinear", "nearest")),
    ])
    data = tf({"image": t1c_path, "label": gtv_path})

    out.mkdir(parents=True, exist_ok=True)
    img = data["image"][0].numpy()
    lab = data["label"][0].numpy().astype(np.uint8)
    affine = np.asarray(data["image"].affine, dtype=np.float64)

    # int16 holds MRI intensities without loss of anything meaningful and halves
    # the file size; the mask is a handful of values.
    nib.save(nib.Nifti1Image(np.rint(img).astype(np.int16), affine), t1c_out.as_posix())
    nib.save(nib.Nifti1Image(lab, affine), gtv_out.as_posix())
    return case_id, f"{img.shape} {lab.sum()} tumour voxels"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4,
                    help="keep this modest: each worker holds a full-resolution volume")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = paths.MEN_TRAIN_1MM
    out_dir.mkdir(parents=True, exist_ok=True)
    # Explicitly the raw directory: "men_train" resolves to the resampled copy
    # once it exists, and this function is what creates it.
    cases = [c for c in load_cases("men_train_raw") if c.has_label]
    if args.limit:
        cases = cases[: args.limit]

    jobs = [(c.case_id, c.images["t1c"].as_posix(), c.label.as_posix(), out_dir.as_posix())
            for c in cases]
    print(f"resampling {len(jobs)} cases to {SPACING} mm with {args.workers} workers")
    print(f"  -> {out_dir}")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case_id, note in pool.map(_process, jobs):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                rate = (time.time() - t0) / done
                print(f"  {done}/{len(jobs)}  {case_id}  {note}  "
                      f"[{rate:.1f}s/case, ~{rate*(len(jobs)-done)/60:.0f} min left]", flush=True)

    size = sum(f.stat().st_size for f in out_dir.rglob("*.nii.gz")) / 1e9
    print(f"\ndone in {(time.time()-t0)/60:.1f} min, {size:.1f} GB on disk")
    print("mri3d.index will now use this directory for men_train automatically")


if __name__ == "__main__":
    main()
