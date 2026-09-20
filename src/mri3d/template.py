"""Average T1c brain, used as the anatomical backdrop for the heatmap viewer.

Every glioma case already lives in the same 182x218x182 SRI24 space, so a plain
voxel-wise mean over a sample of cases gives a usable template brain - no
registration needed. Each scan is normalised to its own 99th percentile first,
so one bright scan cannot dominate the average.

Run: ``python -m mri3d.template --n 200``
"""

from __future__ import annotations

import argparse

import nibabel as nib
import numpy as np

from . import paths
from .index import load_cases
from .scan import GLI_SHAPE, HEAT_SHAPE


def half_res(vol: np.ndarray) -> np.ndarray:
    a, b, c = HEAT_SHAPE
    return vol.reshape(a, 2, b, 2, c, 2).mean(axis=(1, 3, 5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="cases to average")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = [c for c in load_cases("gli_train")]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(cases), size=min(args.n, len(cases)), replace=False)

    acc = np.zeros(HEAT_SHAPE, dtype=np.float64)
    used = 0
    for k, i in enumerate(pick, 1):
        vol = np.asanyarray(nib.load(cases[i].images["t1c"].as_posix()).dataobj, dtype=np.float32)
        if vol.shape != GLI_SHAPE:
            continue
        hi = np.percentile(vol[vol > 0], 99) if (vol > 0).any() else 1.0
        acc += half_res(np.clip(vol / max(hi, 1e-6), 0, 1))
        used += 1
        if k % 50 == 0:
            print(f"  {k}/{len(pick)}", flush=True)

    template = (acc / max(used, 1)).astype(np.float32)
    paths.ensure_dirs()
    np.savez_compressed(paths.OUTPUTS / "template.npz", template=template)
    print(f"averaged {used} cases -> {paths.OUTPUTS / 'template.npz'} {template.shape}")


if __name__ == "__main__":
    main()
