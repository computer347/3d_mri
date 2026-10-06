"""Let the lesion selector look at the MRI.

``mri3d.select`` currently decides from the prediction alone - how big a
component is, how confident the softmax was, how far it sits from the dominant
mass. It never sees the scan. That is the measured ceiling on it: the oracle
selector reaches 0.697 mean lesion-wise Dice on the glioma test split against
the fitted selector's 0.620, and the fitted model's component-level AUC is
0.869. A confident blob in normal-appearing white matter and a genuine FLAIR
hyperintensity look identical to it.

They do not look identical on the images, and the differences are the ones
radiologists name:

* **Contrast against the neighbourhood.** Oedema is bright on T2-FLAIR relative
  to the tissue immediately around it. A false positive usually is not - the
  model has drawn a boundary where the intensities do not change.
* **Enhancement.** Enhancing tumour is, by definition, brighter on T1c than on
  T1n. ``z_t1c_mean - z_t1n_mean`` is that definition, written down.
* **Contralateral asymmetry.** The brain is roughly symmetric and the SRI24
  space these volumes live in makes that usable: compare each component with
  the mirrored location in the other hemisphere. A real lesion differs from its
  mirror; a normal structure the model mistook for tumour does not.
* **Where it is.** Distance to the brain edge separates genuine peripheral
  lesions from the partial-volume rind the model tends to invent.

Nothing here needs inference or retraining - the masks are already on disk, so
this is one pass over the images per split.

    python -m mri3d.image_features --run gli_main --split val
    python -m mri3d.image_features --run gli_main --split test

Writes ``reports/image_features_{run}_{split}.csv``, keyed by
``(case_id, class_name, comp_id)`` so it joins onto the components CSV that
``predict.py --save-components`` wrote. ``mri3d.select`` picks it up
automatically when it is present.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from . import paths
from .data import case_dicts

# Modality order is the sorted filename order that data.case_dicts produces.
GLI_MODALITIES = ["t1c", "t1n", "t2f", "t2w"]
MEN_MODALITIES = ["t1c"]

# Thickness of the shell used for the local-contrast features. Three voxels is
# a compromise measured against the alternatives: one voxel sits inside the
# segmentation's own uncertain boundary and mostly re-measures the lesion, and
# much more than five reaches past the neighbouring tissue into whatever else
# is nearby.
SHELL_ITERS = 3

KEY = ["case_id", "class_name", "comp_id"]


def feature_names(modalities: list[str]) -> list[str]:
    out: list[str] = []
    for m in modalities:
        out += [f"z_{m}_mean", f"z_{m}_contrast", f"z_{m}_mirror"]
    if "t1c" in modalities and "t1n" in modalities:
        out.append("enhancement")       # z_t1c_mean - z_t1n_mean
    if "t2f" in modalities and "t2w" in modalities:
        out.append("flair_minus_t2")    # separates cavity/CSF from oedema
    out += ["dist_to_brain_edge_mm", "frac_outside_brain", "n_voxels_check"]
    return out


def _zscore(vol: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Per-case, per-modality standardisation over brain voxels only.

    BraTS intensities have no absolute meaning and differ by scanner, so a raw
    value is not comparable between cases. Standardising over the brain rather
    than the whole array keeps the air - which is most of the volume and is
    exactly zero here - from setting the scale.
    """
    v = vol[brain]
    mu, sd = float(v.mean()), float(v.std())
    return (vol - mu) / (sd if sd > 1e-6 else 1.0)


def _mirror(vol: np.ndarray, midline: int) -> np.ndarray:
    """The volume reflected across the brain's own midline, along the L-R axis.

    Everything downstream is in RAS, so axis 0 runs right-to-left. Reflecting
    about the array centre would be wrong whenever the head is off-centre, so
    the midline comes from the brain mask's centroid. Written as a flip plus an
    integer roll, which is exact and costs one copy rather than an
    interpolation over 7 million voxels.
    """
    n = vol.shape[0]
    return np.roll(np.flip(vol, axis=0), -(n - 1 - 2 * midline), axis=0)


def case_rows(rec: dict, pred_dir: Path, class_values: dict[int, str],
              modalities: list[str]) -> list[dict]:
    pred = np.asanyarray(nib.as_closest_canonical(
        nib.load((pred_dir / f"{rec['case_id']}_pred.nii.gz").as_posix())).dataobj).astype(np.uint8)
    vols = [np.asanyarray(nib.as_closest_canonical(nib.load(p)).dataobj).astype(np.float32)
            for p in rec["image"]]

    # GLI is skull-stripped so the background is exactly zero. MEN-RT is not, so
    # a zero test would call the whole head "brain"; a low quantile of the
    # non-zero intensities is a crude but stable stand-in there.
    t1c = vols[0]
    brain = t1c > 0
    if brain.mean() > 0.55:  # not skull-stripped
        brain = t1c > np.quantile(t1c[t1c > 0], 0.25)
    if not brain.any():
        brain = t1c > 0

    z = [_zscore(v, brain) for v in vols]
    midline = int(round(ndimage.center_of_mass(brain)[0]))
    zm = [_mirror(v, midline) for v in z]
    # Distance from every voxel to the outside of the brain, in millimetres.
    affine = np.asarray(nib.as_closest_canonical(nib.load(rec["image"][0])).affine)
    spacing = tuple(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))
    edge = ndimage.distance_transform_edt(brain, sampling=spacing)

    rows: list[dict] = []
    for value, name in class_values.items():
        if value == 0:
            continue
        binary = pred == value
        if not binary.any():
            continue
        lab, n = ndimage.label(binary)
        objs = ndimage.find_objects(lab)
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        for j in range(1, n + 1):
            sl = objs[j - 1]
            if sl is None:
                continue
            # Work inside the bounding box padded by the shell width, so the
            # dilation has room and the arithmetic stays local.
            pad = tuple(slice(max(0, s.start - SHELL_ITERS - 1),
                              min(d, s.stop + SHELL_ITERS + 1))
                        for s, d in zip(sl, lab.shape))
            comp = lab[pad] == j
            shell = ndimage.binary_dilation(comp, iterations=SHELL_ITERS) & ~comp & brain[pad]
            row = {"case_id": rec["case_id"], "class_name": name, "comp_id": j,
                   "n_voxels_check": int(counts[j])}
            for m, zz, mm in zip(modalities, z, zm):
                inside = float(zz[pad][comp].mean())
                row[f"z_{m}_mean"] = round(inside, 5)
                row[f"z_{m}_contrast"] = round(
                    inside - float(zz[pad][shell].mean()), 5) if shell.any() else ""
                row[f"z_{m}_mirror"] = round(inside - float(mm[pad][comp].mean()), 5)
            if "t1c" in modalities and "t1n" in modalities:
                row["enhancement"] = round(row["z_t1c_mean"] - row["z_t1n_mean"], 5)
            if "t2f" in modalities and "t2w" in modalities:
                row["flair_minus_t2"] = round(row["z_t2f_mean"] - row["z_t2w_mean"], 5)
            row["dist_to_brain_edge_mm"] = round(float(edge[pad][comp].max()), 3)
            row["frac_outside_brain"] = round(float((~brain[pad])[comp].mean()), 5)
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    class_values = paths.GLI_LABELS if args.dataset == "gli" else paths.MEN_LABELS
    modalities = GLI_MODALITIES if args.dataset == "gli" else MEN_MODALITIES
    pred_dir = paths.OUTPUTS / "runs" / args.run / f"pred_{args.split}"
    recs = [r for r in case_dicts(args.dataset, args.split)
            if (pred_dir / f"{r['case_id']}_pred.nii.gz").exists()]
    if args.limit:
        recs = recs[: args.limit]
    if not recs:
        raise SystemExit(f"no predicted masks in {pred_dir}")

    rows, t0 = [], time.time()
    for i, rec in enumerate(recs, 1):
        rows += case_rows(rec, pred_dir, class_values, modalities)
        print(f"  {i}/{len(recs)}  {rec['case_id']}  {len(rows)} components", flush=True)

    out = paths.REPORTS / f"image_features_{args.run}_{args.split}.csv"
    fields = KEY + feature_names(modalities)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} components in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
