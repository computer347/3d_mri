"""One pass over the dataset, producing everything that needs raw voxels.

Reading the imagery costs ~45 GB of I/O, so this script does all of it at once:

1. **Per-case statistics** -> ``reports/case_index.csv`` (class volumes, tumour
   centroid, bounding box, intensity range). Small enough to commit.
2. **Tumour-location heatmaps** -> ``outputs/heatmaps.npz``: per-class voxel
   frequency summed over cases, at half resolution. Only meaningful for the
   glioma set, whose cases all sit in the same 182x218x182 SRI24 space.
3. **Fingerprints** -> ``outputs/embeddings.npz``: an MD5 of each file (exact
   duplicates) and a 16x16x16 normalised thumbnail of the T1c (near-duplicates).
   Consumed by ``mri3d.duplicates``.

Run: ``python -m mri3d.scan --which all --workers 6``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from . import paths
from .index import Case, load_cases

THUMB = 16  # edge length of the near-duplicate fingerprint
GLI_SHAPE = (182, 218, 182)  # heatmaps only accumulate cases with this shape
HEAT_SHAPE = tuple(s // 2 for s in GLI_SHAPE)
N_GLI_CLASSES = 5


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _thumbnail(vol: np.ndarray) -> np.ndarray:
    """Shape- and intensity-invariant fingerprint of a volume.

    Resampling to a fixed 16^3 grid makes scans of different matrix sizes
    comparable; z-scoring over the head voxels makes them invariant to scanner
    gain, so the same brain acquired twice still matches.
    """
    vol = vol.astype(np.float32)
    small = ndimage.zoom(vol, np.divide(THUMB, vol.shape), order=1, prefilter=False)
    head = small[small > 0]
    if head.size == 0:
        return np.zeros(THUMB**3, dtype=np.float32)
    small = (small - head.mean()) / (head.std() + 1e-6)
    return small.ravel().astype(np.float32)


def _block_any_half(mask: np.ndarray) -> np.ndarray:
    """Downsample a binary mask 2x: 182x218x182 -> 91x109x91 (all dims even).

    Takes the max over each 2x2x2 block, not the sum, so one case contributes
    at most 1 to a cell. The accumulated heatmap is then literally "how many
    cases had this class here", which is what the viewer reports.
    """
    a, b, c = HEAT_SHAPE
    return mask.reshape(a, 2, b, 2, c, 2).max(axis=(1, 3, 5))


def _scan_one(case: Case) -> tuple[dict, np.ndarray | None, np.ndarray]:
    t1c_path = case.images["t1c"]
    img = nib.load(t1c_path)
    vol = np.asanyarray(img.dataobj, dtype=np.float32)

    row: dict = {
        "dataset": case.dataset,
        "subset": case.subset,
        "case_id": case.case_id,
        "patient": case.patient,
        "timepoint": case.timepoint,
        "shape": "x".join(str(s) for s in vol.shape),
        "spacing": "x".join(f"{z:.2f}" for z in img.header.get_zooms()[:3]),
        "t1c_max": float(vol.max()),
        "t1c_nonzero_frac": float((vol > 0).mean()),
        "t1c_md5": _md5(t1c_path),
        # Hash of the decoded voxels, not the file. Two .nii.gz files can hold
        # identical imagery yet differ on disk (different gzip settings), so the
        # file hash alone misses real duplicates - it does in BraTS-MEN-RT.
        "t1c_vox_md5": hashlib.md5(np.ascontiguousarray(vol).tobytes()).hexdigest(),
        "has_label": case.has_label,
    }

    heat = None
    if case.has_label:
        seg = np.asanyarray(nib.load(case.label).dataobj).astype(np.uint8)
        n_classes = N_GLI_CLASSES if case.dataset == "gli" else 2
        counts = np.bincount(seg.ravel(), minlength=n_classes)[:n_classes]
        for value in range(1, n_classes):
            name = (paths.GLI_LABELS if case.dataset == "gli" else paths.MEN_LABELS)[value]
            row[f"vox_{name}"] = int(counts[value])
        row["vox_total"] = int(seg.size)

        fg = seg > 0
        row["vox_tumour"] = int(fg.sum())
        if row["vox_tumour"]:
            idx = np.argwhere(fg)
            row["centroid"] = ",".join(f"{v:.1f}" for v in idx.mean(0))
            lo, hi = idx.min(0), idx.max(0)
            row["bbox"] = ",".join(f"{a}:{b}" for a, b in zip(lo, hi))
        if case.dataset == "gli" and seg.shape == GLI_SHAPE:
            heat = np.stack(
                [_block_any_half((seg == v).astype(np.uint16)) for v in range(1, N_GLI_CLASSES)]
            ).astype(np.uint16)

    return row, heat, _thumbnail(vol)


def _scan_shard(cases: list[Case]) -> tuple[list[dict], np.ndarray, list[str], np.ndarray]:
    """Process a shard in one worker, accumulating heatmaps locally.

    Workers return one summed heatmap per shard rather than one per case, which
    keeps what crosses the process boundary small.
    """
    rows: list[dict] = []
    accum = np.zeros((N_GLI_CLASSES - 1, *HEAT_SHAPE), dtype=np.uint32)
    ids: list[str] = []
    thumbs: list[np.ndarray] = []
    for case in cases:
        row, heat, thumb = _scan_one(case)
        rows.append(row)
        if heat is not None:
            accum += heat
        ids.append(case.case_id)
        thumbs.append(thumb)
    return rows, accum, ids, np.stack(thumbs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="all",
                    choices=["all", "gli_train", "gli_val", "men_train"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N cases")
    args = ap.parse_args()

    paths.ensure_dirs()
    cases = load_cases(args.which)
    if args.limit:
        cases = cases[: args.limit]
    print(f"scanning {len(cases)} cases with {args.workers} workers", flush=True)

    shards = [cases[i :: args.workers] for i in range(args.workers)]
    t0 = time.time()
    rows: list[dict] = []
    accum = np.zeros((N_GLI_CLASSES - 1, *HEAT_SHAPE), dtype=np.uint32)
    ids: list[str] = []
    thumbs: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (r, a, i, t) in enumerate(pool.map(_scan_shard, shards), 1):
            rows.extend(r)
            accum += a
            ids.extend(i)
            thumbs.append(t)
            print(f"  shard {done}/{len(shards)} ({len(rows)} cases, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: (r["dataset"], r["subset"], r["case_id"]))
    fields: list[str] = []
    for r in rows:  # union of keys, stable order (MEN rows lack the GLI classes)
        fields += [k for k in r if k not in fields]
    with open(paths.CASE_INDEX, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    np.savez_compressed(paths.HEATMAPS, heat=accum,
                        classes=np.array(paths.GLI_CLASSES), shape=np.array(GLI_SHAPE))
    np.savez_compressed(paths.EMBEDDINGS, ids=np.array(ids), thumbs=np.concatenate(thumbs))

    n_heat = sum(1 for r in rows if r["dataset"] == "gli" and r["has_label"])
    print(f"\ndone in {time.time() - t0:.0f}s")
    print(f"  {paths.CASE_INDEX}  ({len(rows)} cases)")
    print(f"  {paths.HEATMAPS}    (accumulated over {n_heat} labelled glioma cases)")
    print(f"  {paths.EMBEDDINGS}  ({len(ids)} fingerprints)")


if __name__ == "__main__":
    main()
