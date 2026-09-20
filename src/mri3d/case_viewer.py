"""Per-case viewer: what the model called a tumour, against what a human drew.

For each selected case this packs three aligned mosaics - the T1c scan, the
reference mask, and the model's prediction - into one self-contained HTML page,
together with that case's scores. The page can show them as a plain overlay or
as an agreement map (correct / missed / false alarm), which is the view that
actually explains a Dice number.

Only the slices around the tumour are included; empty head slices would triple
the page size and show nothing.

    python -m mri3d.case_viewer --run gli-20260920-0400 --split val --n 8
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from . import paths
from .data import case_dicts
from .metrics import score_case

CONTEXT_SLICES = 6  # keep this many empty slices either side of the tumour


def _mosaic(stack: np.ndarray, scale_to_255: bool = False) -> tuple[Image.Image, dict]:
    """Pack (n, h, w) into a grid PNG."""
    n, h, w = stack.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    sheet = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for k in range(n):
        r, c = divmod(k, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = stack[k]
    return Image.fromarray(sheet, mode="L"), {
        "n_slices": n, "tile_w": w, "tile_h": h, "cols": cols, "rows": rows,
    }


def _prep_case(rec: dict, pred_path: Path, class_values: dict[int, str]) -> tuple[dict, dict]:
    # Reorient everything to canonical RAS before comparing. The scans are stored
    # LAS and the predictions are written in RAS; comparing the raw arrays would
    # silently mirror one against the other (it did - see git history).
    t1c_path = [p for p in rec["image"] if "t1c" in p][0]
    t1c = np.asanyarray(nib.as_closest_canonical(nib.load(t1c_path)).dataobj, dtype=np.float32)
    gt = np.asanyarray(nib.as_closest_canonical(nib.load(rec["label"])).dataobj).astype(np.uint8)
    pred = np.asanyarray(
        nib.as_closest_canonical(nib.load(pred_path.as_posix())).dataobj).astype(np.uint8)

    # Axial slices (z is the last axis in this space), limited to the tumour region.
    interesting = np.where(((gt > 0) | (pred > 0)).any(axis=(0, 1)))[0]
    if interesting.size:
        lo = max(0, int(interesting.min()) - CONTEXT_SLICES)
        hi = min(t1c.shape[2], int(interesting.max()) + CONTEXT_SLICES + 1)
    else:
        mid = t1c.shape[2] // 2
        lo, hi = mid - 20, mid + 20

    def stack(vol, rotate=True):
        sl = [np.rot90(vol[:, :, z]) for z in range(lo, hi)]
        return np.stack(sl)

    hi_val = np.percentile(t1c[t1c > 0], 99) if (t1c > 0).any() else 1.0
    img = np.clip(stack(t1c) / max(hi_val, 1e-6), 0, 1) * 255
    assets = {
        "scan": _mosaic(img.astype(np.uint8)),
        # Masks carry the class value itself (0-4), so the page decodes them exactly.
        "truth": _mosaic(stack(gt)),
        "pred": _mosaic(stack(pred)),
    }
    meta = {
        "case_id": rec["case_id"],
        "slice_offset": lo,
        "geom": assets["scan"][1],
        "scores": score_case(pred, gt, class_values),
    }
    return meta, {k: v[0] for k, v in assets.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=8, help="cases to include")
    ap.add_argument("--pick", default="spread",
                    choices=["spread", "best", "worst", "first"],
                    help="'spread' shows best, median and worst together - the honest sample")
    args = ap.parse_args()

    pred_dir = paths.OUTPUTS / "runs" / args.run / f"pred_{args.split}"
    if not pred_dir.exists():
        raise SystemExit(f"no predictions at {pred_dir}\n"
                         f"run: python -m mri3d.predict --run {args.run} "
                         f"--split {args.split} --save-masks")

    class_values = paths.GLI_LABELS if args.dataset == "gli" else paths.MEN_LABELS
    records = [r for r in case_dicts(args.dataset, args.split)
               if (pred_dir / f"{r['case_id']}_pred.nii.gz").exists()]
    if not records:
        raise SystemExit(f"no predicted masks found in {pred_dir}")

    print(f"scoring {len(records)} predicted cases…")
    prepared = []
    for rec in records:
        meta, imgs = _prep_case(rec, pred_dir / f"{rec['case_id']}_pred.nii.gz", class_values)
        lw = [v for k, v in meta["scores"].items() if k.startswith("lesion_dice_")]
        meta["mean_lesion_dice"] = round(float(np.mean(lw)), 4)
        prepared.append((meta, imgs))
    prepared.sort(key=lambda p: p[0]["mean_lesion_dice"])

    n = min(args.n, len(prepared))
    if args.pick == "worst":
        chosen = prepared[:n]
    elif args.pick == "best":
        chosen = prepared[-n:]
    elif args.pick == "first":
        chosen = prepared[:n]
    else:  # spread: evenly sampled across the score range
        idx = np.linspace(0, len(prepared) - 1, n).round().astype(int)
        chosen = [prepared[i] for i in dict.fromkeys(idx.tolist())]

    assets: dict[str, str] = {}
    cases = []
    for meta, imgs in chosen:
        for kind, img in imgs.items():
            key = f"{meta['case_id']}_{kind}"
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)
            assets[key] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        cases.append(meta)

    page_meta = {
        "run": args.run,
        "split": args.split,
        "dataset": args.dataset,
        # Sub-regions carry a label value (the overlay colours them); the
        # composites TC and WT are scored but not drawn, so they have none.
        "classes": ([{"value": v, "name": n_} for v, n_ in class_values.items() if v]
                    + ([{"value": None, "name": r} for r in ("TC", "WT")]
                       if args.dataset == "gli" else [])),
        "cases": cases,
    }
    template = (Path(__file__).parent / "case_viewer_template.html").read_text(encoding="utf-8")
    html = (template
            .replace("__META_JSON__", json.dumps(page_meta))
            .replace("__ASSETS_JSON__", json.dumps(assets)))
    out = paths.REPORTS / f"predictions_{args.run}_{args.split}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB, {len(cases)} cases)")
    for c in cases:
        print(f"  {c['case_id']}  mean lesion-Dice {c['mean_lesion_dice']}")


if __name__ == "__main__":
    main()
