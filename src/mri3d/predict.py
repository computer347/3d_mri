"""Run a trained model over whole cases and score it.

Inference is sliding-window regardless of how training was done: the window
overlaps by 25% and blends with a Gaussian, so voxels near a window edge - where
the model has least context - are weighted down rather than producing visible
seams between windows.

    python -m mri3d.predict --run gli-20260920-0400 --split test --save-masks
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from . import paths
from .data import case_dicts, val_transforms
from .metrics import score_case
from .model import build_model


def load_run(run: str, device):
    ckpt = torch.load(paths.OUTPUTS / "runs" / run / "best.pt", map_location=device,
                      weights_only=False)
    model = build_model(ckpt["n_classes"], ckpt["in_channels"],
                        ckpt["args"].get("width", 16)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory under outputs/runs/")
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-masks", action="store_true",
                    help="write predicted masks as .nii.gz (needed by the case viewer)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_run(args.run, device)
    patch = tuple(ckpt["args"].get("patch", (128, 128, 128)))
    class_values = (paths.GLI_LABELS if args.dataset == "gli" else paths.MEN_LABELS)

    files = case_dicts(args.dataset, args.split)
    if args.limit:
        files = files[: args.limit]
    tf = val_transforms(ckpt["n_classes"])

    pred_dir = paths.OUTPUTS / "runs" / args.run / f"pred_{args.split}"
    if args.save_masks:
        pred_dir.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for i, rec in enumerate(files, 1):
        data = tf(rec)
        image = data["image"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            logits = sliding_window_inference(image, patch, 1, model, overlap=0.25, mode="gaussian")
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        ref = data["label"][0].cpu().numpy().astype(np.uint8)

        row = {"case_id": rec["case_id"]}
        row.update(score_case(pred, ref, class_values))
        rows.append(row)

        if args.save_masks:
            # The prediction is in the transform's frame (Orientationd -> RAS), which
            # is NOT the frame the file on disk uses: BraTS GLI is stored LAS, so the
            # array has been flipped left-right along the way. Writing it with the
            # original affine would produce a mask that is silently mirrored - the
            # tumour on the wrong side of the head. Save the affine that matches the
            # array we actually produced.
            affine = np.asarray(data["label"].affine, dtype=np.float64)
            nib.save(nib.Nifti1Image(pred, affine), pred_dir / f"{rec['case_id']}_pred.nii.gz")

        done = f"{i}/{len(files)}"
        mean_lw = np.mean([v for k, v in row.items() if k.startswith("lesion_dice_")])
        print(f"  {done:>9}  {rec['case_id']}  lesion-Dice {mean_lw:.3f}", flush=True)

    out_csv = paths.REPORTS / f"scores_{args.run}_{args.split}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for key in rows[0]:
        if key.startswith(("dice_", "lesion_dice_")):
            name = key.split("_")[-1]
            # Average only over cases where the class is actually present, and
            # report that count: a mean over all cases is dominated by the
            # trivially correct empty ones.
            vals = [r[key] for r in rows if r.get(f"present_{name}")]
            summary[key] = {"mean_where_present": round(float(np.mean(vals)), 4) if vals else None,
                            "n_cases": len(vals),
                            "mean_all_cases": round(float(np.mean([r[key] for r in rows])), 4)}
    (paths.REPORTS / f"summary_{args.run}_{args.split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} cases in {time.time() - t0:.0f}s -> {out_csv}")
    for k, v in summary.items():
        if k.startswith("lesion_dice_"):
            print(f"  {k:<22} {v['mean_where_present']}  (n={v['n_cases']} with the class present)")


if __name__ == "__main__":
    main()
