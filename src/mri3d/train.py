"""Training loop.

Validation runs on *whole* volumes with a sliding window, not on patches: patch
scores flatter the model, because a patch sampled around a tumour is an easier
problem than a whole head where most of the volume can produce false positives.
The checkpoint is selected on mean validation Dice, and the test split is never
touched here.

Debug first, then train:
    python -m mri3d.train --overfit 5 --epochs 40   # must reach Dice ~1.0
    python -m mri3d.train --epochs 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

from . import paths
from .data import PATCH, loaders
from .metrics import dice
from .model import build_loss, build_model, describe
from .seeding import set_seed


def validate(model, loader, device, n_classes, patch, amp_dtype) -> dict:
    model.eval()
    per_class: list[list[float]] = [[] for _ in range(n_classes - 1)]
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = sliding_window_inference(
                    image, patch, sw_batch_size=1, predictor=model, overlap=0.25, mode="gaussian",
                )
            pred = logits.argmax(dim=1, keepdim=True).cpu().numpy()
            ref = label.cpu().numpy()
            for v in range(1, n_classes):
                per_class[v - 1].append(dice(pred == v, ref == v))
    return {"per_class": [float(np.mean(c)) if c else 0.0 for c in per_class],
            "mean": float(np.mean([np.mean(c) for c in per_class if c]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--width", type=int, default=16, help="SegResNet init_filters")
    ap.add_argument("--patch", type=int, nargs=3, default=list(PATCH))
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--overfit", type=int, default=0,
                    help="debug: train and validate on the same N cases; Dice must reach ~1.0")
    ap.add_argument("--cache-rate", type=float, default=0.0)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--mode", default="patch", choices=["patch", "full"],
                    help="'full' trains on whole volumes; only fits at --width 8")
    ap.add_argument("--train-limit", type=int, default=0,
                    help="use only the first N training cases (for A/B comparisons)")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="validate on only the first N val cases (keeps A/B runs comparable)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for python/numpy/torch/MONAI; recorded in the checkpoint")
    args = ap.parse_args()

    # Seed before anything that draws: weight init, the augmentation transforms
    # and the class-balanced patch sampler all consume randomness. See
    # mri3d.seeding for what this does and does not guarantee.
    set_seed(args.seed)

    if args.mode == "full" and args.width > 8:
        print(f"warning: full-volume training at width {args.width} needs ~9.9 GB and will "
              f"spill to system RAM (~15 s/step). Width 8 fits in 5.4 GB at 0.36 s/step.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    patch = tuple(args.patch)
    run = args.run_name or (f"overfit{args.overfit}" if args.overfit else
                            time.strftime(f"{args.dataset}-%Y%m%d-%H%M"))
    out_dir = paths.OUTPUTS / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, n_classes = loaders(
        args.dataset, batch_size=args.batch_size, workers=args.workers,
        cache_rate=args.cache_rate, limit=args.overfit, patch=patch,
        mode=args.mode, val_limit=args.val_limit,
    )
    if args.train_limit and not args.overfit:
        train_loader.dataset.data = train_loader.dataset.data[: args.train_limit]
    # A full-volume model is validated in one window covering the whole head;
    # a patch model slides its training-sized window across it.
    val_roi = (192, 224, 192) if args.mode == "full" else patch
    in_channels = 4 if args.dataset == "gli" else 1
    model = build_model(n_classes, in_channels, args.width).to(device)
    loss_fn = build_loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    shape_note = f"patch {patch}" if args.mode == "patch" else "full volumes"
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    class_names = list((paths.GLI_LABELS if args.dataset == "gli"
                        else paths.MEN_LABELS).values())[1:]

    print(f"run {run} | {describe(model)} | {shape_note} | {n_classes} classes")
    print(f"tensorboard --logdir {paths.OUTPUTS / 'runs'}")
    print(f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | device {device}")
    if args.overfit:
        print("OVERFIT MODE: train == val. Dice near 1.0 proves the pipeline; "
              "anything less is a bug, not a hyperparameter.")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, t0 = [], time.time()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = loss_fn(model(image), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        line = {"epoch": epoch, "loss": float(np.mean(losses)),
                "lr": sched.get_last_lr()[0], "secs": round(time.time() - t0, 1)}
        writer.add_scalar("train/loss", line["loss"], epoch)
        writer.add_scalar("train/lr", line["lr"], epoch)
        writer.add_scalar("train/epoch_seconds", line["secs"], epoch)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            v = validate(model, val_loader, device, n_classes, val_roi, amp_dtype)
            line.update(val_mean=v["mean"], val_per_class=v["per_class"])
            writer.add_scalar("val/dice_mean", v["mean"], epoch)
            for name, d in zip(class_names, v["per_class"]):
                writer.add_scalar(f"val/dice_{name}", d, epoch)
            if v["mean"] > best:
                best = v["mean"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "n_classes": n_classes, "in_channels": in_channels,
                            "epoch": epoch, "val_mean": best}, out_dir / "best.pt")
                line["saved"] = True
            peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            print(f"epoch {epoch:>4} loss {line['loss']:.4f} "
                  f"val {v['mean']:.4f} [" + " ".join(f"{d:.3f}" for d in v["per_class"]) + "]"
                  f" {line['secs']}s peak {peak:.1f}GB" + ("  *saved" if line.get("saved") else ""),
                  flush=True)
        else:
            print(f"epoch {epoch:>4} loss {line['loss']:.4f} {line['secs']}s", flush=True)

        history.append(line)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    writer.close()
    print(f"\nbest validation Dice {best:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
