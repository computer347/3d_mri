"""Training-curve report.

Reads the ``history.json`` a run writes each epoch and draws it as a standalone
page: loss against epoch, and per-class validation Dice, so the shape of a run
is legible without starting TensorBoard. Multiple runs can be charted together,
which is how the patch-vs-whole-volume arms were compared.

    python -m mri3d.curves --runs gli-20260920-0400 ab_A_patch_w16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import paths

# Categorical slots in fixed order, with a step per theme. Both columns pass the
# six checks (lightness band, chroma floor, CVD separation, normal-vision floor,
# contrast) against their own surface - verified with the dataviz validator, not
# by eye. My first attempt failed twice: a teal that read as grey, then a blue
# and violet only 3.7 deltaE apart under deuteranopia.
RUN_COLOURS = [  # (light, dark)
    ("#2a78d6", "#3987e5"),  # blue
    ("#eb6834", "#d95926"),  # orange
    ("#1baf7a", "#199e70"),  # aqua
    ("#eda100", "#c98500"),  # yellow
]
# Tumour-class hues, shared with the atlas and case viewer so one colour means
# one tissue type across every page. These sit on a near-black scan rather than
# a chart surface, so they are chosen for separation on that ground; their worst
# adjacent CVD separation is 16.3 deltaE, comfortably clear.
CLASS_COLOURS = {"NETC": "#3b82f6", "SNFH": "#eab308", "ET": "#ef4444",
                 "RC": "#a855f7", "GTV": "#ef4444"}


def load_history(run: str) -> dict:
    path = paths.OUTPUTS / "runs" / run / "history.json"
    if not path.exists():
        raise SystemExit(f"no history at {path}")
    history = json.loads(path.read_text(encoding="utf-8"))
    ckpt = paths.OUTPUTS / "runs" / run / "best.pt"
    dataset = "gli"
    for part in run.split("-") + run.split("_"):
        if part in ("men", "men_rt", "menrt"):
            dataset = "men_rt"
    classes = list((paths.GLI_LABELS if dataset == "gli" else paths.MEN_LABELS).values())[1:]
    return {
        "run": run,
        "classes": classes,
        "epochs": [h["epoch"] for h in history],
        "loss": [h["loss"] for h in history],
        "lr": [h.get("lr") for h in history],
        "secs": [h.get("secs") for h in history],
        "val": [{"epoch": h["epoch"], "mean": h["val_mean"], "per_class": h["val_per_class"]}
                for h in history if "val_mean" in h],
        "has_checkpoint": ckpt.exists(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--title", default="Training Curves")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = [load_history(r) for r in args.runs]
    if len(runs) > len(RUN_COLOURS):
        raise SystemExit(f"charting more than {len(RUN_COLOURS)} runs would need cycled "
                         f"hues, which stop being distinguishable; chart them in groups")
    for i, r in enumerate(runs):
        r["colour"], r["colour_dark"] = RUN_COLOURS[i]

    meta = {"title": args.title, "runs": runs, "class_colours": CLASS_COLOURS}
    template = (Path(__file__).parent / "curves_template.html").read_text(encoding="utf-8")
    html = template.replace("__META_JSON__", json.dumps(meta))

    out = Path(args.out) if args.out else paths.REPORTS / f"curves_{'_vs_'.join(args.runs)}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e3:.0f} KB)")
    for r in runs:
        best = max((v["mean"] for v in r["val"]), default=float("nan"))
        print(f"  {r['run']}: {len(r['epochs'])} epochs, best val Dice {best:.4f}")


if __name__ == "__main__":
    main()
