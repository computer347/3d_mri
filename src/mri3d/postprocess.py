"""Drop spurious lesions from predicted masks.

The trained model segments the main tumour well but scatters extra blobs: on
the glioma test split it invented ~192 SNFH and ~162 WT lesions across 203
cases, roughly one per case. Volumetric Dice barely registers them - they are
small relative to the true mass - but lesion-wise Dice charges a full zero for
each, which is why SNFH scored 0.851 volumetric and 0.533 lesion-wise.

Two filters live here, and they differ in what they are allowed to look at.

``filter_small`` removes connected components below a size threshold. It is the
standard fix, it costs no retraining, and it mostly failed: size does not
separate real components from spurious ones, because the scoring already
ignores anything under 50 voxels, so every threshold large enough to remove an
invented lesion also deletes true ones. Its threshold is a *hyperparameter*,
swept on validation and only then applied to test.

``filter_by_selector`` (from ``mri3d.select``) scores each component with a
logistic regression over confidence, geometry and spatial context, and keeps it
if the posterior clears a threshold *derived* from the lesion-wise scoring
function rather than swept. It can only exist because ``predict.py
--save-components`` now preserves the softmax statistics that
``logits.argmax`` used to discard.

    python -m mri3d.postprocess --run gli_main --sweep        # choose size on val
    python -m mri3d.postprocess --run gli_main --apply 100    # score test with it
    python -m mri3d.postprocess --run gli_main --selector --split test
"""

from __future__ import annotations

import argparse
import csv
import json

import nibabel as nib
import numpy as np
from scipy import ndimage

from . import paths
from .data import case_dicts
from .metrics import score_case

# Per-class minimum component size in voxels. A single threshold for every class
# is a poor fit: SNFH lesions are large and numerous, NETC ones can be genuinely
# small, so the sweep reports per class and the applied value may differ.
DEFAULT_GRID = [0, 25, 50, 100, 200, 400, 800, 1600]


def filter_small(mask: np.ndarray, min_voxels: int | dict[int, int]) -> np.ndarray:
    """Zero out connected components smaller than the threshold, class by class.

    ``min_voxels`` is either one value for every class, or a per-class mapping
    ``{label: threshold}``. Per-class matters here: a single global threshold
    helps the large diffuse classes and destroys the focal ones, because NETC
    lesions are legitimately small.
    """
    per_class = min_voxels if isinstance(min_voxels, dict) else None
    if per_class is None and min_voxels <= 0:
        return mask
    out = mask.copy()
    for value in np.unique(mask):
        if value == 0:
            continue
        # `if per_class is not None`, not `if per_class`: an empty mapping means
        # "no class has a threshold", and testing it for truthiness instead sent
        # the dict itself down the scalar branch to be compared against 0.
        threshold = per_class.get(int(value), 0) if per_class is not None else min_voxels
        if threshold <= 0:
            continue
        binary = mask == value
        labelled, n = ndimage.label(binary)
        if n == 0:
            continue
        sizes = np.bincount(labelled.ravel())
        too_small = np.isin(labelled, np.where(sizes < threshold)[0]) & binary
        out[too_small] = 0
    return out


def per_class_thresholds(run: str, class_values: dict[int, str]) -> dict[int, int]:
    """Best threshold for each sub-region, read off the validation sweep.

    Only the four labelled sub-regions get a threshold; TC and WT are derived
    from them, so they cannot be filtered independently.
    """
    sweep = json.loads((paths.REPORTS / f"postprocess_sweep_{run}.json").read_text(encoding="utf-8"))
    out: dict[int, int] = {}
    for value, name in class_values.items():
        if value == 0 or name not in sweep[0]:
            continue
        best = max(sweep, key=lambda s: s[name]["lesion"])
        out[value] = int(best["min_voxels"])
    return out


def _load_pair(rec, pred_dir):
    pred = np.asanyarray(nib.as_closest_canonical(
        nib.load((pred_dir / f"{rec['case_id']}_pred.nii.gz").as_posix())).dataobj).astype(np.uint8)
    ref = np.asanyarray(nib.as_closest_canonical(
        nib.load(rec["label"])).dataobj).astype(np.uint8)
    return pred, ref


def selector_filter(run: str, split: str) -> dict:
    """Everything ``filter_by_selector`` needs, loaded once for a whole split.

    Returns the keep/drop decision per ``(case_id, class_name, comp_id)`` and
    the voxel count the components CSV recorded, which the filter re-checks
    against the mask on disk.
    """
    from . import select

    model = select.load_selector(run)
    comps = select.load_components(run, split)
    keep = select.keep_table(model["model"], comps, model["p_star"])
    sizes = {(c["case_id"], c["class_name"], c["comp_id"]): c["n_voxels"] for c in comps}
    return {"keep": keep, "sizes": sizes, "p_star": model["p_star"]}


def evaluate_rows(run: str, dataset: str, split: str, min_voxels: int | dict,
                  limit: int = 0, selector: dict | None = None) -> list[dict]:
    """Per-case scores after post-processing. ``selector`` overrides ``min_voxels``."""
    from .select import filter_by_selector

    pred_dir = paths.OUTPUTS / "runs" / run / f"pred_{split}"
    class_values = paths.GLI_LABELS if dataset == "gli" else paths.MEN_LABELS
    recs = [r for r in case_dicts(dataset, split)
            if (pred_dir / f"{r['case_id']}_pred.nii.gz").exists()]
    if limit:
        recs = recs[:limit]

    rows = []
    for rec in recs:
        pred, ref = _load_pair(rec, pred_dir)
        if selector is not None:
            pred = filter_by_selector(pred, rec["case_id"], class_values,
                                      selector["keep"], selector["sizes"])
        else:
            pred = filter_small(pred, min_voxels)
        rows.append({"case_id": rec["case_id"], **score_case(pred, ref, class_values)})
    return rows


def summarise(rows: list[dict], min_voxels: int | dict) -> dict:
    regions = [k[len("lesion_dice_"):] for k in rows[0] if k.startswith("lesion_dice_")]
    out: dict = {"n_cases": len(rows),
                 "min_voxels": min_voxels if not isinstance(min_voxels, dict)
                 else {str(k): v for k, v in min_voxels.items()}}
    for r in regions:
        present = [x for x in rows if x[f"present_{r}"]]
        out[r] = {
            "lesion": round(float(np.mean([x[f"lesion_dice_{r}"] for x in present])), 4),
            "volumetric": round(float(np.mean([x[f"dice_{r}"] for x in present])), 4),
            "false_lesions": int(sum(x[f"lesions_false_{r}"] for x in present)),
            "missed_lesions": int(sum(x[f"lesions_missed_{r}"] for x in present)),
            "n": len(present),
        }
    out["mean_lesion"] = round(float(np.mean([out[r]["lesion"] for r in regions])), 4)
    return out


def evaluate(run: str, dataset: str, split: str, min_voxels: int | dict,
             limit: int = 0, selector: dict | None = None) -> dict:
    return summarise(evaluate_rows(run, dataset, split, min_voxels, limit, selector), min_voxels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--sweep", action="store_true", help="sweep thresholds on the val split")
    ap.add_argument("--apply", type=int, default=None,
                    help="score the test split with this single threshold")
    ap.add_argument("--per-class", action="store_true",
                    help="use the per-class thresholds the sweep selected, checking them on "
                         "validation first and only then scoring test")
    ap.add_argument("--grid", type=int, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--selector", action="store_true",
                    help="filter with the fitted lesion selector (reports/selector_{run}.json) "
                         "instead of a size threshold")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="which split --selector scores")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        print(f"sweeping minimum lesion size on the VALIDATION split ({args.run})\n")
        results = []
        for mv in args.grid:
            res = evaluate(args.run, args.dataset, "val", mv, args.limit)
            results.append(res)
            regions = [k for k in res if k not in ("n_cases", "min_voxels", "mean_lesion")]
            fp = sum(res[r]["false_lesions"] for r in regions)
            fn = sum(res[r]["missed_lesions"] for r in regions)
            print(f"  min {mv:>5} voxels: mean lesion-Dice {res['mean_lesion']:.4f}  "
                  f"false {fp:>4}  missed {fn:>4}", flush=True)
        best = max(results, key=lambda r: r["mean_lesion"])
        print(f"\nbest on validation: {best['min_voxels']} voxels "
              f"(mean lesion-Dice {best['mean_lesion']:.4f})")
        out = paths.REPORTS / f"postprocess_sweep_{args.run}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")
        print(f"\nnow apply it to test:  python -m mri3d.postprocess --run {args.run} "
              f"--apply {best['min_voxels']}")

    if args.per_class:
        class_values = paths.GLI_LABELS if args.dataset == "gli" else paths.MEN_LABELS
        thresholds = per_class_thresholds(args.run, class_values)
        named = {class_values[k]: v for k, v in thresholds.items()}
        print(f"per-class thresholds selected on validation: {named}\n")

        for split in ("val", "test"):
            before = evaluate(args.run, args.dataset, split, 0, args.limit)
            after = evaluate(args.run, args.dataset, split, thresholds, args.limit)
            regions = [k for k in after if k not in ("n_cases", "min_voxels", "mean_lesion")]
            print(f"--- {split.upper()} ({before['n_cases']} cases)")
            print(f"{'region':<6} {'before':>8} {'after':>8} {'change':>8}   {'false':>6} -> {'':>4}")
            for r in regions:
                d = after[r]["lesion"] - before[r]["lesion"]
                print(f"{r:<6} {before[r]['lesion']:>8.4f} {after[r]['lesion']:>8.4f} {d:>+8.4f}   "
                      f"{before[r]['false_lesions']:>6} -> {after[r]['false_lesions']:<4}")
            print(f"{'mean':<6} {before['mean_lesion']:>8.4f} {after['mean_lesion']:>8.4f} "
                  f"{after['mean_lesion'] - before['mean_lesion']:>+8.4f}\n", flush=True)
            path = paths.REPORTS / f"scores_{args.run}_{split}_perclass.json"
            path.write_text(json.dumps({"thresholds": named, "before": before, "after": after},
                                       indent=2), encoding="utf-8")
        print("wrote per-split JSON to reports/")

    if args.selector:
        sel = selector_filter(args.run, args.split)
        print(f"\nscoring the {args.split.upper()} split with the learned selector, "
              f"p* = {sel['p_star']}")
        before = evaluate(args.run, args.dataset, args.split, 0, args.limit)
        after = evaluate(args.run, args.dataset, args.split, 0, args.limit, selector=sel)
        regions = [k for k in after if k not in ("n_cases", "min_voxels", "mean_lesion")]
        print(f"\n{'region':<6} {'lesion before':>14} {'after':>8} {'change':>8}   "
              f"{'false before':>13} {'after':>6}  {'missed':>7} {'after':>6}")
        for r in regions:
            d = after[r]["lesion"] - before[r]["lesion"]
            print(f"{r:<6} {before[r]['lesion']:>14.4f} {after[r]['lesion']:>8.4f} "
                  f"{d:>+8.4f}   {before[r]['false_lesions']:>13} {after[r]['false_lesions']:>6}"
                  f"  {before[r]['missed_lesions']:>7} {after[r]['missed_lesions']:>6}")
        print(f"\nmean lesion-Dice {before['mean_lesion']:.4f} -> {after['mean_lesion']:.4f} "
              f"({after['mean_lesion'] - before['mean_lesion']:+.4f})")
        path = paths.REPORTS / f"scores_{args.run}_{args.split}_selector.json"
        path.write_text(json.dumps({"p_star": sel["p_star"], "before": before, "after": after},
                                   indent=2), encoding="utf-8")
        print(f"wrote {path}")

    if args.apply is not None:
        print(f"\nscoring the TEST split with min lesion size {args.apply} voxels")
        before = evaluate(args.run, args.dataset, "test", 0, args.limit)
        after = evaluate(args.run, args.dataset, "test", args.apply, args.limit)
        regions = [k for k in after if k not in ("n_cases", "min_voxels", "mean_lesion")]
        print(f"\n{'region':<6} {'lesion before':>14} {'after':>8} {'change':>8}   "
              f"{'false before':>13} {'after':>6}")
        for r in regions:
            d = after[r]["lesion"] - before[r]["lesion"]
            print(f"{r:<6} {before[r]['lesion']:>14.4f} {after[r]['lesion']:>8.4f} "
                  f"{d:>+8.4f}   {before[r]['false_lesions']:>13} {after[r]['false_lesions']:>6}")
        print(f"\nmean lesion-Dice {before['mean_lesion']:.4f} -> {after['mean_lesion']:.4f} "
              f"({after['mean_lesion'] - before['mean_lesion']:+.4f})")

        path = paths.REPORTS / f"scores_{args.run}_test_postprocessed.json"
        path.write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
