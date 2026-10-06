"""What the model actually gets wrong, and whether it is one problem or two.

``lesion_wise_dice`` charges a full zero for a predicted component that matches
no reference lesion. It does not ask *why* the component is unmatched, and the
answer turns out to matter more than anything else measured in this project.

Two things get counted identically and are not the same:

1. **An invented lesion.** The model marked tissue the reference calls
   background. Deleting it is the right response, and that is what
   ``mri3d.select`` does.
2. **A mislabelled sub-region.** The model found real tumour and called it
   NETC where the reference says SNFH. Nothing was invented. Lesion-wise Dice
   charges *twice* for this - a zero for the "spurious" component, and another
   zero for the reference lesion that is now "missed" - and deleting the
   component fixes half of that by throwing away correctly detected tumour.

This module tabulates which of the two each failure is, then attributes the
deficit with three oracles that each fix exactly one thing:

    relabel   rename every false-positive component that sits mostly on
              reference tumour to that sub-region. Shapes untouched.
    delete    remove every false-positive component (the selector's ceiling).
    both      relabel where the tissue is real, delete where it is not.

    python -m mri3d.failure_anatomy --run gli_main --split test
    python -m mri3d.failure_anatomy --run gli_main --split val --no-oracles
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from . import paths
from .data import case_dicts
from .metrics import match_components, score_case
from .postprocess import _load_pair, summarise

# A component counts as sitting "on" reference tumour when at least this much
# of it does. Half is the natural cut: below it, calling the component a
# renamed lesion rather than an invented one would be the minority reading of
# its own voxels.
TUMOUR_MAJORITY = 0.5


def analyse(run: str, dataset: str, split: str, oracles: bool = True,
            limit: int = 0) -> dict:
    class_values = paths.GLI_LABELS if dataset == "gli" else paths.MEN_LABELS
    names = [n for v, n in class_values.items() if v != 0]
    pred_dir = paths.OUTPUTS / "runs" / run / f"pred_{split}"
    recs = [r for r in case_dicts(dataset, split)
            if (pred_dir / f"{r['case_id']}_pred.nii.gz").exists()]
    if limit:
        recs = recs[:limit]

    fp_is = defaultdict(Counter)     # fp class -> what the reference calls it
    fn_is = defaultdict(Counter)     # missed class -> what we predicted there
    fp_frac = defaultdict(list)
    fp_by_case, fn_by_case = Counter(), Counter()
    rows = {k: [] for k in ("none", "relabel", "delete", "both")}
    renamed = removed = 0

    for n, rec in enumerate(recs, 1):
        pred, ref = _load_pair(rec, pred_dir)
        relab, dele, both = pred.copy(), pred.copy(), pred.copy()
        for value, name in class_values.items():
            if value == 0:
                continue
            p, r = pred == value, ref == value
            if not p.any() and not r.any():
                continue
            m = match_components(p, r)

            for c in m.comps:
                if not c.false_positive:
                    continue
                blob = m.pred_lab == c.comp_id
                vals, counts = np.unique(ref[blob], return_counts=True)
                hit = {int(x): int(k) for x, k in zip(vals, counts)}
                tumour = c.n_voxels - hit.get(0, 0)
                frac = tumour / c.n_voxels
                fp_frac[name].append(frac)
                fp_by_case[rec["case_id"]] += 1
                removed += 1
                if frac >= TUMOUR_MAJORITY:
                    target = max((k for k in hit if k != 0), key=lambda k: hit[k])
                    fp_is[name][class_values[target]] += 1
                    relab[blob] = target
                    both[blob] = target
                    renamed += 1
                else:
                    fp_is[name]["background"] += 1
                    both[blob] = 0
                dele[blob] = 0

            for lesion in m.scored_refs:
                if lesion.detected:
                    continue
                fn_by_case[rec["case_id"]] += 1
                blob = m.ref_lab == lesion.ref_id
                vals, counts = np.unique(pred[blob], return_counts=True)
                hit = {int(x): int(k) for x, k in zip(vals, counts)}
                covered = lesion.n_voxels - hit.get(0, 0)
                if covered / lesion.n_voxels >= TUMOUR_MAJORITY:
                    other = max((k for k in hit if k != 0), key=lambda k: hit[k])
                    fn_is[name][class_values[other]] += 1
                else:
                    fn_is[name]["nothing"] += 1

        if oracles:
            for key, mask in (("none", pred), ("relabel", relab),
                              ("delete", dele), ("both", both)):
                rows[key].append(score_case(mask, ref, class_values))
        print(f"  {n}/{len(recs)}  {rec['case_id']}", flush=True)

    out: dict = {
        "run": run, "split": split, "n_cases": len(recs),
        "false_positives": {n: {"total": int(sum(fp_is[n].values())),
                                "reference_says": dict(fp_is[n]),
                                "mean_tumour_voxel_fraction":
                                    round(float(np.mean(fp_frac[n])), 3) if fp_frac[n] else None}
                            for n in names if fp_is[n]},
        "missed_lesions": {n: {"total": int(sum(fn_is[n].values())),
                               "predicted_as": dict(fn_is[n])}
                           for n in names if fn_is[n]},
        "n_false_positives_renamed": renamed,
        "n_false_positives_total": removed,
    }
    counts = np.array(sorted(fp_by_case.values(), reverse=True))
    out["concentration"] = {
        "cases_with_no_false_positive": len(recs) - len(fp_by_case),
        "cases_with_a_missed_lesion": len(fn_by_case),
        "cases_with_both": sum(1 for c in fp_by_case if fn_by_case.get(c)),
        "share_held_by_worst": {
            f"{int(f*100)}%": round(float(counts[:max(1, int(len(recs)*f))].sum() / counts.sum()), 3)
            for f in (0.1, 0.2, 0.5)} if counts.size else {},
    }
    if oracles:
        out["attribution"] = {k: summarise(v, k) for k, v in rows.items()}
    return out


def _print(out: dict) -> None:
    print(f"\n=== {out['split'].upper()}: what the reference says is under each "
          f"FALSE POSITIVE ===")
    for name, d in out["false_positives"].items():
        parts = ", ".join(f"{k} {v} ({v/d['total']:.0%})"
                          for k, v in sorted(d["reference_says"].items(),
                                             key=lambda kv: -kv[1]))
        print(f"  {name:5} n={d['total']:>4}  {parts}")
        print(f"  {'':5} {'':>6}  mean tumour-voxel fraction "
              f"{d['mean_tumour_voxel_fraction']}")

    print(f"\n=== {out['split'].upper()}: what was predicted where a lesion was MISSED ===")
    for name, d in out["missed_lesions"].items():
        parts = ", ".join(f"{k} {v} ({v/d['total']:.0%})"
                          for k, v in sorted(d["predicted_as"].items(), key=lambda kv: -kv[1]))
        print(f"  {name:5} n={d['total']:>4}  {parts}")

    c = out["concentration"]
    print(f"\n=== concentration ===")
    print(f"  {out['n_false_positives_total']} false positives over {out['n_cases']} cases; "
          f"{c['cases_with_no_false_positive']} cases have none")
    for k, v in c["share_held_by_worst"].items():
        print(f"    worst {k} of cases hold {v:.0%}")
    print(f"  {out['n_false_positives_renamed']} of {out['n_false_positives_total']} "
          f"false positives sit mostly on real tumour - they are mislabelled, not invented")

    if "attribution" in out:
        a = out["attribution"]
        keys = ("none", "relabel", "delete", "both")
        regions = [k for k in a["none"] if k not in ("n_cases", "min_voxels", "mean_lesion")]
        for metric, label in (("lesion", "lesion-wise"), ("volumetric", "volumetric")):
            print(f"\n=== {label} Dice under each oracle ===")
            print(f"  {'region':6}" + "".join(f"{k:>9}" for k in keys))
            for r in regions:
                print(f"  {r:6}" + "".join(f"{a[k][r][metric]:>9.4f}" for k in keys))
            if metric == "lesion":
                print(f"  {'mean':6}" + "".join(f"{a[k]['mean_lesion']:>9.4f}" for k in keys))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--no-oracles", action="store_true",
                    help="tabulate the failures but skip the (slower) re-scoring")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = analyse(args.run, args.dataset, args.split, not args.no_oracles, args.limit)
    _print(out)
    path = paths.REPORTS / f"failure_anatomy_{args.run}_{args.split}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
