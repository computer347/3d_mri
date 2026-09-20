"""Patient-level train/val/test splits.

The official BraTS validation set has no public labels, so it cannot be scored
locally. This module therefore carves a local validation *and* a held-out test
split out of the labelled training data:

* **train** - what the model fits on.
* **val** - watched every epoch; drives checkpoint selection and early stopping.
* **test** - scored once, at the end. Kept separate from ``val`` because
  choosing a checkpoint on ``val`` leaks into ``val``'s score.

Grouping rule: cases are grouped by patient, and any cross-patient pair that
``mri3d.duplicates`` flagged as near-identical is merged into the same group
(union-find). Whole groups are then assigned to a split, so no brain can appear
on two sides. Group sizes vary (one patient may have 10 timepoints), so groups
are placed largest-first into whichever split is furthest below its target
share - this keeps the *case* proportions close to the requested ratio even
though the *groups* are what gets assigned.

Run: ``python -m mri3d.splits --seed 42``
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from . import paths
from .index import load_cases


class _Union:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _duplicate_links() -> list[tuple[str, str]]:
    """Cross-patient near-duplicate pairs, if the audit has been run."""
    if not paths.DUPLICATES.exists():
        return []
    with open(paths.DUPLICATES, newline="", encoding="utf-8") as f:
        return [(r["patient_a"], r["patient_b"]) for r in csv.DictReader(f)
                if r["same_patient"].lower() == "false"]


def build_groups(cases) -> dict[str, list]:
    uf = _Union()
    for c in cases:
        uf.find(c.patient)
    links = _duplicate_links()
    for a, b in links:
        uf.union(a, b)
    if links:
        print(f"merged {len(links)} cross-patient duplicate link(s) into shared groups")
    groups: dict[str, list] = defaultdict(list)
    for c in cases:
        groups[uf.find(c.patient)].append(c)
    return dict(groups)


def assign(groups: dict[str, list], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    # Largest groups first: placing a 10-timepoint patient last would overshoot
    # whichever split it landed in.
    keys.sort(key=lambda k: -len(groups[k]))

    total = sum(len(v) for v in groups.values())
    target = {k: v * total for k, v in ratios.items()}
    counts = {k: 0 for k in ratios}
    out: dict[str, list[str]] = {k: [] for k in ratios}
    for k in keys:
        # Deficit relative to target, normalised so small splits stay competitive.
        split = max(counts, key=lambda s: (target[s] - counts[s]) / max(target[s], 1))
        out[split].extend(c.case_id for c in groups[k])
        counts[split] += len(groups[k])
    return {k: sorted(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test")
    args = ap.parse_args()

    tr, va, te = (float(x) for x in args.ratios.split(","))
    ratios = {"train": tr, "val": va, "test": te}
    paths.ensure_dirs()

    manifest: dict = {"seed": args.seed, "ratios": ratios, "datasets": {}}
    for name, which in (("gli", "gli_train"), ("men_rt", "men_train")):
        cases = [c for c in load_cases(which) if c.has_label]
        groups = build_groups(cases)
        splits = assign(groups, ratios, args.seed)
        by_case = {c.case_id: c for c in cases}
        manifest["datasets"][name] = splits

        print(f"\n{name}: {len(cases)} labelled cases in {len(groups)} patient groups")
        for s, ids in splits.items():
            pats = {by_case[i].patient for i in ids}
            print(f"  {s:<5} {len(ids):>5} cases ({len(ids)/len(cases):5.1%})  "
                  f"{len(pats):>4} patients")
        overlap = set(splits["train"]) & (set(splits["val"]) | set(splits["test"]))
        pat = {s: {by_case[i].patient for i in ids} for s, ids in splits.items()}
        cross = (pat["train"] & pat["val"]) | (pat["train"] & pat["test"]) | (pat["val"] & pat["test"])
        assert not overlap and not cross, "split leak detected"
        print("  checked: no case or patient appears in more than one split")

    with open(paths.SPLITS, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {paths.SPLITS}")


def load_split(dataset: str, split: str) -> list[str]:
    with open(paths.SPLITS, encoding="utf-8") as f:
        return json.load(f)["datasets"][dataset][split]


if __name__ == "__main__":
    main()
