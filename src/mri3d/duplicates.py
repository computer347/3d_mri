"""Duplicate and near-duplicate audit.

Why this exists: if the same scan (or the same brain) appears in both training
and validation, the reported score measures memorisation, not generalisation.
Two things can cause it here.

* **Exact duplicates** - the same file content under two case IDs. Caught by
  comparing MD5s.
* **Near-duplicates** - a different acquisition of the same brain, or a
  re-release of a case under a new ID. Caught by correlating 16x16x16
  normalised thumbnails of the T1c (see ``mri3d.scan``). Cosine similarity on
  those thumbnails is ~1.0 for the same head and falls away quickly for
  different ones.

Same-patient pairs are *expected* to be near-identical (BraTS-GLI follows
patients over time), so they are reported separately: they are not an error,
they are precisely why splits must be made per patient. Cross-patient pairs
above the threshold are the ones that need attention, and ``mri3d.splits``
reads this file so any such pair is forced onto the same side of the split.

Run: ``python -m mri3d.duplicates --threshold 0.99``
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from . import paths


def load_fingerprints() -> tuple[np.ndarray, np.ndarray, dict[str, dict]]:
    data = np.load(paths.EMBEDDINGS, allow_pickle=False)
    ids, thumbs = data["ids"], data["thumbs"].astype(np.float32)
    with open(paths.CASE_INDEX, newline="", encoding="utf-8") as f:
        meta = {r["case_id"]: r for r in csv.DictReader(f)}
    return ids, thumbs, meta


def cosine_matrix(thumbs: np.ndarray) -> np.ndarray:
    x = thumbs - thumbs.mean(axis=1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x @ x.T


def find_pairs(ids: np.ndarray, sim: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    iu = np.triu_indices(len(ids), k=1)
    hit = sim[iu] >= threshold
    return [(int(i), int(j), float(s))
            for i, j, s in zip(iu[0][hit], iu[1][hit], sim[iu][hit])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.99,
                    help="cosine similarity above which two cases are 'near-duplicate'")
    args = ap.parse_args()

    ids, thumbs, meta = load_fingerprints()
    print(f"{len(ids)} cases fingerprinted")

    # 1. Exact duplicates, by decoded voxels rather than by file bytes: the
    #    three real duplicates in BraTS-MEN-RT have identical imagery but
    #    different gzip streams, so a file hash reports nothing.
    key = "t1c_vox_md5" if "t1c_vox_md5" in next(iter(meta.values())) else "t1c_md5"
    by_hash: dict[str, list[str]] = {}
    for cid, row in meta.items():
        by_hash.setdefault(row[key], []).append(cid)
    exact = {k: v for k, v in by_hash.items() if len(v) > 1}
    by_file: dict[str, int] = {}
    for row in meta.values():
        by_file[row["t1c_md5"]] = by_file.get(row["t1c_md5"], 0) + 1
    file_dups = sum(1 for n in by_file.values() if n > 1)
    print(f"\nexact duplicates by voxel content: {len(exact)} group(s) "
          f"(by file hash alone: {file_dups})")
    for h, group in list(exact.items())[:10]:
        pats = {meta[c]["patient"] for c in group}
        tag = "SAME patient" if len(pats) == 1 else "DIFFERENT patients"
        print(f"  {h[:12]}  {sorted(group)}  [{tag}]")

    # 2. Near-duplicates.
    sim = cosine_matrix(thumbs)
    pairs = find_pairs(ids, sim, args.threshold)
    off = sim[np.triu_indices(len(ids), k=1)]
    print(f"\nsimilarity over all {off.size:,} pairs: "
          f"median {np.median(off):.3f}, p99 {np.percentile(off, 99):.3f}, max {off.max():.3f}")

    rows = []
    for i, j, s in sorted(pairs, key=lambda p: -p[2]):
        a, b = str(ids[i]), str(ids[j])
        pa, pb = meta[a]["patient"], meta[b]["patient"]
        rows.append({
            "case_a": a, "case_b": b, "patient_a": pa, "patient_b": pb,
            "similarity": round(s, 5),
            "same_patient": pa == pb,
            "same_file": meta[a]["t1c_md5"] == meta[b]["t1c_md5"],
            "subset_a": meta[a]["subset"], "subset_b": meta[b]["subset"],
            "dataset_a": meta[a]["dataset"], "dataset_b": meta[b]["dataset"],
        })

    cross = [r for r in rows if not r["same_patient"]]
    within = [r for r in rows if r["same_patient"]]
    print(f"\npairs >= {args.threshold}: {len(rows)}  "
          f"({len(within)} same-patient, {len(cross)} CROSS-PATIENT)")
    for r in cross[:20]:
        flag = "SAME FILE" if r["same_file"] else "near-dup"
        print(f"  {flag}  {r['similarity']:.4f}  {r['case_a']} ({r['subset_a']}) <-> "
              f"{r['case_b']} ({r['subset_b']})")
    if not cross:
        print("  none - no cross-patient leakage detected at this threshold")

    # Always show the closest cross-patient pairs, threshold or not: a pair just
    # under the cut is exactly what a fixed threshold would hide.
    iu = np.triu_indices(len(ids), k=1)
    order = np.argsort(-sim[iu])[:2000]
    shown = 0
    print("\nclosest cross-patient pairs overall:")
    for idx in order:
        i, j = int(iu[0][idx]), int(iu[1][idx])
        a, b = str(ids[i]), str(ids[j])
        if meta[a]["patient"] == meta[b]["patient"]:
            continue
        print(f"  {sim[i, j]:.4f}  {a} <-> {b}")
        shown += 1
        if shown == 8:
            break

    paths.ensure_dirs()
    with open(paths.DUPLICATES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                           ["case_a", "case_b", "patient_a", "patient_b", "similarity",
                            "same_patient", "same_file", "subset_a", "subset_b",
                            "dataset_a", "dataset_b"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {paths.DUPLICATES}")


if __name__ == "__main__":
    main()
