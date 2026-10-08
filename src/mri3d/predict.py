"""Run a trained model over whole cases and score it.

Inference is sliding-window regardless of how training was done: the window
overlaps by 25% and blends with a Gaussian, so voxels near a window edge - where
the model has least context - are weighted down rather than producing visible
seams between windows.

    python -m mri3d.predict --run gli_main --split test --save-masks
    python -m mri3d.predict --run gli_main --split val --save-components --mc-dropout 8

``--save-components`` writes one row per predicted connected component, with
the softmax statistics that ``argmax`` would otherwise throw away. That line -
``pred = logits.argmax(dim=1)`` - is why the earlier size-threshold experiment
could only ever filter on geometry: by the time the mask reached
``mri3d.postprocess`` every probability had already been discarded. The
components CSV is the fix, and it is what ``mri3d.select`` learns from.

No probability volume is ever written to disk. 182x218x182 x 5 classes at fp16
is ~72 MB per case, ~15 GB for one split; the per-component statistics are
~1 kB per case and carry the part that a selector can use.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage

from . import paths
from .data import case_dicts, val_transforms
from .fallback import fill_if_empty
from .metrics import GLI_REGIONS, match_components, score_case
from .model import build_model
from .seeding import set_seed

# Column order for reports/components_{run}_{split}.csv. Declared once so
# mri3d.select can rely on it and so a missing feature fails loudly rather than
# silently becoming a column of empty strings.
COMPONENT_FIELDS = [
    "case_id", "run", "split", "dataset",
    "class_name", "class_value", "comp_id",
    "n_voxels", "volume_mm3",
    "centroid_i", "centroid_j", "centroid_k",
    "mean_prob", "p90_prob", "max_prob", "mean_border_prob",
    "mc_mean_var",
    "dist_to_dominant_mm",
    "rank_by_volume", "rank_by_prob", "n_comps_in_class",
    "matched", "matched_ref_ids", "dice_with_matched",
    # Mean probability of every class over the component's voxels (class value
    # 0-4; empty where the dataset has fewer classes), and the best class other
    # than the assigned one. These are what mri3d.relabel decides from.
    *[f"prob_c{v}" for v in range(5)],
    "runner_up_value", "runner_up_prob",
    # What the reference says is under the component: its majority tumour class,
    # 0 when less than half of it is tumour. A training target for
    # mri3d.relabel, never an input to any decision.
    "ref_majority_value", "ref_tumour_frac",
]

REF_LESION_FIELDS = [
    "case_id", "run", "split", "dataset",
    "class_name", "class_value", "ref_id",
    "n_voxels", "volume_mm3", "dist_to_dominant_mm",
    "detected", "dice", "matched_comp_ids",
]


def load_run(run: str, device):
    ckpt = torch.load(paths.OUTPUTS / "runs" / run / "best.pt", map_location=device,
                      weights_only=False)
    model = build_model(ckpt["n_classes"], ckpt["in_channels"],
                        ckpt["args"].get("width", 16)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# All 8 combinations of flipping the three spatial axes of a (B, C, X, Y, Z)
# tensor - the same three axes data.train_transforms flips with RandFlipd.
FLIP_DIMS = [tuple(d for d, bit in zip((2, 3, 4), bits) if bit)
             for bits in itertools.product((0, 1), repeat=3)]


def _infer(model, image, patch, tta: bool = False):
    """Sliding-window logits; with ``tta``, averaged over the 8 axis flips.

    The model trained with random flips on every axis, so each flipped copy is
    an input it has learned to handle, and the disagreement between them is
    noise that averaging removes. Probabilities are averaged, not logits -
    a single overconfident view should not be able to outvote the other seven -
    and the log of the mean is returned so callers can keep applying softmax:
    softmax(log p) == p when p already sums to one.
    """
    if not tta:
        return sliding_window_inference(image, patch, 1, model, overlap=0.25, mode="gaussian")
    mean = None
    for dims in FLIP_DIMS:
        x = torch.flip(image, dims) if dims else image
        logits = sliding_window_inference(x, patch, 1, model, overlap=0.25, mode="gaussian")
        p = torch.softmax(logits.float(), dim=1)
        p = torch.flip(p, dims) if dims else p
        mean = p if mean is None else mean + p
        del logits, p
    return torch.log((mean / len(FLIP_DIMS)).clamp_min(1e-12))


def _ensemble_infer(models, image, patch, tta: bool = False):
    """Like ``_infer``, but averaging probabilities over several models.

    Probabilities, not logits, for the same reason as the flip average; one
    model reduces to exactly ``_infer``, so a single --run is unchanged.
    """
    if len(models) == 1:
        return _infer(models[0], image, patch, tta)
    mean = None
    for m in models:
        p = torch.softmax(_infer(m, image, patch, tta).float(), dim=1)
        mean = p if mean is None else mean + p
        del p
    return torch.log((mean / len(models)).clamp_min(1e-12))


def mc_dropout_variance(model, image, patch, n_passes: int, device) -> np.ndarray:
    """Per-voxel, per-class variance of the softmax under T stochastic passes.

    The model already trains with ``dropout_prob=0.2``, so a predictive
    distribution is available at inference for the cost of T forward passes and
    no retraining at all: keep every other module in ``eval()`` - batch-norm
    statistics must stay frozen, or the "uncertainty" measured would mostly be
    batch-norm noise - and put only the dropout modules back into ``train()``.

    Variance accumulates with Welford's algorithm. The naive alternative, stack
    T volumes and call ``np.var``, would hold 8 x 5 x 182 x 218 x 182 float32 =
    1.2 GB; Welford holds two arrays regardless of T.
    """
    was_training = {m: m.training for m in model.modules()}
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d,
                          torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
    try:
        mean = m2 = None
        for t in range(n_passes):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=device.type == "cuda"):
                logits = _infer(model, image, patch)
            x = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            del logits
            if mean is None:
                mean = np.zeros_like(x)
                m2 = np.zeros_like(x)
            delta = x - mean
            mean += delta / (t + 1)
            m2 += delta * (x - mean)
        # Sample variance (n-1): with T as small as 8 the population estimate is
        # noticeably biased low, and the feature is compared across components.
        return m2 / max(1, n_passes - 1)
    finally:
        for m, was in was_training.items():
            m.train(was)


def _dominant_mass(pred: np.ndarray, class_values: dict[int, str]) -> np.ndarray:
    """Largest connected component of predicted whole tumour.

    Whole tumour is ``GLI_REGIONS["WT"]`` = NETC + SNFH + ET, which excludes the
    resection cavity. That is deliberate and matches ``metrics.GLI_REGIONS``: RC
    is a post-surgical void that can be large and is not scored as part of WT,
    so including it would move the reference point for
    ``dist_to_dominant_mm`` away from the tumour it is meant to describe.
    """
    values = (GLI_REGIONS["WT"] if set(class_values) >= {1, 2, 3}
              else tuple(v for v in class_values if v != 0))
    wt = np.isin(pred, values)
    if not wt.any():
        return np.zeros_like(wt)
    lab, n = ndimage.label(wt)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def _border_mean(prob: np.ndarray, labelled: np.ndarray, comp_id: int,
                 sl: tuple[slice, ...]) -> float:
    """Mean class probability on the 1-voxel shell just outside a component.

    A real lesion that the model is confident about tends to fall off sharply
    at its edge; a diffuse artefact bleeds outward, so its surroundings still
    carry appreciable probability. The shell is computed inside the component's
    bounding box padded by one voxel - dilating the full volume once per
    component would be ~200x more work for the same answer.
    """
    pad = tuple(slice(max(0, s.start - 1), min(d, s.stop + 1))
                for s, d in zip(sl, labelled.shape))
    sub = labelled[pad] == comp_id
    shell = ndimage.binary_dilation(sub) & ~sub
    if not shell.any():
        return float("nan")
    return float(prob[pad][shell].mean())


def component_rows(pred: np.ndarray, probs: np.ndarray, ref: np.ndarray,
                   spacing: np.ndarray, class_values: dict[int, str],
                   mc_var: np.ndarray | None, meta: dict) -> tuple[list[dict], list[dict]]:
    """One row per predicted component and one per reference lesion, for a case.

    Matching is delegated to ``metrics.match_components`` rather than
    reimplemented, so "matched" here means exactly what it means to the metric
    the selector is trying to move.
    """
    voxel_mm3 = float(np.prod(spacing))
    mass = _dominant_mass(pred, class_values)
    # Distance from every voxel to the dominant mass, in millimetres. EDT of the
    # complement, with physical sampling: GLI is 1 mm isotropic but MEN-RT
    # voxels run from 0.37 to 1.5 mm, so assuming unit spacing would make the
    # same anatomical distance read four times larger on one case than another.
    dist = (ndimage.distance_transform_edt(~mass, sampling=tuple(spacing))
            if mass.any() else None)

    comp_rows: list[dict] = []
    ref_rows: list[dict] = []
    for value, name in class_values.items():
        if value == 0:
            continue
        p = pred == value
        r = ref == value
        if not p.any() and not r.any():
            continue

        m = match_components(p, r)
        lab, n = m.pred_lab, len(m.comps)
        prob_v = probs[value]
        idx = list(range(1, n + 1))

        if n:
            objs = ndimage.find_objects(lab)
            means = ndimage.mean(prob_v, lab, idx)
            maxes = ndimage.maximum(prob_v, lab, idx)
            p90s = ndimage.labeled_comprehension(
                prob_v, lab, idx, lambda x: np.percentile(x, 90), float, 0.0)
            cents = ndimage.center_of_mass(p, lab, idx)
            mins = (ndimage.minimum(dist, lab, idx) if dist is not None
                    else [float("nan")] * n)
            mcs = (ndimage.mean(mc_var[value], lab, idx) if mc_var is not None
                   else [None] * n)
            means, maxes, p90s = np.atleast_1d(means), np.atleast_1d(maxes), np.atleast_1d(p90s)
            mins = np.atleast_1d(mins)
            mcs = mcs if mc_var is None else np.atleast_1d(mcs)
            if n == 1 and not isinstance(cents, list):
                cents = [cents]

            class_means = {v: np.atleast_1d(ndimage.mean(probs[v], lab, idx))
                           for v in class_values}

            sizes = np.array([c.n_voxels for c in m.comps], dtype=np.int64)
            # 1 = largest / most confident, so "rank 1" reads the same way in
            # both columns and a higher rank always means a weaker candidate.
            rank_vol = (-sizes).argsort().argsort() + 1
            rank_prob = (-np.asarray(means)).argsort().argsort() + 1

            for k, c in enumerate(m.comps):
                per_class = {v: float(class_means[v][k]) for v in class_values}
                others = {v: p_ for v, p_ in per_class.items() if v not in (0, value)}
                ru = max(others, key=others.get) if others else None
                under = ref[objs[k]][lab[objs[k]] == c.comp_id]
                counts = np.bincount(under, minlength=max(class_values) + 1)
                tumour = int(counts[1:].sum())
                frac = tumour / max(1, c.n_voxels)
                majority = int(np.argmax(counts[1:]) + 1) if frac >= 0.5 else 0
                comp_rows.append({
                    **meta,
                    "class_name": name, "class_value": value, "comp_id": c.comp_id,
                    "n_voxels": c.n_voxels,
                    "volume_mm3": round(c.n_voxels * voxel_mm3, 3),
                    "centroid_i": round(float(cents[k][0]), 2),
                    "centroid_j": round(float(cents[k][1]), 2),
                    "centroid_k": round(float(cents[k][2]), 2),
                    "mean_prob": round(float(means[k]), 6),
                    "p90_prob": round(float(p90s[k]), 6),
                    "max_prob": round(float(maxes[k]), 6),
                    "mean_border_prob": round(_border_mean(prob_v, lab, c.comp_id, objs[k]), 6),
                    "mc_mean_var": (None if mc_var is None else round(float(mcs[k]), 9)),
                    "dist_to_dominant_mm": round(float(mins[k]), 3),
                    "rank_by_volume": int(rank_vol[k]),
                    "rank_by_prob": int(rank_prob[k]),
                    "n_comps_in_class": n,
                    "matched": c.matched,
                    "matched_ref_ids": " ".join(str(i) for i in c.matched_ref_ids),
                    "dice_with_matched": round(c.dice_with_matched, 6),
                    **{f"prob_c{v}": round(per_class[v], 6) for v in per_class},
                    "runner_up_value": ru,
                    "runner_up_prob": None if ru is None else round(others[ru], 6),
                    "ref_majority_value": majority,
                    "ref_tumour_frac": round(frac, 4),
                })

        scored = m.scored_refs
        if scored:
            ridx = [lesion.ref_id for lesion in scored]
            rmins = (ndimage.minimum(dist, m.ref_lab, ridx) if dist is not None
                     else [float("nan")] * len(ridx))
            rmins = np.atleast_1d(rmins)
            for k, lesion in enumerate(scored):
                ref_rows.append({
                    **meta,
                    "class_name": name, "class_value": value, "ref_id": lesion.ref_id,
                    "n_voxels": lesion.n_voxels,
                    "volume_mm3": round(lesion.n_voxels * voxel_mm3, 3),
                    "dist_to_dominant_mm": round(float(rmins[k]), 3),
                    "detected": lesion.detected,
                    "dice": round(lesion.dice, 6),
                    "matched_comp_ids": " ".join(str(i) for i in lesion.matched_comp_ids),
                })
    return comp_rows, ref_rows


def _write_csv(path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory under outputs/runs/")
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="seed for torch/numpy/MONAI")
    ap.add_argument("--save-masks", action="store_true",
                    help="write predicted masks as .nii.gz (needed by the case viewer)")
    ap.add_argument("--save-components", action="store_true",
                    help="write per-component softmax/geometry features to reports/")
    ap.add_argument("--mc-dropout", type=int, default=0, metavar="T",
                    help="T stochastic forward passes for a per-component variance "
                         "feature (0 = off). Costs T x inference, no retraining.")
    ap.add_argument("--suffix", default="", help="appended to the components CSV names")
    ap.add_argument("--tta", action="store_true",
                    help="average probabilities over all 8 axis flips (8 x inference). "
                         "Use with a separate --run name so the plain results are kept.")
    ap.add_argument("--members", nargs="+", default=None, metavar="RUN",
                    help="ensemble: average the probabilities of these runs' checkpoints. "
                         "Outputs are written under --run, which then needs no best.pt. "
                         "MC dropout (if on) uses the first member only.")
    ap.add_argument("--fill-empty", action="store_true",
                    help="if a case's mask is empty, keep the half-max region around the "
                         "most confident voxel (mri3d.fallback). For datasets where every "
                         "case has a target, such as MEN-RT.")
    ap.add_argument("--relabel", default=None, metavar="JSON",
                    help="rename components with a fitted mri3d.relabel model before "
                         "scoring and before the components CSV is written")
    ap.add_argument("--relabel-oof", action="store_true",
                    help="use the fold model that never saw each case's patient "
                         "(for scoring the split the relabeller was fitted on)")
    args = ap.parse_args()
    relabel_payload = (json.loads(open(args.relabel, encoding="utf-8").read())
                       if args.relabel else None)

    # benchmark=False: cuDNN's autotuner picks its algorithm against the free
    # workspace, so with it on the same seeded inference gives different numbers
    # depending on what else is using the GPU. See mri3d.seeding for the
    # measurement. Inference here produces the numbers that get reported, so it
    # buys reproducibility with throughput rather than the other way round.
    set_seed(args.seed, benchmark=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [load_run(r, device) for r in (args.members or [args.run])]
    models = [m for m, _ in loaded]
    model, ckpt = loaded[0]
    if len({c["n_classes"] for _, c in loaded}) != 1:
        raise SystemExit("ensemble members must predict the same classes")
    patch = tuple(ckpt["args"].get("patch", (128, 128, 128)))
    class_values = (paths.GLI_LABELS if args.dataset == "gli" else paths.MEN_LABELS)

    files = case_dicts(args.dataset, args.split)
    if args.limit:
        files = files[: args.limit]
    tf = val_transforms(ckpt["n_classes"])

    pred_dir = paths.OUTPUTS / "runs" / args.run / f"pred_{args.split}"
    if args.save_masks:
        pred_dir.mkdir(parents=True, exist_ok=True)

    rows, comp_rows, ref_rows, t0 = [], [], [], time.time()
    for i, rec in enumerate(files, 1):
        data = tf(rec)
        image = data["image"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            logits = _ensemble_infer(models, image, patch, args.tta)
        # Softmax then argmax, rather than argmax on the logits. Softmax is
        # monotonic per voxel so the saved mask is bit-identical to what this
        # script produced before; the difference is that the probabilities now
        # survive long enough to be summarised.
        probs = torch.softmax(logits.float(), dim=1)
        pred = probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        need_probs = args.save_components or args.fill_empty or relabel_payload is not None
        probs_np = probs[0].cpu().numpy() if need_probs else None
        filled = False
        if args.fill_empty:
            pred, filled = fill_if_empty(pred, probs_np)
        n_renamed = 0
        if relabel_payload is not None:
            from .relabel import apply as relabel_apply
            aff = np.asarray(data["label"].affine, dtype=np.float64)
            pred, n_renamed = relabel_apply(pred, probs_np, np.sqrt((aff[:3, :3] ** 2).sum(axis=0)),
                                            class_values, relabel_payload,
                                            rec["case_id"], args.relabel_oof)
        del logits, probs
        ref = data["label"][0].cpu().numpy().astype(np.uint8)

        row = {"case_id": rec["case_id"]}
        row.update(score_case(pred, ref, class_values))
        rows.append(row)

        if args.save_components:
            mc_var = (mc_dropout_variance(model, image, patch, args.mc_dropout, device)
                      if args.mc_dropout else None)
            affine = np.asarray(data["label"].affine, dtype=np.float64)
            spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
            meta = {"case_id": rec["case_id"], "run": args.run,
                    "split": args.split, "dataset": args.dataset}
            cr, rr = component_rows(pred, probs_np, ref, spacing, class_values, mc_var, meta)
            comp_rows += cr
            ref_rows += rr
            del probs_np, mc_var

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
        print(f"  {done:>9}  {rec['case_id']}  lesion-Dice {mean_lw:.3f}"
              f"{'  (filled empty mask)' if filled else ''}"
              f"{f'  ({n_renamed} renamed)' if n_renamed else ''}", flush=True)

    tag = f"{args.run}_{args.split}{args.suffix}"
    if args.save_components:
        _write_csv(paths.REPORTS / f"components_{tag}.csv", COMPONENT_FIELDS, comp_rows)
        _write_csv(paths.REPORTS / f"ref_lesions_{tag}.csv", REF_LESION_FIELDS, ref_rows)
        print(f"\n{len(comp_rows)} components, {len(ref_rows)} scored reference lesions "
              f"-> reports/components_{tag}.csv")

    out_csv = paths.REPORTS / f"scores_{args.run}_{args.split}.csv"
    # A --limit run is a smoke test. It must not overwrite the committed scores
    # for the split with a three-case subset, which is exactly what it used to
    # do and which is very hard to notice afterwards.
    if not args.limit:
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
    if not args.limit:
        (paths.REPORTS / f"summary_{args.run}_{args.split}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} cases in {time.time() - t0:.0f}s -> {out_csv}")
    for k, v in summary.items():
        if k.startswith("lesion_dice_"):
            print(f"  {k:<22} {v['mean_where_present']}  (n={v['n_cases']} with the class present)")


if __name__ == "__main__":
    main()
