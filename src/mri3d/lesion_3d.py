"""Turn the masks back into geometry: a rotatable 3D view of what the model found.

Every other view in this project is a stack of 2D slices, which is how the data
is stored but not how a tumour is shaped. A false-positive component that looks
like a small blob on three consecutive axial slices is much easier to judge when
you can see it sitting 40 mm from the main mass on the other side of the
ventricles. That judgement is exactly what ``mri3d.select`` automates, so this
page colours each predicted component by the **selector's verdict**:

  kept, real          the selector was right to keep it
  kept, spurious      a false positive that survived
  dropped, spurious   the win - an invented lesion, deleted
  dropped, real       the cost - a genuine lesion the selector deleted

Those four layers are the trade-off in `reports/results.md` made visible per
case, and the last one is the layer worth staring at: on the glioma test split
the selector deletes 93 genuine detections, and five of them are in cases whose
score went down as a result.

Surfaces come from marching cubes. Positions are quantised to 1/16 of a voxel
(the volume is 182 voxels across, so uint16 is ample and halves the payload),
buffers are deflate-compressed, and the page inflates them with the browser's
own ``DecompressionStream``. The result is one self-contained HTML file per run
that opens from disk with no web server - the same constraint
``heatmap_page.py`` works under, for the same reason.

    python -m mri3d.lesion_3d --run gli_main --split test
    python -m mri3d.lesion_3d --run gli_main --split test --cases BraTS-GLI-00005-100
    python -m mri3d.lesion_3d --run men_long --split test --dataset men_rt --pick first
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure

from . import paths
from .data import case_dicts
from .metrics import match_components

# Sub-voxel quantisation for vertex positions. 1/16 voxel is 0.06 mm on the
# glioma grid - far below anything the segmentation resolves - and lets a
# position ride in a uint16 instead of a float32.
QUANT = 16

# The brain shell is context, not measurement, so it is meshed at half
# resolution. The smoothing is heavier than it looks like it should be: at
# sigma 0.7 the iso-surface picks up intensity noise and the cortex comes out
# crumbly rather than folded, and it costs 2.5x the triangles to do it.
BRAIN_STEP, BRAIN_SIGMA = 3, 1.8
# Iso-level for the brain surface, as a fraction of the median non-zero T1c
# intensity. A fraction of the *maximum* would be at the mercy of one bright
# voxel; the median of the tissue is stable across cases with very different
# scanners, which these have.
#
# The value has to cut *into* the tissue range, not sit below it. BraTS volumes
# are skull-stripped, so a low level just re-traces the strip mask and yields a
# smooth egg. Sulci are CSF-filled and dark on T1, so a level near the grey
# matter intensity is what carves them out and makes the surface read as a
# brain rather than a hull.
BRAIN_LEVEL = 0.90

# Lesion resolution by size, because the two ends of the range are being looked
# at for different reasons. A 200-voxel satellite is the thing under judgement
# and is meshed exactly; the dominant mass is context, its surface is enormously
# convoluted, and at full resolution one oedema region cost 111k triangles and
# 800 kB - more than the rest of the page together. Marching cubes at step s
# costs roughly 1/s^2 of the triangles, so the big structures get most of the
# saving and lose only detail nobody is inspecting.
#   (max voxels, step, smoothing sigma)
LESION_LOD = [(2_000, 1, 0.6), (20_000, 2, 1.0), (np.inf, 3, 1.4)]
# Components below this are not meshed at all; they are smaller than the page
# can usefully draw and there can be hundreds of them.
MIN_MESH_VOXELS = 20

# "main" is not a verdict, it is the largest matched component of each class -
# the dominant mass. It gets its own translucent layer because it is context,
# not a decision: it is kept at any threshold anybody would choose, it is
# usually 50x the volume of everything else, and left opaque it simply hides
# the components the page exists to show. This is the same "not the largest in
# its class" split that mri3d.select uses to estimate d, for the same reason.
VERDICTS = {
    "main": ("main mass (largest, matched)", "#14b8a6"),
    "kept_real": ("kept, real", "#22c55e"),
    "kept_spurious": ("kept, spurious", "#ef4444"),
    "dropped_spurious": ("dropped, spurious", "#64748b"),
    "dropped_real": ("dropped, real", "#f59e0b"),
    "below_floor": ("under 50 voxels (unscored)", "#a78bfa"),
}
# Shared with heatmap_page.py so a class is the same colour everywhere.
CLASS_COLOURS = {"NETC": "#3b82f6", "SNFH": "#eab308", "ET": "#ef4444",
                 "RC": "#a855f7", "GTV": "#eab308"}


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(arr.tobytes(), 6)).decode("ascii")


def _marching(vol: np.ndarray, sigma: float, step: int = 1, level: float = 0.5):
    """Marching cubes, padded so the surface closes.

    Without the pad, a component touching the edge of its own bounding box
    produces an open shell that renders as a hole when seen from behind.

    ``level`` defaults to 0.5, for the binary masks. The brain shell passes the
    T1c intensities and a tissue level instead: iso-surfacing the intensity
    keeps the sulci, where binarising first and smoothing the result gives a
    smooth bag roughly the shape of a head and not recognisable as a brain.
    """
    v = np.pad(vol.astype(np.float32), 1)
    if sigma:
        smoothed = ndimage.gaussian_filter(v, sigma)
        # Smoothing a very small component can push its peak below the level,
        # which marching_cubes rejects. Fall back to the unsmoothed volume.
        v = smoothed if smoothed.max() > level else v
    if v.max() <= level or v.min() >= level:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(v, level=level, step_size=step)
    except (ValueError, RuntimeError):
        return None
    if len(faces) == 0:
        return None
    return verts - 1.0, faces  # undo the pad


def _merge(parts: list[tuple[np.ndarray, np.ndarray]]):
    if not parts:
        return None
    verts, faces, offset = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def _pack(mesh, shape) -> dict | None:
    """Quantised positions + indices, deflate-compressed, base64 for the page."""
    if mesh is None:
        return None
    verts, faces = mesh
    # Marching cubes indexes (i, j, k); the page wants (x, y, z) in millimetres
    # relative to the volume centre, so the camera orbits the head rather than
    # a corner of the array.
    centred = verts - (np.asarray(shape) - 1) / 2.0
    q = np.clip(np.round((centred + 128.0) * QUANT), 0, 65535).astype("<u2")
    return {"pos": _b64(q), "idx": _b64(faces.astype("<u4")),
            "n": int(len(faces)) * 3, "verts": int(len(verts))}


def _component_meshes(labelled: np.ndarray, ids: list[int]) -> list:
    """One mesh per component, each built inside its own padded bounding box.

    Meshing the whole volume once per component would be ~200x the work for the
    same triangles; ``find_objects`` gives the box, and the vertices are shifted
    back into volume coordinates afterwards. Resolution follows ``LESION_LOD``.
    """
    if not ids:
        return []
    objs = ndimage.find_objects(labelled)
    out = []
    for j in ids:
        sl = objs[j - 1]
        if sl is None:
            continue
        sub = labelled[sl] == j
        n = int(sub.sum())
        step, sigma = next((s, g) for cap, s, g in LESION_LOD if n <= cap)
        m = _marching(sub, sigma, step) or _marching(sub, 0.6, 1)
        if m is None:
            continue
        verts, faces = m
        out.append((verts + [s.start for s in sl], faces))
    return out


def _label_meshes(mask: np.ndarray) -> list:
    """Every connected component of a binary mask, meshed."""
    lab, n = ndimage.label(mask)
    return _component_meshes(lab, list(range(1, n + 1)))


def build_case(rec: dict, pred_dir: Path, class_values: dict[int, str],
               keep: dict | None, sizes: dict | None,
               brain_level: float = BRAIN_LEVEL,
               brain_sigma: float = BRAIN_SIGMA) -> dict:
    """Every mesh for one case, plus the counts that go in the case list."""
    pred = np.asanyarray(nib.as_closest_canonical(
        nib.load((pred_dir / f"{rec['case_id']}_pred.nii.gz").as_posix())).dataobj).astype(np.uint8)
    ref = np.asanyarray(nib.as_closest_canonical(
        nib.load(rec["label"])).dataobj).astype(np.uint8)
    t1c = np.asanyarray(nib.as_closest_canonical(
        nib.load(rec["image"][0])).dataobj).astype(np.float32)

    shape = pred.shape
    layers: dict[str, dict] = {}

    # Brain shell, iso-surfaced from the T1c intensities themselves.
    nz = t1c[t1c > 0]
    level = float(np.median(nz)) * brain_level if nz.size else 0.5
    brain = _marching(t1c, brain_sigma, BRAIN_STEP, level)
    layers["brain"] = _pack(brain, shape)

    counts: dict[str, int] = {k: 0 for k in VERDICTS}
    for value, name in class_values.items():
        if value == 0:
            continue
        p, r = pred == value, ref == value
        if not p.any() and not r.any():
            continue

        if r.any():
            packed = _pack(_merge(_label_meshes(r)), shape)
            if packed:
                layers[f"ref:{name}"] = packed

        if not p.any():
            continue
        m = match_components(p, r)
        biggest = max(m.comps, key=lambda c: c.n_voxels).comp_id if m.comps else None
        buckets: dict[str, list[int]] = {k: [] for k in VERDICTS}
        for c in m.comps:
            if c.n_voxels < MIN_MESH_VOXELS:
                continue
            k = (rec["case_id"], name, c.comp_id)
            if sizes is not None and k in sizes and sizes[k] != c.n_voxels:
                raise RuntimeError(
                    f"component mismatch for {k}: csv says {sizes[k]} voxels, the mask on "
                    f"disk has {c.n_voxels}")
            kept = True if keep is None else keep.get(k, True)
            if c.n_voxels < 50:
                verdict = "below_floor"
            elif c.comp_id == biggest and c.matched and kept:
                # Only when it is both matched and kept. A dominant mass that is
                # spurious, or that the selector would delete, is a genuine
                # finding and must not be quietly filed away as context.
                verdict = "main"
            else:
                verdict = f"{'kept' if kept else 'dropped'}_{'real' if c.matched else 'spurious'}"
            buckets[verdict].append(c.comp_id)
            counts[verdict] += 1
        for verdict, ids in buckets.items():
            packed = _pack(_merge(_component_meshes(m.pred_lab, ids)), shape)
            if packed:
                layers[f"{verdict}:{name}"] = packed

    return {"case_id": rec["case_id"], "shape": list(shape),
            "counts": counts, "layers": {k: v for k, v in layers.items() if v}}


def build_page(cases: list[dict], meta: dict, out: Path) -> Path:
    template = (Path(__file__).parent / "lesion3d_template.html").read_text(encoding="utf-8")
    payload = json.dumps({"cases": cases, "meta": meta,
                          "verdicts": {k: {"label": v[0], "colour": v[1]}
                                       for k, v in VERDICTS.items()},
                          "classColours": CLASS_COLOURS, "quant": QUANT})
    out.write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    return out


def pick_cases(run: str, split: str, recs: list[dict], n: int, strategy: str) -> list[dict]:
    """Which cases to put on the page.

    ``cost`` is the default because it is the view that is hard to get any other
    way: the cases where the selector deleted a genuine detection. A page of
    typical cases shows that the thing works, which the score table already
    says; these are the ones where it is worth asking whether it should have.
    """
    from . import select

    by_id = {r["case_id"]: r for r in recs}
    path = select.components_path(run, split)
    if strategy == "first" or not path.exists():
        return recs[:n]

    comps = select.load_components(run, split)
    model = select.load_selector(run)
    probs = select.posterior(model["model"], comps)
    p_star = model["p_star"]

    score: dict[str, tuple] = {}
    for i, c in enumerate(comps):
        if c["n_voxels"] < select.DECIDABLE_MIN_VOXELS:
            continue
        dropped = probs[i] < p_star
        cid = c["case_id"]
        lost, won = score.get(cid, (0, 0))
        if dropped and c["matched"]:
            lost += 1
        if dropped and not c["matched"]:
            won += 1
        score[cid] = (lost, won)

    key = {"cost": lambda kv: (-kv[1][0], -kv[1][1]),
           "win": lambda kv: (-kv[1][1], kv[1][0])}[strategy]
    order = [c for c, _ in sorted(score.items(), key=key) if c in by_id]
    return [by_id[c] for c in order[:n]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=6, help="how many cases to put on the page")
    ap.add_argument("--cases", nargs="+", default=None, help="explicit case ids")
    ap.add_argument("--pick", default="cost", choices=["cost", "win", "first"],
                    help="cost: cases where the selector deleted a real lesion")
    ap.add_argument("--no-selector", action="store_true",
                    help="colour by matched/spurious only, ignoring any fitted selector")
    ap.add_argument("--brain-sigma", type=float, default=BRAIN_SIGMA,
                    help="smoothing for the brain shell")
    ap.add_argument("--brain-level", type=float, default=BRAIN_LEVEL,
                    help="iso-level for the brain shell, as a fraction of the median "
                         "non-zero T1c intensity (higher carves the sulci deeper)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    class_values = paths.GLI_LABELS if args.dataset == "gli" else paths.MEN_LABELS
    pred_dir = paths.OUTPUTS / "runs" / args.run / f"pred_{args.split}"
    recs = [r for r in case_dicts(args.dataset, args.split)
            if (pred_dir / f"{r['case_id']}_pred.nii.gz").exists()]
    if not recs:
        raise SystemExit(f"no predicted masks in {pred_dir}; run mri3d.predict --save-masks")

    keep = sizes = None
    p_star = None
    if not args.no_selector and (paths.REPORTS / f"selector_{args.run}.json").exists():
        from .postprocess import selector_filter
        try:
            sel = selector_filter(args.run, args.split)
            keep, sizes, p_star = sel["keep"], sel["sizes"], sel["p_star"]
        except FileNotFoundError:
            print("note: no components CSV for this split; colouring by match only")

    if args.cases:
        wanted = set(args.cases)
        chosen = [r for r in recs if r["case_id"] in wanted]
        missing = wanted - {r["case_id"] for r in chosen}
        if missing:
            raise SystemExit(f"not found in {args.split}: {', '.join(sorted(missing))}")
    else:
        chosen = (pick_cases(args.run, args.split, recs, args.n, args.pick)
                  if keep is not None else recs[:args.n])

    out = Path(args.out) if args.out else paths.REPORTS / f"lesion_3d_{args.run}_{args.split}.html"
    built, t0 = [], time.time()
    for i, rec in enumerate(chosen, 1):
        built.append(build_case(rec, pred_dir, class_values, keep, sizes,
                                args.brain_level, args.brain_sigma))
        c = built[-1]["counts"]
        print(f"  {i}/{len(chosen)}  {rec['case_id']}  "
              f"kept {c['kept_real']}R/{c['kept_spurious']}S  "
              f"dropped {c['dropped_spurious']}S/{c['dropped_real']}R", flush=True)

    meta = {"run": args.run, "split": args.split, "dataset": args.dataset,
            "p_star": p_star, "classes": [n for v, n in class_values.items() if v],
            "selector": keep is not None}
    build_page(built, meta, out)
    print(f"\n{len(built)} cases in {time.time() - t0:.0f}s -> {out} "
          f"({out.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
