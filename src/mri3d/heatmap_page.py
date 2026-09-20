"""Turn the accumulated heatmaps into assets the viewer page can draw.

Every slice of every volume is packed into a mosaic PNG (one per volume, per
plane). A browser decodes those natively, so the page loads a few hundred KB of
image instead of megabytes of JSON, and slice scrubbing is a canvas blit rather
than a re-render.

Outputs into ``reports/heatmap_assets/``:
  ``template_{plane}.png``   mean brain, greyscale
  ``{CLASS}_{plane}.png``    per-class case-frequency, greyscale (scaled to max)
  ``meta.json``              geometry, per-class peak counts and prevalence

Run: ``python -m mri3d.heatmap_page``
"""

from __future__ import annotations

import base64
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import paths

# (name, axis sliced along, how to orient the 2D slice for display)
PLANES = {"axial": 2, "coronal": 1, "sagittal": 0}
CLASS_COLOURS = {  # colour-blind safe, distinct in both themes
    "NETC": "#3b82f6",
    "SNFH": "#eab308",
    "ET": "#ef4444",
    "RC": "#a855f7",
}
CLASS_LABELS = {
    "NETC": "non-enhancing tumour core",
    "SNFH": "oedema / FLAIR hyperintensity",
    "ET": "enhancing tumour",
    "RC": "resection cavity",
}


def _slice(vol: np.ndarray, plane: str, k: int) -> np.ndarray:
    axis = PLANES[plane]
    s = np.take(vol, k, axis=axis)
    # Radiological-ish display: rows run superior->inferior, so flip vertically.
    return np.rot90(s)


def _mosaic(vol: np.ndarray, plane: str, vmax: float) -> tuple[Image.Image, dict]:
    n = vol.shape[PLANES[plane]]
    tile = _slice(vol, plane, 0).shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    sheet = np.zeros((rows * tile[0], cols * tile[1]), dtype=np.uint8)
    for k in range(n):
        s = _slice(vol, plane, k).astype(np.float32)
        s = np.clip(s / vmax, 0, 1) * 255 if vmax > 0 else s * 0
        r, c = divmod(k, cols)
        sheet[r * tile[0] : (r + 1) * tile[0], c * tile[1] : (c + 1) * tile[1]] = s.astype(np.uint8)
    meta = {"n_slices": n, "tile_w": tile[1], "tile_h": tile[0], "cols": cols, "rows": rows}
    return Image.fromarray(sheet, mode="L"), meta


def build_page(assets_dir, meta: dict) -> None:
    """Inline the mosaics into a single self-contained HTML file.

    The PNGs go in as data: URIs rather than as sibling files for two reasons:
    the page then opens straight from disk with no web server, and canvas pixel
    reads (which the hover readout needs) are blocked on file:// images loaded
    by URL but allowed for data: URIs.
    """
    template = (Path(__file__).parent / "viewer_template.html").read_text(encoding="utf-8")
    assets = {}
    for png in sorted(assets_dir.glob("*.png")):
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        assets[png.stem] = f"data:image/png;base64,{b64}"

    html = (template
            .replace("__META_JSON__", json.dumps(meta))
            .replace("__ASSETS_JSON__", json.dumps(assets)))
    page = paths.REPORTS / "tumour_atlas.html"
    page.write_text(html, encoding="utf-8")
    print(f"wrote {page} ({page.stat().st_size/1e6:.2f} MB, self-contained)")


def main() -> None:
    out = paths.REPORTS / "heatmap_assets"
    out.mkdir(parents=True, exist_ok=True)

    heat_npz = np.load(paths.HEATMAPS, allow_pickle=False)
    heat = heat_npz["heat"].astype(np.float32)  # (4, 91, 109, 91)
    classes = [str(c) for c in heat_npz["classes"]]
    template = np.load(paths.OUTPUTS / "template.npz")["template"]

    # Prevalence per class, straight from the case index.
    with open(paths.CASE_INDEX, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["dataset"] == "gli" and r["has_label"] == "True"]
    n_cases = len(rows)
    present = {c: sum(1 for r in rows if int(r.get(f"vox_{c}") or 0) > 0) for c in classes}

    meta = {
        "n_cases": n_cases,
        "planes": {},
        "classes": [],
        "volume_shape": [int(s) for s in heat.shape[1:]],
        "voxel_mm": 2.0,  # heatmaps are accumulated at half of 1 mm resolution
    }

    tmax = float(np.percentile(template[template > 0], 99.5))
    for plane in PLANES:
        img, geom = _mosaic(template, plane, tmax)
        img.save(out / f"template_{plane}.png", optimize=True)
        meta["planes"][plane] = geom

    for i, cls in enumerate(classes):
        vmax = float(heat[i].max())
        for plane in PLANES:
            img, _ = _mosaic(heat[i], plane, vmax)
            img.save(out / f"{cls}_{plane}.png", optimize=True)
        meta["classes"].append({
            "name": cls,
            "label": CLASS_LABELS.get(cls, ""),
            "colour": CLASS_COLOURS.get(cls, "#888888"),
            "peak_cases": int(vmax),
            "peak_pct": round(100 * vmax / n_cases, 1),
            "present_cases": present[cls],
            "present_pct": round(100 * present[cls] / n_cases, 1),
        })

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    build_page(out, meta)

    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"wrote {len(list(out.iterdir()))} files to {out} ({total/1e6:.2f} MB)")
    for c in meta["classes"]:
        print(f"  {c['name']:<5} present in {c['present_cases']:>4}/{n_cases} cases "
              f"({c['present_pct']:>4}%), hottest voxel in {c['peak_cases']} cases "
              f"({c['peak_pct']}%)")


if __name__ == "__main__":
    main()
