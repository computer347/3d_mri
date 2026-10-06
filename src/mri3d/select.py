"""A learned lesion selector: keep or drop each predicted component.

The model segments the main tumour well and then invents roughly one extra
lesion per case somewhere else in the head. Volumetric Dice barely registers
these; lesion-wise Dice - what BraTS ranks on - charges a full zero for every
spurious connected component. On the glioma test split, SNFH scores 0.860
volumetric on cases with no false-positive lesion and 0.842 on cases with at
least one. The main tumour is segmented *just as well* in both groups. The
spurious components are additive noise, not a symptom of a degraded
segmentation, which is what makes deleting them a coherent goal rather than a
patch over a worse problem.

An earlier attempt to delete them by **component size** largely failed: a single
global threshold gained +0.009 on validation and -0.0006 on test; per-class
thresholds gained +0.039 on validation and +0.0126 on test. Size does not
separate real from spurious components, and structurally it was the only lever
available, because ``predict.py`` did ``pred = logits.argmax(dim=1)`` and threw
every probability away at that line. ``--save-components`` keeps them.

This module scores each component with a logistic regression over confidence,
geometry and spatial context, and then thresholds the posterior at a value
*derived from the scoring function* rather than tuned - see
``break_even_threshold`` for the derivation. That is the strongest claim this
change makes, so the derivation is written out in full where a reader can check
it.

    python -m mri3d.select --run gli_main --diagnose   # separation + per-feature AUC
    python -m mri3d.select --run gli_main --fit        # fit on val, derive p*, calibrate
    python -m mri3d.select --run gli_main --apply      # score test, once

Discipline: fitting and thresholding happen on the validation split only, and
the test split is scored exactly once. The size-threshold experiment saw
validation overstate the test gain threefold; expect the same haircut.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
from pathlib import Path

import numpy as np

from . import paths
from .metrics import MIN_LESION_VOXELS

# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------

_BOOL_COLS = {"matched", "detected"}
_INT_COLS = {"class_value", "comp_id", "ref_id", "n_voxels", "rank_by_volume",
             "rank_by_prob", "n_comps_in_class"}
_STR_COLS = {"case_id", "run", "split", "dataset", "class_name",
             "matched_ref_ids", "matched_comp_ids"}


def _coerce(key: str, value: str):
    if key in _STR_COLS:
        return value
    if value == "":
        return None
    if key in _BOOL_COLS:
        return value == "True"
    if key in _INT_COLS:
        return int(value)
    return float(value)  # "nan" parses to nan, which is what we want to see


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _coerce(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def components_path(run: str, split: str) -> Path:
    return paths.REPORTS / f"components_{run}_{split}.csv"


def ref_lesions_path(run: str, split: str) -> Path:
    return paths.REPORTS / f"ref_lesions_{run}_{split}.csv"


def image_features_path(run: str, split: str) -> Path:
    return paths.REPORTS / f"image_features_{run}_{split}.csv"


def load_components(run: str, split: str) -> list[dict]:
    """Components, with the image-derived features joined on when they exist.

    The two CSVs are produced by different passes - one during inference, one
    later over the masks on disk - and are tied together only by
    ``scipy.ndimage.label`` numbering the same mask identically. That is true,
    and it is checked rather than assumed: ``n_voxels_check`` must agree with
    the voxel count recorded at inference, or the join is refused. Silently
    attaching one component's intensities to another's geometry would be
    undetectable in every number downstream.
    """
    comps = load_csv(components_path(run, split))
    path = image_features_path(run, split)
    if not path.exists():
        return comps
    extra = {(r["case_id"], r["class_name"], r["comp_id"]): r for r in load_csv(path)}
    missing = 0
    for c in comps:
        k = (c["case_id"], c["class_name"], c["comp_id"])
        r = extra.get(k)
        if r is None:
            missing += 1
            continue
        if int(r["n_voxels_check"]) != int(c["n_voxels"]):
            raise RuntimeError(
                f"image features disagree with the components CSV for {k}: "
                f"{r['n_voxels_check']} voxels against {c['n_voxels']}. The two came from "
                f"different masks; re-run mri3d.image_features.")
        c.update({k2: v for k2, v in r.items() if k2 not in ("n_voxels_check", *_KEY)})
    if missing:
        raise RuntimeError(f"{missing} components have no image features; "
                           f"re-run mri3d.image_features --run {run} --split {split}")
    return comps


_KEY = ("case_id", "class_name", "comp_id")


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

# Only components at or above MIN_LESION_VOXELS are ever scored or deleted.
# Below that size the metric is blind to them: an unmatched 30-voxel component
# is not charged as a false positive, so deleting it cannot gain a thing and
# can only shave volumetric Dice. Including them would also inflate every AUC
# below, because they are a large and trivially separable population.
DECIDABLE_MIN_VOXELS = MIN_LESION_VOXELS

# Raw columns shown in the separation diagnostic, in the order §5 asks for.
DIAGNOSTIC_FEATURES = [
    "dist_to_dominant_mm", "mean_prob", "p90_prob", "max_prob",
    "mean_border_prob", "log_n_voxels", "mc_mean_var", "rank_by_prob",
]

# What the logistic regression actually consumes. Three of the raw columns are
# heavy-tailed over orders of magnitude (voxel counts 50 -> 10^5, distances
# 0 -> 150 mm with a spike at 0, MC variance 10^-6 -> 10^-1). A linear logit in
# the raw value would be dominated by the tail; a log makes "ten times further
# away" a constant step, which is the shape the decision actually has.
FIT_FEATURES = [
    "log_n_voxels", "mean_prob", "p90_prob", "max_prob", "mean_border_prob",
    "log1p_dist", "log_mc_var", "rank_by_prob", "n_comps_in_class",
]

# Features that read the scan rather than the prediction, added by
# mri3d.image_features. Used only when that CSV exists, so every earlier result
# reproduces unchanged. The contrasts and asymmetries carry the signal; the raw
# per-modality means are included because what counts as "bright" differs by
# class - a resection cavity is dark on FLAIR and bright on T2, oedema the
# other way round.
IMAGE_FIT_FEATURES = [
    "z_t1c_mean", "z_t1c_contrast", "z_t1c_mirror",
    "z_t1n_contrast",
    "z_t2f_mean", "z_t2f_contrast", "z_t2f_mirror",
    "z_t2w_mean", "z_t2w_contrast", "z_t2w_mirror",
    "enhancement", "flair_minus_t2",
    "log1p_edge", "frac_outside_brain",
]

# Shown in the separation diagnostic when available: the four that state a
# radiological claim directly, rather than all fourteen.
IMAGE_DIAGNOSTIC_FEATURES = [
    "z_t2f_contrast", "z_t2f_mirror", "enhancement", "flair_minus_t2",
]


# Derived image features and the raw column each needs, so that asking for one
# without the image_features CSV drops it rather than handing the fit a column
# of NaN.
_DERIVED_SOURCE = {"log1p_edge": "dist_to_brain_edge_mm"}


def available_image_features(rows: list[dict], names: list[str]) -> list[str]:
    if not rows:
        return []
    present = set(rows[0])
    return [n for n in names if _DERIVED_SOURCE.get(n, n) in present]


def derive_features(row: dict) -> dict:
    """Columns that are transforms of the CSV, computed the same way everywhere."""
    n = max(1, int(row["n_voxels"]))
    dist = row.get("dist_to_dominant_mm")
    mc = row.get("mc_mean_var")
    edge = row.get("dist_to_brain_edge_mm")
    return {
        "log_n_voxels": math.log(n),
        "log1p_dist": math.log1p(dist) if dist is not None and np.isfinite(dist) else float("nan"),
        # 1e-9 floors the log for components the MC passes agreed on exactly.
        "log_mc_var": (math.log10(mc + 1e-9) if mc is not None and np.isfinite(mc)
                       else float("nan")),
        # Depth into the brain is meaningful in ratios near the surface, where
        # the partial-volume rind lives, and flat once you are well inside.
        "log1p_edge": (math.log1p(edge) if edge is not None and np.isfinite(edge)
                       else float("nan")),
    }


def feature_table(rows: list[dict], names: list[str]) -> np.ndarray:
    out = np.empty((len(rows), len(names)), dtype=np.float64)
    for i, r in enumerate(rows):
        d = {**r, **derive_features(r)}
        for j, name in enumerate(names):
            v = d.get(name)
            out[i, j] = float("nan") if v is None else float(v)
    return out


def class_columns(rows: list[dict]) -> list[str]:
    return sorted({r["class_name"] for r in rows})


def design_matrix(rows: list[dict], names: list[str], classes: list[str],
                  impute: dict[str, float] | None = None
                  ) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Features plus one-hot class indicators, with NaNs imputed by column median.

    ``impute`` is stored in the fitted model and replayed at apply time, so a
    test component with a missing feature is filled with the *validation*
    median rather than a test-derived one. Filling from the test split would be
    a small, real leak of test statistics into a decision rule.

    Class enters as an indicator rather than as four separate models. One shared
    model with a class term has 13 coefficients instead of 4 x 10, shares the
    confidence and geometry structure that is common to all four sub-regions,
    and keeps the coefficients readable - which matters, because they are part
    of the deliverable. The first class is the reference level (dropped), so a
    class coefficient reads as "relative to NETC".
    """
    X = feature_table(rows, names)
    for j, name in enumerate(names):
        if not np.isfinite(X[:, j]).any():
            # A feature the fitted model expects is absent from every row -
            # almost always a missing image_features CSV for this split. Median
            # imputation would quietly fill it with a constant and the selector
            # would keep working while ignoring a third of what it was fitted on.
            raise RuntimeError(
                f"feature '{name}' is missing for every component. If this is an "
                f"image feature, run: python -m mri3d.image_features --run <run> "
                f"--split <split>")
    if impute is None:
        impute = {}
        for j, name in enumerate(names):
            col = X[:, j]
            finite = col[np.isfinite(col)]
            impute[name] = float(np.median(finite)) if finite.size else 0.0
    for j, name in enumerate(names):
        col = X[:, j]
        col[~np.isfinite(col)] = impute[name]
    ind = np.zeros((len(rows), max(0, len(classes) - 1)))
    for i, r in enumerate(rows):
        k = classes.index(r["class_name"])
        if k > 0:
            ind[i, k - 1] = 1.0
    cols = list(names) + [f"is_{c}" for c in classes[1:]]
    return np.hstack([X, ind]) if ind.size else X, cols, impute


# --------------------------------------------------------------------------
# The threshold, derived rather than tuned
# --------------------------------------------------------------------------

def break_even_threshold(L: float, d: float) -> float:
    r"""The posterior at which keeping a marginal component breaks even.

    Take a case whose *other* lesion-wise entries number N and average L, and
    one marginal predicted component c with

        p = P(c is real),  d = the Dice c would score if it is real.

    Lesion-wise Dice for the case is the mean over a list of entries: one per
    scored reference lesion, plus one zero per spurious predicted component.

    * If c is **real**, its reference lesion is an entry either way. Dropping c
      leaves that entry at 0 (the lesion is missed); keeping it makes the entry
      d. The case mean changes by ``+d / N``.
    * If c is **spurious**, keeping it appends a new entry of 0, so the mean
      goes from L to ``N*L / (N+1)``. The case mean changes by ``-L / (N+1)``.

    The expected change from keeping c is therefore

        E[delta] = p * d/N  -  (1-p) * L/(N+1)

    Setting E[delta] = 0 and writing N/(N+1) -> 1 (the two denominators differ
    by one entry; for N >= 3 the correction is under 25% and it vanishes in the
    limit):

        p*d = (1-p)*L    =>    p (d + L) = L    =>    p* = L / (L + d)

    Two consequences worth stating, because both are easy to get backwards:

    1. **The optimal rule is a flat threshold on the posterior.** N cancels, so
       the bar does not rise as more components are kept. The familiar
       intuition that the third lesion should need more evidence than the first
       is real, but it lives in the *prior* - the posterior that the
       third-ranked component is genuine is simply lower - not in the metric.
       A rule that escalated the threshold as well would be double-counting.
    2. **p\* falls as L falls.** A case that is already scoring badly has less
       to lose from one more zero, so the rule correctly becomes more willing to
       gamble on an extra lesion there. With L = 0.72 and d = 0.70, p* = 0.507.

    Assumption worth flagging: the derivation treats c as the sole detector of
    its reference lesion. When a second component also touches that lesion,
    dropping c costs only some Dice rather than the whole entry, so the true
    break-even for those components is lower than p*. That makes the rule
    slightly conservative, not slightly wrong.
    """
    return float(L / (L + d))


def estimate_L_and_d(comp_rows: list[dict], ref_rows: list[dict]) -> dict:
    """Measure L and d on the validation split. Nothing here is hand-set.

    ``L`` - the value of a typical non-spurious lesion-wise entry - is the mean
    Dice over scored reference lesions, which is exactly what an entry is worth
    when it is not one of the zeros a false positive adds.

    ``d`` - the Dice a correct detection scores - is measured over *marginal*
    components, meaning every matched component that is not the largest in its
    (case, class). That restriction matters: the dominant mass is matched with
    Dice ~0.85 and its posterior is never anywhere near the threshold, so
    including it would inflate d, lower p*, and calibrate the rule on a decision
    it never has to make. The unrestricted figure is reported alongside so the
    effect of the choice is visible rather than assumed.
    """
    ref_dice = [r["dice"] for r in ref_rows]
    L = float(np.mean(ref_dice))

    matched = [c for c in comp_rows if c["matched"]]
    marginal = [c for c in matched if c["rank_by_volume"] > 1]
    d_all = float(np.mean([c["dice_with_matched"] for c in matched])) if matched else 0.0
    d_marg = float(np.mean([c["dice_with_matched"] for c in marginal])) if marginal else d_all

    per_class = {}
    for name in sorted({c["class_name"] for c in comp_rows}):
        rl = [r["dice"] for r in ref_rows if r["class_name"] == name]
        cm = [c["dice_with_matched"] for c in matched
              if c["class_name"] == name and c["rank_by_volume"] > 1]
        if rl and cm:
            per_class[name] = {"L": round(float(np.mean(rl)), 4),
                               "d": round(float(np.mean(cm)), 4),
                               "p_star": round(break_even_threshold(
                                   float(np.mean(rl)), float(np.mean(cm))), 4)}

    return {
        "L": round(L, 4),
        "L_definition": "mean Dice over scored reference lesions, validation split",
        "n_ref_lesions": len(ref_rows),
        "d": round(d_marg, 4),
        "d_definition": "mean dice_with_matched over matched components that are not the "
                        "largest in their (case, class)",
        "n_marginal_matched": len(marginal),
        "d_all_matched": round(d_all, 4),
        "p_star": round(break_even_threshold(L, d_marg), 4),
        "p_star_if_d_all_matched": round(break_even_threshold(L, d_all), 4),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC, ties counted as half. Returns nan if a class is empty."""
    pos, neg = scores[labels], scores[~labels]
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < vals.size:  # average ranks within each tie group
        j = i
        while j + 1 < vals.size and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def feature_aucs(rows: list[dict], names: list[str]) -> dict:
    """Per-feature AUC for discriminating matched from unmatched, overall and per class.

    Reported as-is, not flipped to be >= 0.5: the direction is information. A
    feature at 0.20 separates as strongly as one at 0.80, in the opposite sense,
    and reporting |AUC - 0.5| would hide which way round the model should read it.
    """
    X = feature_table(rows, names)
    y = np.array([r["matched"] for r in rows], dtype=bool)
    cls = np.array([r["class_name"] for r in rows])
    out: dict = {"overall": {}, "per_class": {}, "n": int(len(rows)),
                 "n_matched": int(y.sum())}
    for j, name in enumerate(names):
        out["overall"][name] = round(auc(X[:, j], y), 4)
    for c in sorted(set(cls.tolist())):
        m = cls == c
        out["per_class"][c] = {
            "n": int(m.sum()), "n_matched": int(y[m].sum()),
            **{name: round(auc(X[m, j], y[m]), 4) for j, name in enumerate(names)},
        }
    return out


def _fig_to_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def separation_page(rows: list[dict], names: list[str], aucs: dict, out: Path) -> Path:
    """Overlaid histograms of each feature, matched vs unmatched, faceted by class.

    Written as a single self-contained HTML file with the PNGs inlined as data
    URIs, the same pattern ``heatmap_page.py`` uses: the page opens straight
    from disk with no web server and nothing to keep in sync beside it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = feature_table(rows, names)
    y = np.array([r["matched"] for r in rows], dtype=bool)
    cls = np.array([r["class_name"] for r in rows])
    classes = sorted(set(cls.tolist()))

    panels = []
    for c in classes:
        m = cls == c
        fig, axes = plt.subplots(2, math.ceil(len(names) / 2),
                                 figsize=(3.1 * math.ceil(len(names) / 2), 5.4))
        for ax, (j, name) in zip(axes.ravel(), enumerate(names)):
            a, b = X[m & y, j], X[m & ~y, j]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size + b.size == 0:
                ax.set_axis_off()
                continue
            lo, hi = np.percentile(np.concatenate([a, b]), [0.5, 99.5])
            if hi <= lo:
                hi = lo + 1e-6
            bins = np.linspace(lo, hi, 36)
            # Densities, not counts: unmatched components outnumber matched ones
            # several to one, so raw counts would show the imbalance rather than
            # the separation.
            ax.hist(b, bins=bins, density=True, color="#ef4444", alpha=0.55, label="spurious")
            ax.hist(a, bins=bins, density=True, color="#3b82f6", alpha=0.55, label="real")
            ax.set_title(f"{name}\nAUC {aucs['per_class'][c][name]:.3f}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_yticks([])
        for ax in axes.ravel()[len(names):]:
            ax.set_axis_off()
        axes.ravel()[0].legend(fontsize=7, loc="upper left")
        n, nm = aucs["per_class"][c]["n"], aucs["per_class"][c]["n_matched"]
        fig.suptitle(f"{c} - {nm} real / {n - nm} spurious components (validation)",
                     fontsize=11)
        panels.append((c, _fig_to_uri(fig), n, nm))
        plt.close(fig)

    rowsep = "\n".join(
        f'<section><h2>{c} <small>{nm} real / {n - nm} spurious</small></h2>'
        f'<img src="{uri}" alt="{c} feature separation"></section>'
        for c, uri, n, nm in panels)
    table = "\n".join(
        "<tr><td>%s</td>%s</tr>" % (name, "".join(
            f"<td>{aucs['per_class'][c][name]:.3f}</td>" for c in classes)
            + f"<td><b>{aucs['overall'][name]:.3f}</b></td>")
        for name in names)
    html = f"""<!doctype html><meta charset="utf-8">
<title>Component separation</title>
<style>
 body{{font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;
      color:#111;background:#fff}}
 h1{{margin-bottom:.2rem}} h2 small{{font-weight:400;color:#666}}
 img{{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}}
 table{{border-collapse:collapse;margin:1rem 0}}
 th,td{{border:1px solid #e5e7eb;padding:.3rem .6rem;text-align:right;font-variant-numeric:tabular-nums}}
 th:first-child,td:first-child{{text-align:left;font-family:ui-monospace,monospace}}
 p.note{{color:#444;max-width:70ch}}
</style>
<h1>Does anything separate real from spurious components?</h1>
<p class="note">Validation split, {aucs['n']} predicted components of at least
{DECIDABLE_MIN_VOXELS} voxels, of which {aucs['n_matched']} touch a scored reference
lesion. Components below {DECIDABLE_MIN_VOXELS} voxels are excluded: lesion-wise Dice
does not charge for them, so no selector should be deciding their fate.
AUC is for discriminating real from spurious; 0.5 is no information, and a value
below 0.5 separates just as well in the opposite direction.</p>
<h2>Per-feature AUC</h2>
<table><tr><th>feature</th>{''.join(f'<th>{c}</th>' for c in classes)}<th>all</th></tr>
{table}</table>
{rowsep}
"""
    out.write_text(html, encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# The selector
# --------------------------------------------------------------------------

def fit_selector(rows: list[dict], names: list[str],
                 classes: list[str] | None = None) -> dict:
    """Logistic regression on validation components. Returns a portable dict.

    Choices, and what they cost:

    * **Logistic regression, not gradient boosting.** The coefficients are part
      of the deliverable - "distance from the dominant mass carries a weight of
      -1.4 per log-mm" is a claim a reader can argue with, and a boosted
      ensemble is not. Boosting is worth reaching for only if this underfits,
      which the AUC comparison in the report is there to test.
    * **No ``class_weight="balanced"``.** It is the usual reflex for imbalanced
      data and it would be wrong here: reweighting the classes rescales the
      prior, and the whole point of §2.1 is to threshold a genuine posterior at
      a derived value. A balanced model outputs a posterior for a population
      that does not exist. The reliability diagram is the check on that.
    * **Features standardised** on validation means/stds, which are stored here
      and replayed at apply time. Standardising on the test split would leak.
    """
    from sklearn.linear_model import LogisticRegression

    # ``classes`` is passed in by the cross-validation loop so that every fold
    # shares one encoding. Deriving it per fold would silently change the number
    # of columns if a fold happened to contain no NETC at all.
    classes = classes or class_columns(rows)
    X, cols, impute = design_matrix(rows, names, classes)
    y = np.array([r["matched"] for r in rows], dtype=int)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((X - mean) / std, y)
    return {
        "features": names,
        "classes": classes,
        "columns": cols,
        "impute": impute,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "n_train": int(len(rows)),
        "n_train_matched": int(y.sum()),
        "min_voxels": DECIDABLE_MIN_VOXELS,
    }


def posterior(model: dict, rows: list[dict]) -> np.ndarray:
    """P(component is real) for each row, from a fitted selector dict."""
    if not rows:
        return np.zeros(0)
    X, _, _ = design_matrix(rows, model["features"], model["classes"], model["impute"])
    z = ((X - np.array(model["mean"])) / np.array(model["std"])) @ np.array(model["coef"])
    return 1.0 / (1.0 + np.exp(-(z + model["intercept"])))


def out_of_fold_posterior(rows: list[dict], names: list[str], n_folds: int = 5,
                          seed: int = 0) -> np.ndarray:
    """Posteriors from folds that never saw the component's own case.

    Splitting by component would leak: components from one case share a
    dominant mass, a confidence scale and a difficulty, so a sibling in the
    training fold tells you most of the answer. Grouping by ``case_id`` is what
    makes the reliability diagram and the AUC below honest estimates of what the
    fitted model does on cases it has not seen.
    """
    cases = sorted({r["case_id"] for r in rows})
    classes = class_columns(rows)
    rng = np.random.default_rng(seed)
    fold_of = {c: int(f) for c, f in zip(cases, rng.permutation(len(cases)) % n_folds)}
    out = np.zeros(len(rows))
    for f in range(n_folds):
        tr = [r for r in rows if fold_of[r["case_id"]] != f]
        te_idx = [i for i, r in enumerate(rows) if fold_of[r["case_id"]] == f]
        if not te_idx or not tr:
            continue
        m = fit_selector(tr, names, classes)
        out[te_idx] = posterior(m, [rows[i] for i in te_idx])
    return out


def reliability(probs_raw: np.ndarray, probs_fit: np.ndarray, y: np.ndarray,
                out: Path, n_bins: int = 10) -> dict:
    """Predicted vs observed probability that a component is real.

    A ``DiceCELoss``-trained network with ``dropout_prob=0.2`` has no reason to
    be calibrated, and the Dice term specifically distorts probabilities because
    it optimises overlap, not likelihood - so ``mean_prob`` is a confidence
    score, not a probability, and thresholding it at a derived p* would be
    meaningless. The fitted posterior is calibrated by construction (logistic
    regression maximises the likelihood of the labels). This plot is the check
    that it actually is, on held-out folds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(p):
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        xs, ys, ns = [], [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() < 5:
                continue
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
            ns.append(int(m.sum()))
        return np.array(xs), np.array(ys), np.array(ns)

    def ece(p):
        x, o, n = curve(p)
        return float(np.sum(n * np.abs(x - o)) / max(1, n.sum())) if n.size else float("nan")

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="perfect calibration")
    for p, name, colour in ((probs_raw, "raw mean_prob", "#ef4444"),
                            (probs_fit, "fitted posterior", "#3b82f6")):
        x, o, n = curve(p)
        ax.plot(x, o, "o-", color=colour, label=f"{name}  (ECE {ece(p):.3f})")
        for xi, oi, ni in zip(x, o, n):
            ax.annotate(str(ni), (xi, oi), fontsize=6, xytext=(2, 4),
                        textcoords="offset points", color=colour)
    ax.set_xlabel("predicted P(component is real)")
    ax.set_ylabel("observed fraction real")
    ax.set_title("Reliability, validation (out-of-fold, grouped by case)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"ece_raw_mean_prob": round(ece(probs_raw), 4),
            "ece_fitted_posterior": round(ece(probs_fit), 4),
            "auc_raw_mean_prob": round(auc(probs_raw, y.astype(bool)), 4),
            "auc_fitted_posterior": round(auc(probs_fit, y.astype(bool)), 4)}


# --------------------------------------------------------------------------
# Detection metrics: sensitivity, FP/case, FROC
# --------------------------------------------------------------------------

def detection_curve(comp_rows: list[dict], ref_rows: list[dict], probs: np.ndarray,
                    n_cases: int, thresholds: np.ndarray) -> dict:
    """Per-lesion sensitivity against false positives per case, as p* sweeps 0 -> 1.

    Computed from the component and reference tables rather than by re-scoring
    masks at every threshold, which would be ~100 x 203 full re-segmentations
    for the same answer. A reference lesion counts as detected while at least
    one of the components touching it survives; a surviving unmatched component
    of at least ``MIN_LESION_VOXELS`` is a false positive. Those are the same two
    definitions ``metrics.match_components`` uses, so the curve and the Dice
    table cannot drift apart.
    """
    key = {(c["case_id"], c["class_name"], c["comp_id"]): i for i, c in enumerate(comp_rows)}
    # Components too small to decide on are never deleted, so they survive at
    # every threshold and keep their reference lesion detected.
    survives_always = {k for k, i in key.items()
                       if comp_rows[i]["n_voxels"] < DECIDABLE_MIN_VOXELS}
    unmatched = np.array([not c["matched"] for c in comp_rows])
    decidable = np.array([c["n_voxels"] >= DECIDABLE_MIN_VOXELS for c in comp_rows])

    ref_keys: list[list[tuple]] = []
    for r in ref_rows:
        ids = [int(x) for x in (r["matched_comp_ids"] or "").split() if x]
        ref_keys.append([(r["case_id"], r["class_name"], i) for i in ids])

    out = {"threshold": [], "sensitivity": [], "fp_per_case": [], "n_detected": [], "n_fp": []}
    for t in thresholds:
        keep = (probs >= t) | ~decidable
        n_fp = int((keep & unmatched & decidable).sum())
        det = 0
        for ks in ref_keys:
            if any(k in survives_always or (k in key and keep[key[k]]) for k in ks):
                det += 1
        out["threshold"].append(round(float(t), 4))
        out["sensitivity"].append(round(det / max(1, len(ref_rows)), 4))
        out["fp_per_case"].append(round(n_fp / max(1, n_cases), 4))
        out["n_detected"].append(det)
        out["n_fp"].append(n_fp)
    return out


def froc_plot(curves: dict[str, dict], p_star: float, out: Path) -> Path:
    """Sensitivity vs false positives per case, with p* marked.

    The figure that makes the trade-off legible: a single before/after pair of
    numbers hides that the selector is one operating point on a curve, and that
    the curve is what the model can actually do.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colours = {"val": "#3b82f6", "test": "#ef4444"}
    # Stagger the callouts: the two curves sit almost on top of each other at
    # p*, which is the point, so their labels would otherwise overlap.
    offsets = [(14, 26), (18, -34), (14, 54)]
    for n, (split, c) in enumerate(curves.items()):
        x, y_ = np.array(c["fp_per_case"]), np.array(c["sensitivity"])
        colour = colours.get(split, "#111")
        ax.plot(x, y_, "-", color=colour, lw=1.6, label=split)
        t = np.array(c["threshold"])
        k = int(np.argmin(np.abs(t - p_star)))
        ax.plot(x[k], y_[k], "o", color=colour, ms=8, mfc="white", mew=2)
        ax.annotate(f"{split}: {y_[k]:.3f} sensitivity\nat {x[k]:.2f} FP/case",
                    (x[k], y_[k]), fontsize=8, xytext=offsets[n % len(offsets)],
                    textcoords="offset points", color=colour, ha="left",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.8, alpha=0.6))
    ax.set_xlabel("false-positive lesions per case")
    ax.set_ylabel("per-lesion sensitivity")
    ax.set_title(f"FROC: sweeping the selector's posterior threshold\n"
                 f"circles mark the derived p* = {p_star:.3f}, which was not tuned on either curve",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Applying the selector to masks
# --------------------------------------------------------------------------

def keep_table(model: dict, comp_rows: list[dict], p_star: float) -> dict[tuple, bool]:
    """``(case_id, class_name, comp_id) -> keep?`` for every component."""
    probs = posterior(model, comp_rows)
    return {(c["case_id"], c["class_name"], c["comp_id"]):
            bool(c["n_voxels"] < DECIDABLE_MIN_VOXELS or probs[i] >= p_star)
            for i, c in enumerate(comp_rows)}


def filter_by_selector(mask: np.ndarray, case_id: str, class_values: dict[int, str],
                       keep: dict[tuple, bool], sizes: dict[tuple, int] | None = None
                       ) -> np.ndarray:
    """Zero out the components the selector rejected.

    The components CSV was written during inference, from the logits; this runs
    later against the mask on disk. The two are tied together by
    ``scipy.ndimage.label`` producing the same component ids for the same mask -
    true, but worth verifying rather than trusting, so the caller passes
    ``sizes`` and any disagreement in voxel count raises instead of silently
    deleting the wrong blob.
    """
    from scipy import ndimage

    out = mask.copy()
    for value, name in class_values.items():
        if value == 0:
            continue
        binary = mask == value
        if not binary.any():
            continue
        lab, n = ndimage.label(binary)
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        drop = []
        for j in range(1, n + 1):
            k = (case_id, name, j)
            if sizes is not None and k in sizes and sizes[k] != int(counts[j]):
                raise RuntimeError(
                    f"component mismatch for {k}: csv says {sizes[k]} voxels, mask on disk "
                    f"has {counts[j]}. The mask and the components CSV come from different "
                    f"inference runs.")
            if not keep.get(k, True):
                drop.append(j)
        if drop:
            out[np.isin(lab, drop) & binary] = 0
    return out


def load_selector(run: str) -> dict:
    return json.loads((paths.REPORTS / f"selector_{run}.json").read_text(encoding="utf-8"))


def compare_variants(run: str, dataset: str, split: str, thresholds: dict[int, int],
                     keep: dict[tuple, bool], sizes: dict[tuple, int],
                     limit: int = 0) -> dict[str, list[dict]]:
    """Score every case three ways from a single load of the mask off disk.

    ``postprocess.evaluate`` would re-read 203 pairs of volumes for each of the
    three variants. Reading once and scoring three times is the same arithmetic
    at a third of the I/O, and it guarantees the three columns describe the same
    masks rather than three separate passes that could, in principle, diverge.
    """
    from .data import case_dicts
    from .metrics import score_case
    from .postprocess import _load_pair, filter_small

    pred_dir = paths.OUTPUTS / "runs" / run / f"pred_{split}"
    class_values = paths.GLI_LABELS if dataset == "gli" else paths.MEN_LABELS
    recs = [r for r in case_dicts(dataset, split)
            if (pred_dir / f"{r['case_id']}_pred.nii.gz").exists()]
    if limit:
        recs = recs[:limit]

    out: dict[str, list[dict]] = {"none": [], "size": [], "selector": []}
    for rec in recs:
        pred, ref = _load_pair(rec, pred_dir)
        cid = rec["case_id"]
        variants = {
            "none": pred,
            "size": filter_small(pred, thresholds),
            "selector": filter_by_selector(pred, cid, class_values, keep, sizes),
        }
        for name, mask in variants.items():
            out[name].append({"case_id": cid, **score_case(mask, ref, class_values)})
    return out


def case_mean_lesion(row: dict, regions: list[str]) -> float:
    """Per-case mean lesion-wise Dice over the regions actually present."""
    vals = [row[f"lesion_dice_{r}"] for r in regions if row.get(f"present_{r}")]
    return float(np.mean(vals)) if vals else float("nan")


def multifocal_tradeoff(ref_rows: list[dict], comp_rows: list[dict], probs: np.ndarray,
                        p_star: float, before: list[dict], after: list[dict],
                        regions: list[str], distances=(10.0, 20.0, 40.0)) -> dict:
    """What the selector costs on the cases it is least entitled to help.

    Lesion-wise Dice rewards deleting a distant real lesion, because clean
    cases far outnumber multifocal ones: the zero saved on 100 single-focus
    cases outweighs the zero created on 10 multifocal ones. That is a property
    of the metric, not evidence that the deletion was correct, so the two sides
    are quantified separately here rather than netted into one number.
    """
    key = {(c["case_id"], c["class_name"], c["comp_id"]): i for i, c in enumerate(comp_rows)}
    size = {k: comp_rows[i]["n_voxels"] for k, i in key.items()}

    def survives(k) -> bool:
        if k not in key:
            return True
        return size[k] < DECIDABLE_MIN_VOXELS or bool(probs[key[k]] >= p_star)

    lost, kept = [], []
    for r in ref_rows:
        ids = [int(x) for x in (r["matched_comp_ids"] or "").split() if x]
        ks = [(r["case_id"], r["class_name"], i) for i in ids]
        if not r["detected"]:
            continue
        (kept if any(survives(k) for k in ks) else lost).append(r)

    b = {r["case_id"]: r for r in before}
    a = {r["case_id"]: r for r in after}

    out: dict = {"n_scored_ref_lesions": len(ref_rows),
                 "n_detected_before": len(lost) + len(kept),
                 "n_detections_deleted": len(lost),
                 "by_distance": {}}
    for D in distances:
        distant = [r for r in ref_rows if np.isfinite(r["dist_to_dominant_mm"])
                   and r["dist_to_dominant_mm"] > D]
        distant_cases = sorted({r["case_id"] for r in distant})
        deleted = [r for r in lost if r["dist_to_dominant_mm"] > D]
        hurt_cases = sorted({r["case_id"] for r in deleted})
        sub = [c for c in distant_cases if c in b and c in a]
        d_sub = ([case_mean_lesion(a[c], regions) - case_mean_lesion(b[c], regions)
                  for c in sub] if sub else [])
        rest = [c for c in b if c not in set(distant_cases)]
        d_rest = [case_mean_lesion(a[c], regions) - case_mean_lesion(b[c], regions)
                  for c in rest]
        out["by_distance"][str(int(D))] = {
            "n_lesions_beyond": len(distant),
            "n_cases_with_one": len(distant_cases),
            "n_detected_lesions_deleted": len(deleted),
            "n_cases_hurt": len(hurt_cases),
            "delta_mean_lesion_dice_on_those_cases":
                round(float(np.nanmean(d_sub)), 4) if d_sub else None,
            "delta_mean_lesion_dice_on_the_rest":
                round(float(np.nanmean(d_rest)), 4) if d_rest else None,
        }
    return out


def detection_summary(comp_rows: list[dict], ref_rows: list[dict], probs: np.ndarray,
                      p_star: float, n_cases: int) -> dict:
    """Per-lesion sensitivity and false positives per case, before and after.

    These are the detection metrics the README already implies by calling
    connected components "the detections"; they need no retraining and reuse the
    same two CSVs as everything else here.
    """
    curve = detection_curve(comp_rows, ref_rows, probs, n_cases, np.array([0.0, p_star]))
    out = {"n_cases": n_cases,
           "before": {"sensitivity": curve["sensitivity"][0],
                      "fp_per_case": curve["fp_per_case"][0],
                      "n_detected": curve["n_detected"][0], "n_fp": curve["n_fp"][0]},
           "after": {"sensitivity": curve["sensitivity"][1],
                     "fp_per_case": curve["fp_per_case"][1],
                     "n_detected": curve["n_detected"][1], "n_fp": curve["n_fp"][1]},
           "per_class": {}}
    for name in sorted({c["class_name"] for c in comp_rows}):
        cr = [c for c in comp_rows if c["class_name"] == name]
        pr = probs[[i for i, c in enumerate(comp_rows) if c["class_name"] == name]]
        rr = [r for r in ref_rows if r["class_name"] == name]
        c2 = detection_curve(cr, rr, pr, n_cases, np.array([0.0, p_star]))
        out["per_class"][name] = {
            "n_ref_lesions": len(rr),
            "sensitivity": [c2["sensitivity"][0], c2["sensitivity"][1]],
            "fp_per_case": [c2["fp_per_case"][0], c2["fp_per_case"][1]],
            "n_fp": [c2["n_fp"][0], c2["n_fp"][1]],
        }
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def decidable(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["n_voxels"] >= DECIDABLE_MIN_VOXELS]


def _diagnose(run: str) -> dict:
    rows = decidable(load_components(run, "val"))
    names = list(DIAGNOSTIC_FEATURES) + available_image_features(
        rows, IMAGE_DIAGNOSTIC_FEATURES)
    if all(r.get("mc_mean_var") is None for r in rows):
        names = [n for n in names if n != "mc_mean_var"]
        print("note: no MC-dropout column in the validation CSV; dropping that feature")
    a = feature_aucs(rows, names)
    out = separation_page(rows, names, a, paths.REPORTS / "component_separation.html")
    print(f"{a['n']} decidable components, {a['n_matched']} real "
          f"({a['n_matched'] / a['n']:.1%})\n")
    print(f"{'feature':<24}" + "".join(f"{c:>9}" for c in sorted(a['per_class'])) + f"{'all':>9}")
    for n in names:
        print(f"{n:<24}" + "".join(f"{a['per_class'][c][n]:>9.3f}"
                                   for c in sorted(a['per_class'])) + f"{a['overall'][n]:>9.3f}")
    print(f"\nwrote {out}")
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--diagnose", action="store_true",
                    help="per-feature AUC and separation plots, validation only")
    ap.add_argument("--fit", action="store_true",
                    help="fit on validation, derive p*, write reports/selector_{run}.json")
    ap.add_argument("--apply", action="store_true",
                    help="score the held-out split with the fitted selector, ONCE")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="which split --apply scores")
    ap.add_argument("--dataset", default="gli", choices=["gli", "men_rt"])
    ap.add_argument("--force", action="store_true",
                    help="re-run --apply on a split that has already been scored")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image-features", action="store_true",
                    help="add the features from mri3d.image_features to the fit. Off by "
                         "default: measured on validation they add +0.002 out-of-fold AUC "
                         "at best and cost calibration, because the segmentation CNN has "
                         "already consumed the intensities (see reports/results.md)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.diagnose:
        _diagnose(args.run)

    if args.fit:
        comp = decidable(load_components(args.run, "val"))
        refs = load_csv(ref_lesions_path(args.run, "val"))
        names = list(FIT_FEATURES)
        if args.image_features:
            names += available_image_features(comp, IMAGE_FIT_FEATURES)
        if all(r.get("mc_mean_var") is None for r in comp):
            names = [n for n in names if n != "log_mc_var"]

        # ``d`` is estimated over the same population the selector decides on -
        # components at or above the metric's floor - so that the threshold is
        # calibrated on the decisions it will actually make.
        derived = estimate_L_and_d(comp, refs)
        p_star = derived["p_star"]
        model = fit_selector(comp, names)

        oof = out_of_fold_posterior(comp, names, seed=args.seed)
        y = np.array([r["matched"] for r in comp], dtype=int)
        raw = np.array([r["mean_prob"] for r in comp])
        cal = reliability(raw, oof, y, paths.REPORTS / f"reliability_{args.run}.png")

        # How much of the validation gain is the model having seen these
        # components? Detection at p* with in-sample posteriors, against the
        # same thing with out-of-fold ones. The gap between the two is
        # optimism; the gap between out-of-fold validation and test is
        # distribution shift. Separating them costs nothing here - both
        # posterior vectors are already in hand - and it is the only way to
        # know which of the two to blame when test disappoints.
        all_comp = load_components(args.run, "val")
        n_val_cases = len({r["case_id"] for r in all_comp})
        insample = posterior(model, comp)
        optimism = {}
        for label, p in (("in_sample", insample), ("out_of_fold", oof)):
            full = np.zeros(len(all_comp))
            pos = {(r["case_id"], r["class_name"], r["comp_id"]): i for i, r in enumerate(comp)}
            for i, r in enumerate(all_comp):
                k = (r["case_id"], r["class_name"], r["comp_id"])
                full[i] = p[pos[k]] if k in pos else 1.0  # below the floor: never dropped
            c = detection_curve(all_comp, refs, full, n_val_cases, np.array([0.0, p_star]))
            optimism[label] = {"sensitivity": c["sensitivity"][1],
                               "fp_per_case": c["fp_per_case"][1],
                               "n_fp": c["n_fp"][1], "n_detected": c["n_detected"][1]}
            if label == "in_sample":
                optimism["before"] = {"sensitivity": c["sensitivity"][0],
                                      "fp_per_case": c["fp_per_case"][0],
                                      "n_fp": c["n_fp"][0], "n_detected": c["n_detected"][0]}

        payload = {"run": args.run, "threshold": derived, "p_star": p_star,
                   "model": model, "calibration": cal,
                   "validation_detection": optimism,
                   "coefficients": dict(zip(model["columns"], [round(c, 4) for c in model["coef"]])),
                   "intercept": round(model["intercept"], 4)}
        path = paths.REPORTS / f"selector_{args.run}.json"
        if path.exists() and "applied" in json.loads(path.read_text(encoding="utf-8")):
            # Refitting invalidates any held-out score already recorded here, so
            # the record is dropped rather than left to look like it describes
            # the new model. --apply then has to be run again, deliberately.
            print("note: refitting; the previously recorded --apply results are discarded")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(f"L = {derived['L']}  d = {derived['d']}  ->  p* = {p_star}")
        print(f"  (d over all matched components would be {derived['d_all_matched']}, "
              f"giving p* = {derived['p_star_if_d_all_matched']})")
        print(f"\nout-of-fold AUC: raw mean_prob {cal['auc_raw_mean_prob']}, "
              f"fitted posterior {cal['auc_fitted_posterior']}")
        print(f"expected calibration error: raw {cal['ece_raw_mean_prob']}, "
              f"fitted {cal['ece_fitted_posterior']}")
        b = optimism["before"]
        print(f"\nvalidation detection at p*  (sensitivity / FP per case):")
        print(f"  no filtering   {b['sensitivity']:.4f} / {b['fp_per_case']:.3f}")
        for label in ("in_sample", "out_of_fold"):
            o = optimism[label]
            print(f"  {label:<14} {o['sensitivity']:.4f} / {o['fp_per_case']:.3f}")
        print("\nstandardised coefficients (logit units per SD):")
        for k, v in sorted(payload["coefficients"].items(), key=lambda kv: -abs(kv[1])):
            print(f"  {k:<22} {v:+.3f}")
        print(f"\nwrote {path}")

    if args.apply:
        _apply(args)


def _apply(args) -> None:
    """Score a held-out split with the fitted selector. Once.

    The guard below is not ceremony. The previous post-processing experiment
    found validation overstating the test gain threefold, which is exactly the
    situation in which a second test run "with slightly different settings"
    turns a held-out number into a fitted one. Re-running requires ``--force``
    and the reason belongs in the write-up.
    """
    from .postprocess import per_class_thresholds

    payload = load_selector(args.run)
    if args.split in payload.get("applied", {}) and not args.force:
        raise SystemExit(
            f"{args.split} has already been scored with this selector "
            f"(see reports/selector_{args.run}.json). Applying twice and keeping the "
            f"better number is how a held-out split stops being held out. Use --force "
            f"only if you intend to say so in the report.")

    split = args.split
    class_values = paths.GLI_LABELS if args.dataset == "gli" else paths.MEN_LABELS
    comp_all = load_components(args.run, split)
    refs = load_csv(ref_lesions_path(args.run, split))
    probs = posterior(payload["model"], comp_all)
    p_star = payload["p_star"]
    keep = keep_table(payload["model"], comp_all, p_star)
    sizes = {(c["case_id"], c["class_name"], c["comp_id"]): c["n_voxels"] for c in comp_all}

    thresholds = per_class_thresholds(args.run, class_values)
    print(f"selector p* = {p_star};  size baseline = "
          f"{ {class_values[k]: v for k, v in thresholds.items()} }\n")

    variants = compare_variants(args.run, args.dataset, split, thresholds, keep, sizes,
                                args.limit)
    regions = [k[len("lesion_dice_"):] for k in variants["none"][0]
               if k.startswith("lesion_dice_")]
    from .postprocess import summarise
    labels = {"none": "none", "size": thresholds, "selector": f"selector p*={p_star}"}
    summaries = {name: summarise(rows, labels[name]) for name, rows in variants.items()}
    n_cases = len(variants["none"])

    print(f"{'region':<6} {'none':>8} {'size':>8} {'selector':>10}   "
          f"{'volumetric none':>16} {'selector':>9}   {'FP none':>8} {'size':>6} {'sel':>6}   "
          f"{'missed none':>12} {'size':>6} {'sel':>6}")
    for r in regions:
        n, s, v = (summaries["none"][r], summaries["size"][r], summaries["selector"][r])
        print(f"{r:<6} {n['lesion']:>8.4f} {s['lesion']:>8.4f} {v['lesion']:>10.4f}   "
              f"{n['volumetric']:>16.4f} {v['volumetric']:>9.4f}   "
              f"{n['false_lesions']:>8} {s['false_lesions']:>6} {v['false_lesions']:>6}   "
              f"{n['missed_lesions']:>12} {s['missed_lesions']:>6} {v['missed_lesions']:>6}")
    print(f"{'mean':<6} {summaries['none']['mean_lesion']:>8.4f} "
          f"{summaries['size']['mean_lesion']:>8.4f} "
          f"{summaries['selector']['mean_lesion']:>10.4f}   "
          f"(selector {summaries['selector']['mean_lesion'] - summaries['none']['mean_lesion']:+.4f}, "
          f"size {summaries['size']['mean_lesion'] - summaries['none']['mean_lesion']:+.4f})")

    detect = detection_summary(comp_all, refs, probs, p_star, n_cases)
    print(f"\nper-lesion sensitivity {detect['before']['sensitivity']:.4f} -> "
          f"{detect['after']['sensitivity']:.4f};  false positives per case "
          f"{detect['before']['fp_per_case']:.3f} -> {detect['after']['fp_per_case']:.3f}")

    trade = multifocal_tradeoff(refs, comp_all, probs, p_star,
                                variants["none"], variants["selector"], regions)
    print("\nmultifocal cost:")
    fmt = lambda x: "    n/a" if x is None else f"{x:+.4f}"
    for D, t in trade["by_distance"].items():
        print(f"  beyond {D:>3} mm: {t['n_lesions_beyond']:>4} scored lesions in "
              f"{t['n_cases_with_one']:>3} cases; selector deletes "
              f"{t['n_detected_lesions_deleted']:>3} detections in {t['n_cases_hurt']:>3} cases "
              f"(those cases {fmt(t['delta_mean_lesion_dice_on_those_cases'])}, "
              f"the rest {fmt(t['delta_mean_lesion_dice_on_the_rest'])})")

    # FROC over both splits: validation is the curve the rule was chosen on,
    # test is the curve it actually landed on, and the gap between them is the
    # haircut this project keeps finding.
    grid = np.linspace(0.0, 1.0, 101)
    curves = {}
    for s in ("val", split) if split != "val" else ("val",):
        if not components_path(args.run, s).exists():
            continue
        cr = load_components(args.run, s)
        rr = load_csv(ref_lesions_path(args.run, s))
        n = len({c["case_id"] for c in cr})
        curves[s] = detection_curve(cr, rr, posterior(payload["model"], cr), n, grid)
    froc = froc_plot(curves, p_star, paths.REPORTS / f"froc_{args.run}.png")

    payload.setdefault("applied", {})[split] = {
        "n_cases": n_cases,
        "p_star": p_star,
        "size_baseline_thresholds": {class_values[k]: v for k, v in thresholds.items()},
        "scores": summaries,
        "detection": detect,
        "multifocal_tradeoff": trade,
    }
    payload["froc"] = {s: c for s, c in curves.items()}
    out = paths.REPORTS / f"selector_{args.run}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out} and {froc}")


if __name__ == "__main__":
    main()
