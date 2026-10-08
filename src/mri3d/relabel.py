"""Rename predicted components the model found but named wrong.

``mri3d.failure_anatomy`` showed that half of the glioma false positives are not
invented at all: 229 of 462 sit mostly on real tumour and carry the wrong
sub-region name, and lesion-wise Dice charges for each of them twice. An oracle
that renames them is worth +0.0611 - more than the selector's +0.044 - and it is
out of the selector's reach, because the selector can only delete.

The evidence for the right name is in the network already: the runner-up class
in the softmax. This module learns, on validation, when the runner-up should
win:

    y = 1  if the reference's majority tumour class under the component is the
           runner-up class (the rename would be right)
    y = 0  otherwise (the assigned class is right, or the tissue is background)

and renames a component when that posterior clears 0.5. The cut is not swept.
A wrong rename of a correct component costs about what a right rename of a
mislabelled one gains - one lesion moves between "matched" and "false
positive + missed" either way - and renaming an *invented* lesion only moves a
false positive between classes. So 0.5 is the break-even for the first case and
conservative for the second.

The fit is logistic regression with the selector's own design matrix
(``select.design_matrix``: median imputation and standardisation stored from
validation, class as an indicator) so it reads the same way. Five fold models,
grouped by *patient* so two timepoints of one person never sit on both sides,
let the validation split be relabelled out-of-fold: every validation case is
renamed by a model that never saw that patient, which is what makes the
validation score an honest basis for deciding whether to use this at all.

    python -m mri3d.relabel --run gli_rl_base --out reports/relabel_gli_main.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import paths
from .select import auc, components_path, decidable, design_matrix, load_csv

THRESHOLD = 0.5
N_FOLDS = 5

BASE_FEATURES = ["log_n_voxels", "mean_prob", "runner_up_prob", "margin",
                 "log_ratio", "prob_c0", "log1p_dist"]


def _patient(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def eligible(rows: list[dict]) -> list[dict]:
    """Components a rename can be decided for: above the metric's floor, with a runner-up."""
    return [r for r in decidable(rows) if r.get("runner_up_value") is not None]


def augment(rows: list[dict], ru_values: list[int]) -> list[str]:
    """Add the derived columns in place; return the feature names to fit on."""
    for r in rows:
        pa, pr = float(r["mean_prob"]), float(r["runner_up_prob"])
        r["margin"] = pa - pr
        r["log_ratio"] = math.log((pr + 1e-6) / (pa + 1e-6))
        for v in ru_values:
            r[f"ru_is_{v}"] = float(int(r["runner_up_value"]) == v)
    # The first runner-up class is the reference level, as with the class term.
    return BASE_FEATURES + [f"ru_is_{v}" for v in ru_values[1:]]


def target(rows: list[dict]) -> np.ndarray:
    return np.array([int(r["ref_majority_value"]) == int(r["runner_up_value"])
                     for r in rows], dtype=int)


def _fit(rows: list[dict], names: list[str], classes: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression

    X, cols, impute = design_matrix(rows, names, classes)
    y = target(rows)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    if y.min() == y.max():
        # One outcome only (a small fold can have no renameable component):
        # no signal to learn, so predict the base rate - which, at 0 or 1, is
        # also the honest answer - rather than fail.
        rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        coef, intercept = [0.0] * len(cols), math.log(rate / (1 - rate))
    else:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X - mean) / std, y)
        coef, intercept = clf.coef_[0].tolist(), float(clf.intercept_[0])
    return {"features": names, "classes": classes, "columns": cols, "impute": impute,
            "mean": mean.tolist(), "std": std.tolist(),
            "coef": coef, "intercept": intercept,
            "n_train": int(len(rows)), "n_positive": int(y.sum())}


def posterior(model: dict, rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.zeros(0)
    X, _, _ = design_matrix(rows, model["features"], model["classes"], model["impute"])
    z = ((X - np.array(model["mean"])) / np.array(model["std"])) @ np.array(model["coef"])
    return 1.0 / (1.0 + np.exp(-(z + model["intercept"])))


def fit(rows: list[dict], seed: int = 0) -> dict:
    rows = eligible(rows)
    ru_values = sorted({int(r["runner_up_value"]) for r in rows})
    names = augment(rows, ru_values)
    classes = sorted({r["class_name"] for r in rows})
    full = _fit(rows, names, classes)

    patients = sorted({_patient(r["case_id"]) for r in rows})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    fold_of = {patients[i]: k % N_FOLDS for k, i in enumerate(order)}
    oof = np.zeros(len(rows))
    folds = []
    for k in range(N_FOLDS):
        train = [r for r in rows if fold_of[_patient(r["case_id"])] != k]
        held = [i for i, r in enumerate(rows) if fold_of[_patient(r["case_id"])] == k]
        m = _fit(train, names, classes)
        oof[held] = posterior(m, [rows[i] for i in held])
        folds.append({"patients": sorted(p for p, f in fold_of.items() if f == k), "model": m})

    y = target(rows)
    rename = oof > THRESHOLD
    matched = np.array([bool(r["matched"]) for r in rows])
    diag = {
        "n_eligible": int(len(rows)),
        "n_runner_up_correct": int(y.sum()),
        # select.auc indexes with the labels, so they must be boolean: an int
        # array would be read as row positions and give a meaningless number.
        "out_of_fold_auc": round(auc(oof, y.astype(bool)), 4),
        "out_of_fold_renames": int(rename.sum()),
        "out_of_fold_rename_precision": round(float(y[rename].mean()), 4) if rename.any() else None,
        # Renaming a component that is currently matched is the expensive mistake.
        "out_of_fold_renames_of_matched": int((rename & matched).sum()),
        "renameable_false_positives_caught": int((rename & y.astype(bool) & ~matched).sum()),
    }
    coefs = dict(zip(full["columns"], [round(c, 4) for c in full["coef"]]))
    return {"threshold": THRESHOLD, "ru_values": ru_values, "model": full,
            "folds": folds, "diagnostics": diag, "coefficients": coefs}


def _model_for(payload: dict, case_id: str | None, out_of_fold: bool) -> dict:
    if not out_of_fold:
        return payload["model"]
    patient = _patient(case_id)
    for f in payload["folds"]:
        if patient in f["patients"]:
            return f["model"]
    # A patient the fit never saw at all (not a validation case): the full
    # model is already out-of-sample for it.
    return payload["model"]


def apply(pred: np.ndarray, probs: np.ndarray, spacing: np.ndarray,
          class_values: dict[int, str], payload: dict,
          case_id: str | None = None, out_of_fold: bool = False) -> tuple[np.ndarray, int]:
    """Rename components whose runner-up posterior clears the threshold.

    Features come from ``predict.component_rows`` - the function that wrote the
    CSV the model was fitted on - with an empty reference, so nothing about the
    answer can leak into the decision. All decisions are made on the mask as
    predicted and applied together, so the order classes are visited in cannot
    matter.
    """
    from .metrics import match_components
    from .predict import component_rows

    empty = np.zeros_like(pred)
    rows, _ = component_rows(pred, probs, empty, spacing, class_values, None, {"case_id": case_id})
    rows = eligible(rows)
    if not rows:
        return pred, 0
    augment(rows, payload["ru_values"])
    p = posterior(_model_for(payload, case_id, out_of_fold), rows)
    out = pred.copy()
    labels = {}
    n = 0
    for r, pr in zip(rows, p):
        if pr <= payload["threshold"]:
            continue
        v = int(r["class_value"])
        if v not in labels:
            labels[v] = match_components(pred == v, empty == 1).pred_lab
        out[labels[v] == int(r["comp_id"])] = int(r["runner_up_value"])
        n += 1
    return out, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run whose val components CSV to fit on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_csv(components_path(args.run, "val"))
    if "runner_up_value" not in rows[0]:
        raise SystemExit("components CSV has no runner-up columns; re-run mri3d.predict "
                         "with --save-components on the current code")
    payload = fit(rows, args.seed)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    d = payload["diagnostics"]
    print(f"eligible components {d['n_eligible']}; runner-up is the right name for "
          f"{d['n_runner_up_correct']}")
    print(f"out-of-fold AUC {d['out_of_fold_auc']}; at p > {THRESHOLD}: "
          f"{d['out_of_fold_renames']} renames, precision {d['out_of_fold_rename_precision']}, "
          f"{d['out_of_fold_renames_of_matched']} of them currently matched (the costly kind), "
          f"{d['renameable_false_positives_caught']} mislabelled false positives fixed")
    print("standardised coefficients:")
    for k, v in sorted(payload["coefficients"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<16} {v:+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
