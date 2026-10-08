"""The relabeller: features are recorded right, and a rename moves exactly one component."""

from __future__ import annotations

import numpy as np

from mri3d import relabel
from mri3d.predict import component_rows
from mri3d.select import design_matrix

CLASSES = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}


def _scene():
    """Two NETC components: one confident, one nearly tied with SNFH."""
    shape = (24, 24, 24)
    probs = np.zeros((5, *shape), np.float32)
    probs[0] = 1.0
    pred = np.zeros(shape, np.uint8)
    # A: confident NETC (0.9 vs SNFH 0.05); reference agrees it is NETC
    a = (slice(2, 8), slice(2, 8), slice(2, 8))
    probs[:, a[0], a[1], a[2]] = 0
    probs[1][a], probs[2][a], probs[0][a] = 0.9, 0.05, 0.05
    pred[a] = 1
    # B: NETC by a hair (0.48 vs SNFH 0.46); reference says SNFH
    b = (slice(14, 20), slice(14, 20), slice(14, 20))
    probs[:, b[0], b[1], b[2]] = 0
    probs[1][b], probs[2][b], probs[0][b] = 0.48, 0.46, 0.06
    pred[b] = 1
    ref = np.zeros(shape, np.uint8)
    ref[a] = 1
    ref[b] = 2
    return pred, probs, ref, a, b


def test_component_rows_record_runner_up_and_reference_majority():
    pred, probs, ref, a, b = _scene()
    rows, _ = component_rows(pred, probs, ref, np.ones(3), CLASSES, None, {"case_id": "X-1"})
    by_size = {round(r["prob_c1"], 2): r for r in rows}
    ra, rb = by_size[0.9], by_size[0.48]
    assert ra["runner_up_value"] == 2 and rb["runner_up_value"] == 2
    assert abs(rb["runner_up_prob"] - 0.46) < 1e-5
    assert ra["ref_majority_value"] == 1          # reference agrees with NETC
    assert rb["ref_majority_value"] == 2          # reference says SNFH: rename target
    assert rb["ref_tumour_frac"] == 1.0


def _payload_renaming_small_margins():
    """A hand-built model: posterior is high exactly when the margin is small."""
    pred, probs, ref, *_ = _scene()
    rows, _ = component_rows(pred, probs, ref, np.ones(3), CLASSES, None, {"case_id": "X-1"})
    rows = relabel.eligible(rows)
    names = relabel.augment(rows, [2])
    classes = ["NETC"]
    X, cols, impute = design_matrix(rows, names, classes)
    coef = np.zeros(len(cols))
    coef[cols.index("margin")] = -100.0
    model = {"features": names, "classes": classes, "columns": cols, "impute": impute,
             "mean": [0.0] * len(cols), "std": [1.0] * len(cols),
             "coef": coef.tolist(), "intercept": 5.0}
    return {"threshold": 0.5, "ru_values": [2], "model": model, "folds": []}


def test_apply_renames_only_the_near_tie_and_uses_no_reference():
    pred, probs, ref, a, b = _scene()
    out, n = relabel.apply(pred, probs, np.ones(3), CLASSES, _payload_renaming_small_margins(), "X-1")
    assert n == 1
    assert (out[b] == 2).all()        # renamed to the runner-up
    assert (out[a] == 1).all()        # the confident one is untouched
    assert (out[(pred == 0)] == 0).all()


def test_fit_learns_the_runner_up_signal_out_of_fold():
    """Synthetic validation rows: small margin <=> runner-up is the right name."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        good = i % 2 == 0
        pa = rng.uniform(0.5, 0.9)
        pr = pa - (rng.uniform(0.0, 0.05) if good else rng.uniform(0.3, 0.45))
        rows.append({"case_id": f"BraTS-GLI-{i // 4:05d}-{i % 4:03d}", "class_name": "NETC",
                     "class_value": 1, "comp_id": i, "n_voxels": 500, "mean_prob": pa,
                     "runner_up_prob": max(pr, 0.01), "runner_up_value": 2, "prob_c0": 0.05,
                     "dist_to_dominant_mm": 5.0, "matched": not good,
                     "ref_majority_value": 2 if good else 1})
    payload = relabel.fit(rows)
    d = payload["diagnostics"]
    assert d["out_of_fold_auc"] > 0.95
    assert d["out_of_fold_rename_precision"] > 0.9
    assert len(payload["folds"]) == relabel.N_FOLDS
    # Folds are by patient: no patient appears in two folds
    seen = [p for f in payload["folds"] for p in f["patients"]]
    assert len(seen) == len(set(seen))
