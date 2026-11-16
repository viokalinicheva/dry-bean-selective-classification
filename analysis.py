"""Dry bean classification: a frozen, group-separated referral benchmark.

Run from this directory: python analysis.py. Raw inputs are never overwritten.
Every test result is derived from the single cached prediction table.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import platform
import shutil
import warnings
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ARCHIVE_SHA = "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
ARFF_SHA = "b2a4a76a2aedfb8ed415adfc1bfc70b5f202cb00cb72e500766e364c14834014"
LABELS = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
EXPECTED_COUNTS = [1322, 522, 1630, 3546, 1928, 2027, 2636]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def save_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def read_predictions(path):
    # Round-trip parsing preserves scores exactly at the frozen threshold.
    return pd.read_csv(path, float_precision="round_trip")


def threshold_curve(scores, correct):
    """Exhaustive, whole-tie candidate scan; no monotonicity assumption."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(scores) == 0 or len(scores) != len(correct):
        raise ValueError("Nonempty aligned validation arrays are required")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Scores must be finite and between zero and one")
    records = []
    for cutoff in np.unique(np.r_[0.0, scores]):
        accepted = scores >= cutoff
        count = int(accepted.sum())
        errors = int((accepted & ~correct).sum())
        records.append({"threshold": float(cutoff), "accepted": count,
                        "errors": errors, "coverage": count / len(scores),
                        "accepted_error": errors / count if count else None,
                        "feasible": bool(count > 0 and 20 * errors <= count)})
    return pd.DataFrame(records)


def select_threshold(scores, correct):
    curve = threshold_curve(scores, correct)
    feasible = curve.loc[curve.feasible].sort_values(
        ["accepted", "threshold"], ascending=[False, True])
    if feasible.empty:
        return {"status": "refer_all", "threshold": None,
                "accepted": 0, "errors": 0, "coverage": 0.0,
                "accepted_error": None}, curve
    row = feasible.iloc[0]
    return {"status": "threshold", "threshold": float(row.threshold),
            "accepted": int(row.accepted), "errors": int(row.errors),
            "coverage": float(row.coverage),
            "accepted_error": float(row.accepted_error)}, curve


def accepted_mask(scores, policy):
    if policy["status"] == "refer_all":
        return np.zeros(len(scores), dtype=bool)
    return np.asarray(scores) >= policy["threshold"]


def audit_and_split():
    # I verify the preserved bytes before parsing the feature table.
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = (ROOT / "data/raw/dry-bean-dataset.zip").read_bytes()
    assert sha256(raw) == ARCHIVE_SHA, "Raw archive changed"
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".arff")]
        assert len(names) == 1
        arff = archive.read(names[0])
    assert sha256(arff) == ARFF_SHA, "ARFF source changed"
    text = arff.decode("utf-8")
    columns = [line.split()[1] for line in text.splitlines()
               if line.upper().startswith("@ATTRIBUTE")]
    frame = pd.read_csv(io.StringIO(text.split("@DATA", 1)[1]), names=columns)
    X, y = frame.drop(columns="Class"), frame["Class"]
    assert X.shape == (13611, 16)
    assert y.value_counts().reindex(LABELS).tolist() == EXPECTED_COUNTS
    assert frame.isna().sum().sum() == 0 and np.isfinite(X).all().all()
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X)
    assert not (X.nunique() <= 1).any()
    groups = X.groupby(list(X.columns), sort=False, dropna=False).ngroup()
    assert groups.nunique() == 13543
    assert X.duplicated().sum() == 68 and X.duplicated(keep=False).sum() == 136
    assert frame.loc[X.duplicated(), "Class"].eq("HOROZ").all()
    assert frame.assign(group=groups).groupby("group").Class.nunique().max() == 1
    assert (X > 0).all().all()
    assert (X.MinorAxisLength <= X.MajorAxisLength).all()
    assert (X.Area <= X.ConvexArea).all()
    for column in ["Eccentricity", "Extent", "Solidity", "roundness", "Compactness"]:
        assert X[column].between(0, 1).all()
    equations = {
        "AspectRation": X.MajorAxisLength / X.MinorAxisLength,
        "EquivDiameter": np.sqrt(4 * X.Area / np.pi),
        "Solidity": X.Area / X.ConvexArea,
        "roundness": 4 * np.pi * X.Area / X.Perimeter ** 2,
        "Compactness": X.EquivDiameter / X.MajorAxisLength,
    }
    equation_errors = {k: float(np.max(np.abs(X[k] - v))) for k, v in equations.items()}
    for column, expected in equations.items():
        assert np.allclose(X[column], expected, atol=1e-6, rtol=1e-6)
    # I reproduce the prespecified group ordering and split algorithm.
    folds = np.full(len(frame), -1, dtype=int)
    rng = np.random.default_rng(680)
    for label in sorted(y.unique()):
        class_groups = groups[y == label].unique()
        rng.shuffle(class_groups)
        for fold_id, heldout in enumerate(np.array_split(class_groups, 5)):
            folds[groups.isin(heldout)] = fold_id
    assert (folds >= 0).all()
    partition = np.where(folds == 0, "test", np.where(folds == 1, "validation", "train"))
    manifest = pd.DataFrame({"row_id": np.arange(len(frame)), "feature_group": groups,
                             "outer_fold": folds, "partition": partition, "actual": y})
    assert manifest.groupby("feature_group").partition.nunique().max() == 1
    split_rows = []
    for name, expected_rows, expected_groups in [("train", 8163, 8121),
                                                ("validation", 2725, 2710),
                                                ("test", 2723, 2712)]:
        mask = partition == name
        assert mask.sum() == expected_rows and groups[mask].nunique() == expected_groups
        class_counts = y[mask].value_counts().reindex(LABELS).astype(int).to_dict()
        assert class_counts["BOMBAY"] == (312 if name == "train" else 105)
        split_rows.append({"partition": name, "rows": int(mask.sum()),
                           "groups": int(groups[mask].nunique()), "class_counts": class_counts})
    train_ids = np.flatnonzero(partition == "train")
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=68002).split(
        X.iloc[train_ids], y.iloc[train_ids], groups.iloc[train_ids]))
    heldout_counts = np.zeros(len(train_ids), dtype=int)
    cv_records = []
    for fold_id, (fit_pos, hold_pos) in enumerate(cv):
        fit_ids, hold_ids = train_ids[fit_pos], train_ids[hold_pos]
        assert not set(groups.iloc[fit_ids]) & set(groups.iloc[hold_ids])
        assert set(y.iloc[fit_ids]) == set(LABELS) == set(y.iloc[hold_ids])
        heldout_counts[hold_pos] += 1
        cv_records.extend({"row_id": int(i), "cv_heldout_fold": fold_id}
                          for i in hold_ids)
    assert np.all(heldout_counts == 1)
    manifest.to_csv(RESULTS / "split_manifest.csv", index=False)
    pd.DataFrame(cv_records).sort_values("row_id").to_csv(RESULTS / "cv_manifest.csv", index=False)
    audit = {"archive_sha256": ARCHIVE_SHA, "arff_sha256": ARFF_SHA,
             "rows": len(frame), "predictors": list(X.columns),
             "class_counts": y.value_counts().reindex(LABELS).astype(int).to_dict(),
             "missing_cells": 0, "nonfinite_cells": 0, "constant_features": [],
             "extra_exact_vectors": 68, "duplicate_rows_involved": 136,
             "unique_vectors": 13543, "duplicate_extra_class": "HOROZ",
             "conflicting_group_labels": 0, "formula_max_absolute_errors": equation_errors,
             "splits": split_rows,
             "transformations": ["No joins, imputations, exclusions, or feature renames",
                                 "Exact-vector grouping for separation, not row deletion",
                                 "StandardScaler fitted within training folds for logistic regression only"]}
    save_json("audit.json", audit)
    save_json("environment.json", {"python": platform.python_version(),
              "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
              "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
              "joblib": version("joblib"), "threadpoolctl": version("threadpoolctl"),
              "nbformat": version("nbformat"), "nbclient": version("nbclient"),
              "ipykernel": version("ipykernel")})
    print(f"Audit passed: {len(frame):,} rows; {groups.nunique():,} vectors; no split overlap.", flush=True)
    return {"X": X, "y": y, "groups": groups, "manifest": manifest,
            "train_ids": train_ids, "cv": cv, "audit": audit}


def candidate_specs():
    specs = [{"id": f"LR_C{c:g}", "kind": "logistic", "C": c,
              "tie_rank": [0, c, 0]} for c in [0.1, 1.0, 10.0, 100.0]]
    specs += [{"id": f"RF_d{d}_leaf{leaf}", "kind": "random_forest",
               "max_depth": d, "min_samples_leaf": leaf, "tie_rank": [1, d, -leaf]}
              for d, leaf in itertools.product([8, 12], [2, 5])]
    return specs


def make_model(spec):
    if spec["kind"] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=spec["C"], solver="lbfgs", max_iter=2000, tol=1e-4, class_weight=None))
    if spec["kind"] == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"], max_features="sqrt", bootstrap=True,
            class_weight=None, random_state=68002, n_jobs=1)
    return DummyClassifier(strategy="most_frequent")


def fit_checked(model, X, y):
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X, y)
    if hasattr(model, "named_steps"):
        scaler = model.named_steps["standardscaler"]
        assert scaler.n_samples_seen_ == len(X)
        assert np.allclose(scaler.mean_, X.mean(axis=0).to_numpy())
    return model


def compare_models(state):
    if (RESULTS / "freeze.json").exists():
        freeze = load_json("freeze.json")
        assert freeze["protocol_sha256"] == sha256((ROOT / "protocol.json").read_bytes())
        state["selected"] = load_json("model_selection.json")["selected"]
        print("Reusing prespecified training-CV results; no new model search.", flush=True)
        return pd.read_csv(RESULTS / "cv_summary.csv", float_precision="round_trip")
    X, y = state["X"], state["y"]
    records = []
    specs = candidate_specs() + [{"id": "Majority", "kind": "baseline", "tie_rank": [2, 0, 0]}]
    for spec in specs:
        for fold_id, (fit_pos, hold_pos) in enumerate(state["cv"]):
            fit_ids, hold_ids = state["train_ids"][fit_pos], state["train_ids"][hold_pos]
            model = fit_checked(make_model(spec), X.iloc[fit_ids], y.iloc[fit_ids])
            with threadpool_limits(limits=1):
                pred = model.predict(X.iloc[hold_ids])
            records.append({"candidate": spec["id"], "fold": fold_id,
                            "fit_rows": len(fit_ids), "heldout_rows": len(hold_ids),
                            "macro_f1": float(f1_score(y.iloc[hold_ids], pred, labels=LABELS,
                                                       average="macro", zero_division=0)),
                            "accuracy": float(accuracy_score(y.iloc[hold_ids], pred))})
        print(f"CV finished: {spec['id']}", flush=True)
    folds = pd.DataFrame(records)
    summary = folds.groupby("candidate", sort=False).agg(
        mean_macro_f1=("macro_f1", "mean"), sd_macro_f1=("macro_f1", "std"),
        mean_accuracy=("accuracy", "mean")).reset_index()
    eligible = summary[summary.candidate != "Majority"]
    maximum = eligible.mean_macro_f1.max()
    tied = set(eligible.loc[maximum - eligible.mean_macro_f1 <= 1e-12, "candidate"])
    selected = min((s for s in specs if s["id"] in tied), key=lambda s: tuple(s["tie_rank"]))
    folds.to_csv(RESULTS / "cv_folds.csv", index=False)
    summary.to_csv(RESULTS / "cv_summary.csv", index=False)
    save_json("model_selection.json", {"selected": selected, "best_mean_macro_f1": float(maximum),
              "eligible_candidates": 8, "cv_fits_including_baseline": 45,
              "selected_using": "Training-only five-fold grouped macro-F1"})
    state["selected"] = selected
    print(summary.to_string(index=False), flush=True)
    return summary


def prediction_table(state, ids, model, baseline=None):
    """Predict each exact vector once, expanding to original row weights."""
    X, groups = state["X"], state["groups"]
    subgroups = groups.iloc[ids]
    rep_mask = ~subgroups.duplicated()
    rep_ids = ids[rep_mask.to_numpy()]
    rep_groups = groups.iloc[rep_ids].to_numpy()
    with threadpool_limits(limits=1):
        proba = model.predict_proba(X.iloc[rep_ids])
        direct_pred = model.predict(X.iloc[rep_ids])
    classes = np.asarray(model.classes_)
    assert list(classes) == LABELS
    assert np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    pred = classes[np.argmax(proba, axis=1)]
    assert np.array_equal(pred, direct_pred)
    positions = pd.Series(np.arange(len(rep_ids)), index=rep_groups).loc[subgroups].to_numpy()
    expanded = proba[positions]
    table = state["manifest"].iloc[ids].copy().reset_index(drop=True)
    table["predicted"] = pred[positions]
    table["score"] = expanded.max(axis=1)
    table["correct"] = table.actual.eq(table.predicted)
    for i, label in enumerate(classes):
        table[f"score_{label}"] = expanded[:, i]
    if baseline is not None:
        with threadpool_limits(limits=1):
            table["baseline_predicted"] = baseline.predict(X.iloc[ids])
    return table


def freeze_policy(state):
    if (RESULTS / "freeze.json").exists():
        freeze = load_json("freeze.json")
        assert freeze["protocol_sha256"] == sha256((ROOT / "protocol.json").read_bytes())
        assert freeze["selected"] == state["selected"]
        assert freeze["model_sha256"] == sha256((RESULTS / "frozen_model.joblib").read_bytes())
        frozen_models = joblib.load(RESULTS / "frozen_model.joblib")
        state.update(model=frozen_models["selected"], baseline=frozen_models["baseline"], freeze=freeze)
        print("Reusing original frozen model and policy; no refit or threshold reselection.", flush=True)
        return freeze
    # I fit on training only, then select the score cutoff using validation only.
    ids = state["train_ids"]
    model = fit_checked(make_model(state["selected"]), state["X"].iloc[ids], state["y"].iloc[ids])
    baseline = fit_checked(DummyClassifier(strategy="most_frequent"),
                           state["X"].iloc[ids], state["y"].iloc[ids])
    val_ids = np.flatnonzero(state["manifest"].partition.eq("validation"))
    val = prediction_table(state, val_ids, model)
    policy, curve = select_threshold(val.score, val.correct)
    val["accepted"] = accepted_mask(val.score, policy)
    val.to_csv(RESULTS / "validation_predictions.csv", index=False, float_format="%.17g")
    curve.to_csv(RESULTS / "validation_risk_coverage.csv", index=False, float_format="%.17g")
    model_path = RESULTS / "frozen_model.joblib"
    joblib.dump({"selected": model, "baseline": baseline}, model_path)
    freeze = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "selected": state["selected"], "training_rows": len(ids),
              "validation_rows": len(val), "policy": policy,
              "model_sha256": sha256(model_path.read_bytes()),
              "protocol_sha256": sha256((ROOT / "protocol.json").read_bytes()),
              "test_predictions_created": False,
              "note": "Frozen before first test prediction. No train-plus-validation refit."}
    save_json("freeze.json", freeze)
    state.update(model=model, baseline=baseline, freeze=freeze)
    print("Frozen policy:", json.dumps(policy), flush=True)
    return freeze


def evaluate_test(state):
    freeze = load_json("freeze.json")
    assert freeze["test_predictions_created"] is False
    assert sha256((RESULTS / "frozen_model.joblib").read_bytes()) == freeze["model_sha256"]
    cache_path = RESULTS / "test_predictions.csv"
    if cache_path.exists():
        # Exact re-execution reuses the cache; incompatible reruns must be explicit.
        receipt = load_json("test_receipt.json")
        assert receipt["model_sha256"] == freeze["model_sha256"], "Existing cache belongs to another fit"
        assert receipt["cache_sha256"] == sha256(cache_path.read_bytes())
        assert receipt["freeze_sha256"] == sha256((RESULTS / "freeze.json").read_bytes())
        table = read_predictions(cache_path)
    else:
        test_ids = np.flatnonzero(state["manifest"].partition.eq("test"))
        table = prediction_table(state, test_ids, state["model"], state["baseline"])
        table["accepted"] = accepted_mask(table.score, freeze["policy"])
        table.to_csv(cache_path, index=False, float_format="%.17g")
        save_json("test_receipt.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "freeze_sha256": sha256((RESULTS / "freeze.json").read_bytes()),
                  "model_sha256": freeze["model_sha256"],
                  "cache_sha256": sha256(cache_path.read_bytes()),
                  "rows": len(table), "prediction_batch": "Selected model and training-majority baseline only"})
    assert len(table) == 2723 and table.feature_group.nunique() == 2712
    assert np.array_equal(table.accepted, accepted_mask(table.score, freeze["policy"]))
    assert table.groupby("feature_group").predicted.nunique().max() == 1
    print(f"Held-out cache verified: {len(table):,} rows. All further test analysis uses this file.", flush=True)
    return table


def metric_from_cm(cm):
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    true, pred = cm.sum(axis=1), cm.sum(axis=0)
    f1 = np.divide(2 * tp, true + pred, out=np.zeros(7), where=(true + pred) != 0)
    recall = np.divide(tp, true, out=np.zeros(7), where=true != 0)
    return {"accuracy": float(tp.sum() / total), "macro_f1": float(f1.mean()),
            "macro_recall": float(recall.mean())}


def compute_metrics(table):
    cm = confusion_matrix(table.actual, table.predicted, labels=LABELS)
    baseline_cm = confusion_matrix(table.actual, table.baseline_predicted, labels=LABELS)
    accepted = table.accepted.to_numpy(dtype=bool)
    errors = ~table.actual.eq(table.predicted).to_numpy()
    n, accepted_n = len(table), int(accepted.sum())
    accepted_errors = int((accepted & errors).sum())
    assert cm.sum() == n
    result = {"rows": n, "groups": int(table.feature_group.nunique()),
              "full_coverage": metric_from_cm(cm), "baseline": metric_from_cm(baseline_cm),
              "accepted": accepted_n, "referred": n - accepted_n,
              "accepted_errors": accepted_errors, "total_errors": int(errors.sum()),
              "referred_errors": int((~accepted & errors).sum()),
              "coverage": accepted_n / n,
              "accepted_error": accepted_errors / accepted_n if accepted_n else None,
              "observed_target_met": bool(accepted_n > 0 and 20 * accepted_errors <= accepted_n)}
    assert accepted_errors <= accepted_n
    assert result["accepted_errors"] + result["referred_errors"] == result["total_errors"]
    return result, cm


def summarize_results():
    # I deliberately reload the cache rather than predict again.
    receipt = load_json("test_receipt.json")
    assert receipt["cache_sha256"] == sha256((RESULTS / "test_predictions.csv").read_bytes())
    table = read_predictions(RESULTS / "test_predictions.csv")
    assert np.array_equal(table.accepted, accepted_mask(table.score, load_json("freeze.json")["policy"]))
    metrics, cm = compute_metrics(table)
    class_rows = []
    for i, label in enumerate(LABELS):
        subset = table[table.actual == label]
        acc = subset[subset.accepted]
        errors = int((acc.actual != acc.predicted).sum())
        full_precision = float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else 0.0
        full_recall = float(cm[i, i] / cm[i].sum())
        class_rows.append({"class": label, "support": len(subset), "accepted": len(acc),
            "referred": len(subset) - len(acc), "referral_rate": 1 - len(acc) / len(subset),
            "accepted_errors": errors, "accepted_error": errors / len(acc) if len(acc) else None,
            "full_precision": full_precision, "full_recall": full_recall,
            "full_f1": 2 * full_precision * full_recall / (full_precision + full_recall)
                       if full_precision + full_recall else 0.0})
    per_class = pd.DataFrame(class_rows)
    assert per_class.support.sum() == metrics["rows"]
    assert per_class.accepted.sum() == metrics["accepted"]
    assert per_class.accepted_errors.sum() == metrics["accepted_errors"]
    per_class.to_csv(RESULTS / "per_class.csv", index=False)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RESULTS / "confusion_matrix.csv")
    unique_metrics, _ = compute_metrics(table.drop_duplicates("feature_group"))
    sensitivity = {"primary_row_weighted": metrics, "one_per_vector": unique_metrics,
                   "interpretation": "Evaluation weighting only; frozen model and threshold unchanged. Not a training sensitivity."}
    save_json("metrics.json", metrics)
    save_json("sensitivity.json", sensitivity)
    bootstrap = group_bootstrap(table)
    save_json("bootstrap.json", bootstrap)
    print(json.dumps(metrics, indent=2), flush=True)
    print(per_class.to_string(index=False), flush=True)
    return metrics


def group_bootstrap(table, replicates=2000, seed=68003):
    unique = table.drop_duplicates("feature_group").copy()
    multiplicity = table.groupby("feature_group").size().reindex(unique.feature_group).to_numpy()
    # Whole feature groups move together; member-row multiplicity is retained.
    assert multiplicity.sum() == len(table)
    assert table.groupby("feature_group")["actual"].nunique().max() == 1
    assert table.groupby("feature_group")["accepted"].nunique().max() == 1
    codes = {label: i for i, label in enumerate(LABELS)}
    truth = unique.actual.map(codes).to_numpy()
    pred = unique.predicted.map(codes).to_numpy()
    base = unique.baseline_predicted.map(codes).to_numpy()
    accepted = unique.accepted.to_numpy(dtype=bool)
    errors = truth != pred
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts * multiplicity
        sampled_cm = np.bincount(7 * truth + pred, weights=weights, minlength=49).reshape(7, 7)
        base_cm = np.bincount(7 * truth + base, weights=weights, minlength=49).reshape(7, 7)
        selected_metrics, baseline_metrics = metric_from_cm(sampled_cm), metric_from_cm(base_cm)
        accepted_n = int(weights[accepted].sum())
        draws.append({"macro_f1": selected_metrics["macro_f1"],
                      "accuracy": selected_metrics["accuracy"],
                      "macro_recall": selected_metrics["macro_recall"],
                      "macro_f1_gain_over_majority": selected_metrics["macro_f1"] - baseline_metrics["macro_f1"],
                      "coverage": accepted_n / weights.sum(),
                      "accepted_error": float(weights[accepted & errors].sum() / accepted_n) if accepted_n else np.nan})
    frame = pd.DataFrame(draws)
    frame.to_csv(RESULTS / "bootstrap_draws.csv", index=False)
    return {"replicates": replicates, "seed": seed,
            "method": "Unstratified whole-feature-group resampling with member-row multiplicity",
            "scope": "Conditional on fitted model and validation-selected rule; internal collection only",
            "intervals": {column: {"lower": float(frame[column].quantile(0.025)) if frame[column].notna().any() else None,
                                   "upper": float(frame[column].quantile(0.975)) if frame[column].notna().any() else None,
                                   "undefined_replicates": int(frame[column].isna().sum())}
                          for column in frame}}


def _save_figure(fig, number):
    fig.savefig(FIGURES / f"figure_{number}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"figure_{number}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_results(state):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#555555"]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.7), gridspec_kw={"height_ratios": [1, 1.45]})
    counts = state["y"].value_counts().reindex(LABELS)
    bars = axes[0].barh(LABELS, counts, color=palette)
    axes[0].bar_label(bars, padding=3, fmt="{:,.0f}")
    axes[0].set(xlabel="Supplied rows (all 13,611)", xlim=(0, 4400), title="A. Unequal class support")
    axes[0].tick_params(axis="y", labelsize=8)
    for label, color in zip(LABELS, palette):
        ids = state["train_ids"][state["y"].iloc[state["train_ids"]].to_numpy() == label]
        axes[1].scatter(state["X"].Area.iloc[ids], state["X"].AspectRation.iloc[ids],
                        s=5, alpha=.25, color=color, label=label, rasterized=True)
    axes[1].set(xscale="log", xlabel="Area (pixel count; log scale)", ylabel="Aspect ratio (unitless)",
                title="B. Morphology in training rows only")
    axes[1].legend(fontsize=8, markerscale=2, loc="upper right")
    fig.tight_layout(); _save_figure(fig, 1)

    cv = pd.read_csv(RESULTS / "cv_folds.csv")
    summary = pd.read_csv(RESULTS / "cv_summary.csv").sort_values("mean_macro_f1")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for pos, row in enumerate(summary.itertuples()):
        values = cv.loc[cv.candidate == row.candidate, "macro_f1"].to_numpy()
        color = "#AA3377" if row.candidate == state["selected"]["id"] else "#4477AA"
        ax.scatter(values, pos + np.linspace(-.1, .1, 5), color=color, s=24, alpha=.7)
        ax.errorbar(row.mean_macro_f1, pos, xerr=row.sd_macro_f1, fmt="D", color="black", capsize=3)
        ax.text(.14, pos, f"{row.mean_macro_f1:.4f} ± {row.sd_macro_f1:.4f}", va="center", fontsize=9)
    ax.set(yticks=range(len(summary)), yticklabels=summary.candidate,
           xlabel="Macro-F1 (seven classes)",
           title="Training-only grouped CV: mean ± fold SD", xlim=(0, 1))
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout(); _save_figure(fig, 2)

    cm = pd.read_csv(RESULTS / "confusion_matrix.csv", index_col=0).to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    proportions = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i,j]}\n{100*proportions[i,j]:.1f}%", ha="center", va="center",
                    color="white" if proportions[i,j] > .55 else "black", fontsize=8)
    ax.set(xticks=range(7), xticklabels=LABELS, yticks=range(7),
           yticklabels=[f"{label} (n={cm[i].sum()})" for i, label in enumerate(LABELS)],
           xlabel="Predicted variety", ylabel="Supplied variety", title="Held-out test: counts and within-class percentages")
    ax.tick_params(axis="both", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout(); _save_figure(fig, 3)

    curve = pd.read_csv(RESULTS / "validation_risk_coverage.csv").sort_values("coverage")
    policy = load_json("freeze.json")["policy"]
    metrics = load_json("metrics.json")
    intervals = load_json("bootstrap.json")["intervals"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(100 * curve.coverage, 100 * curve.accepted_error, color="#4477AA", label="Validation thresholds")
    if policy["accepted_error"] is not None:
        ax.scatter(100 * policy["coverage"], 100 * policy["accepted_error"], marker="D", s=65,
                   color="#4477AA", label="Selected validation rule", zorder=5)
    if metrics["accepted_error"] is not None:
        x, y = 100 * metrics["coverage"], 100 * metrics["accepted_error"]
        ax.hlines(y, 100*intervals["coverage"]["lower"], 100*intervals["coverage"]["upper"], color="#AA3377")
        ax.vlines(x, 100*intervals["accepted_error"]["lower"], 100*intervals["accepted_error"]["upper"], color="#AA3377")
        ax.scatter(x, y, color="#AA3377", label="Fixed test policy (95% intervals)", zorder=5)
    ax.axhline(5, color="black", linestyle="--", linewidth=1, label="Provisional 5% target")
    ax.set(xlabel="Automatic coverage (% of rows)", ylabel="Errors among automatically labeled rows (%)",
           xlim=(0, 102), ylim=(0, max(10, float(curve.accepted_error.max()) * 100 + 1,
                                      100*(intervals["accepted_error"]["upper"] or 0)+1)),
           title="Validation curve and one frozen test policy")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); _save_figure(fig, 4)

    pc = pd.read_csv(RESULTS / "per_class.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.0), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    positions = np.arange(len(pc))
    axes[0].barh(positions, 100 * pc.referral_rate, color=palette)
    axes[0].set(yticks=positions, yticklabels=pc["class"], xlabel="Referred within class (%)", xlim=(0, 13))
    axes[0].tick_params(axis="y", labelsize=8)
    axes[1].set(xlim=(0, 1), ylim=(-.6, 7.2))
    axes[1].axis("off")
    axes[1].text(.18, 6.8, "Referred / n", ha="center", fontsize=8)
    axes[1].text(.76, 6.8, "Errors / accepted", ha="center", fontsize=8)
    for pos, row in zip(positions, pc.itertuples()):
        axes[1].text(.18, pos, f"{row.referred}/{row.support}", va="center", ha="center", fontsize=9)
        axes[1].text(.76, pos, f"{row.accepted_errors}/{row.accepted}", va="center", ha="center", fontsize=9)
    fig.suptitle("Held-out referral burden by true supplied variety", fontsize=10)
    fig.tight_layout(); _save_figure(fig, 5)
    print("Five figures created from verified audit, training folds, and cached predictions.", flush=True)


def run_all():
    state = audit_and_split()
    compare_models(state)
    freeze_policy(state)
    evaluate_test(state)
    summarize_results()
    plot_results(state)
    return state


if __name__ == "__main__":
    run_all()
