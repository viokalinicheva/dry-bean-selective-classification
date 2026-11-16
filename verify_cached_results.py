"""Independently verify cached numerical artifacts; never fit or predict a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, recall_score,
)

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
LABELS = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
CHECKS = []


def read_csv(name):
    return pd.read_csv(RESULTS / name, float_precision="round_trip")


def read_json(name):
    return json.loads((RESULTS / name).read_text())


def check(condition, name):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)


def close(actual, expected, name):
    check(np.allclose(actual, expected, atol=1e-13, rtol=0, equal_nan=True), name)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_metrics(table):
    truth, pred, baseline = table.actual, table.predicted, table.baseline_predicted
    accepted = table.accepted.to_numpy(dtype=bool)
    incorrect = truth.to_numpy() != pred.to_numpy()
    def all_class(p):
        return {"accuracy": accuracy_score(truth, p),
                "macro_f1": f1_score(truth, p, labels=LABELS, average="macro", zero_division=0),
                "macro_recall": recall_score(truth, p, labels=LABELS, average="macro", zero_division=0)}
    count, errors = int(accepted.sum()), int((accepted & incorrect).sum())
    return {"rows": len(table), "groups": table.feature_group.nunique(),
            "full_coverage": all_class(pred), "baseline": all_class(baseline),
            "accepted": count, "referred": len(table) - count,
            "accepted_errors": errors, "total_errors": int(incorrect.sum()),
            "referred_errors": int((~accepted & incorrect).sum()),
            "coverage": count / len(table),
            "accepted_error": errors / count if count else None,
            "observed_target_met": bool(count and 20 * errors <= count)}


def compare_dict(got, expected, prefix):
    for key, value in expected.items():
        if isinstance(value, dict):
            compare_dict(got[key], value, f"{prefix}.{key}")
        elif value is None:
            check(got[key] is None, f"{prefix}.{key}")
        else:
            close(got[key], value, f"{prefix}.{key}")


test, val = read_csv("test_predictions.csv"), read_csv("validation_predictions.csv")
freeze, receipt = read_json("freeze.json"), read_json("test_receipt.json")
policy = freeze["policy"]
for table, name in [(test, "test"), (val, "validation")]:
    check(table.accepted.dtype == bool and table.correct.dtype == bool, f"{name}.boolean parsing")
    check(table.row_id.is_unique, f"{name}.unique row IDs")
    check(np.array_equal(table.correct, table.actual.eq(table.predicted)), f"{name}.correct flags")
    probabilities = table[[f"score_{label}" for label in LABELS]].to_numpy()
    check(np.isfinite(probabilities).all() and (probabilities >= 0).all()
          and (probabilities <= 1).all(), f"{name}.score range")
    close(probabilities.sum(axis=1), np.ones(len(table)), f"{name}.score sums")
    check(np.array_equal(probabilities.max(axis=1), table.score), f"{name}.max scores")
    check(np.array_equal(np.asarray(LABELS)[probabilities.argmax(axis=1)], table.predicted),
          f"{name}.argmax labels")
    check(np.array_equal(table.accepted, table.score >= policy["threshold"]),
          f"{name}.full-precision acceptance")
    check(table.groupby("feature_group")[["actual", "predicted", "score", "accepted"]]
          .nunique().to_numpy().max() == 1, f"{name}.whole identical groups")

recomputed = primary_metrics(test)
compare_dict(recomputed, read_json("metrics.json"), "metrics")
close(balanced_accuracy_score(test.actual, test.predicted),
      recomputed["full_coverage"]["macro_recall"], "balanced accuracy equals macro recall")
cm = confusion_matrix(test.actual, test.predicted, labels=LABELS)
close(cm, pd.read_csv(RESULTS / "confusion_matrix.csv", index_col=0).to_numpy(), "confusion counts")
pc = read_csv("per_class.csv").set_index("class").loc[LABELS]
precision, recall, f1, support = precision_recall_fscore_support(
    test.actual, test.predicted, labels=LABELS, zero_division=0)
for field, values in [("support", support), ("full_precision", precision),
                      ("full_recall", recall), ("full_f1", f1)]:
    close(values, pc[field], f"per_class.{field}")
for label in LABELS:
    subset = test[test.actual.eq(label)]
    accepted = subset[subset.accepted]
    errors = int(accepted.actual.ne(accepted.predicted).sum())
    close([len(accepted), len(subset) - len(accepted), errors],
          pc.loc[label, ["accepted", "referred", "accepted_errors"]].astype(float),
          f"per_class.{label} counts")
    close([1 - len(accepted) / len(subset), errors / len(accepted)],
          pc.loc[label, ["referral_rate", "accepted_error"]].astype(float),
          f"per_class.{label} rates")

# I enumerate score candidates independently rather than calling analysis.py.
candidates = []
for threshold in sorted(set([0.0] + val.score.tolist())):
    accepted = val.score.ge(threshold)
    count = int(accepted.sum())
    errors = int((accepted & ~val.correct).sum())
    candidates.append((threshold, count, errors))
feasible = [row for row in candidates if row[1] > 0 and 20 * row[2] <= row[1]]
chosen = sorted(feasible, key=lambda row: (-row[1], row[0]))[0]
check(chosen == (policy["threshold"], policy["accepted"], policy["errors"]),
      "validation maximum-coverage feasible threshold")
curve = read_csv("validation_risk_coverage.csv").sort_values("threshold")
close(curve[["threshold", "accepted", "errors"]], np.asarray(candidates), "full validation curve")
check(all(20 * errors > count for _, count, errors in candidates if count > chosen[1]),
      "all higher-coverage validation candidates fail empirical target")

sensitivity = read_json("sensitivity.json")
compare_dict(primary_metrics(test.drop_duplicates("feature_group")),
             sensitivity["one_per_vector"], "unique-vector sensitivity")
compare_dict(recomputed, sensitivity["primary_row_weighted"], "row-weight sensitivity")
folds, summary = read_csv("cv_folds.csv"), read_csv("cv_summary.csv").set_index("candidate")
for candidate, rows in folds.groupby("candidate"):
    close([rows.macro_f1.mean(), rows.macro_f1.std(ddof=1), rows.accuracy.mean()],
          summary.loc[candidate, ["mean_macro_f1", "sd_macro_f1", "mean_accuracy"]],
          f"CV aggregation {candidate}")
eligible = summary.drop(index="Majority")
check(eligible.mean_macro_f1.idxmax() == freeze["selected"]["id"] == "LR_C100",
      "CV selection agrees with maximum training-only mean")

bootstrap, draws = read_json("bootstrap.json"), read_csv("bootstrap_draws.csv")
check(len(draws) == bootstrap["replicates"] == 2000, "bootstrap replicate count")
for column in draws:
    close([draws[column].quantile(.025), draws[column].quantile(.975)],
          [bootstrap["intervals"][column]["lower"], bootstrap["intervals"][column]["upper"]],
          f"bootstrap stored interval {column}")
    check(draws[column].isna().sum() == bootstrap["intervals"][column]["undefined_replicates"],
          f"bootstrap undefined count {column}")

# I reconstruct each seeded group sample and independently use sklearn metrics.
unique = test.drop_duplicates("feature_group")
multiplicity = test.groupby("feature_group").size().reindex(unique.feature_group).to_numpy()
rng = np.random.default_rng(bootstrap["seed"])
replicated = []
for _ in range(2000):
    weights = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique)) * multiplicity
    accepted = unique.accepted.to_numpy()
    accepted_count = weights[accepted].sum()
    f1_model = f1_score(unique.actual, unique.predicted, labels=LABELS, average="macro",
                        sample_weight=weights, zero_division=0)
    f1_base = f1_score(unique.actual, unique.baseline_predicted, labels=LABELS, average="macro",
                       sample_weight=weights, zero_division=0)
    replicated.append({"macro_f1": f1_model,
        "accuracy": accuracy_score(unique.actual, unique.predicted, sample_weight=weights),
        "macro_recall": recall_score(unique.actual, unique.predicted, labels=LABELS,
                                       average="macro", sample_weight=weights, zero_division=0),
        "macro_f1_gain_over_majority": f1_model - f1_base,
        "coverage": accepted_count / weights.sum(),
        "accepted_error": weights[accepted & unique.actual.ne(unique.predicted).to_numpy()].sum() / accepted_count})
close(pd.DataFrame(replicated)[draws.columns], draws, "all seeded bootstrap draws independently reproduced")

for path, expected, name in [
    (PROJECT / "protocol.json", freeze["protocol_sha256"], "protocol hash"),
    (RESULTS / "frozen_model.joblib", freeze["model_sha256"], "model byte hash"),
    (RESULTS / "freeze.json", receipt["freeze_sha256"], "freeze hash"),
    (RESULTS / "test_predictions.csv", receipt["cache_sha256"], "cache hash")]:
    check(digest(path) == expected, name)
check(datetime.fromisoformat(freeze["created_utc"]) < datetime.fromisoformat(receipt["created_utc"]),
      "freeze timestamp precedes test receipt")
notebook = json.loads((PROJECT / "dry_bean_classification.ipynb").read_text())
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
check([cell["execution_count"] for cell in code_cells] == list(range(1, 8)), "notebook execution order")
check(not any(output.get("output_type") == "error" for cell in code_cells for output in cell["outputs"]),
      "notebook has no error outputs")

report = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Independent cached-artifact verification only: no model loading, fitting, or predict calls",
    "passed_checks": len(CHECKS), "checks": CHECKS,
    "selected_model": freeze["selected"]["id"], "validation_policy": policy,
    "test_recomputed": recomputed,
    "interpretation_guardrails": [
        "Observed accepted error 124/2536 = 4.8896% meets the empirical 5% target, but its 95% group-bootstrap interval is 4.0832%-5.7778% and crosses 5%.",
        "SIRA accepted errors 45/471 = 9.5541% and DERMASON 36/652 = 5.5215% exceed 5%; the global target is not a per-class guarantee.",
        "BOMBAY had 0/105 accepted errors in this internal test. Zero observed errors do not establish zero population risk.",
        "75 of 199 model errors occur in referred rows; referrals are not observed human corrections.",
        "Unique-vector sensitivity is evaluation weighting only, not a deduplicated-training experiment."]}
destination = RESULTS / "cached_verification.json"
destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": report["status"], "passed_checks": len(CHECKS),
                  "report": str(destination.relative_to(PROJECT)), "test": recomputed,
                  "validation_chosen": chosen}, indent=2))
