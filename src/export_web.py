"""Export the trained model for the browser demo in docs/.

1. Writes docs/model.json with the trees of the gradient boosting model.
2. Compares the browser predictor (docs/model.js, run in Node.js) with
   scikit-learn on all 10,000 dataset rows and 20,000 random inputs, and
   writes the result to results/web_export_check.json. Also checks the page's
   °C entry: every temperature converted to °C and back must be unchanged and
   give the same probability. Stops on any mismatch.
3. Copies the training out-of-fold and test-set predictions to docs/ and
   checks the page's cost calculation (costDecision in docs/model.js) against
   an independent Python implementation, for a range of cost pairs: threshold
   chosen on training data, cost measured on the test set. Stops on any
   difference.
4. Copies results/metrics.json, web_export_check.json and caught_by_mode.png
   to docs/, where the page reads them.

Usage:
    python -m src.export_web
"""

import json
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import features as F

ROOT = Path(__file__).resolve().parent.parent


def export(model):
    clf = model.steps[-1][1]
    if clf.n_trees_per_iteration_ != 1:
        raise ValueError("Expected a binary classifier with one tree per iteration.")
    trees = []
    for (predictor,) in clf._predictors:
        nodes = predictor.nodes
        if nodes["is_categorical"].any():
            raise ValueError("Categorical splits are not supported by the browser predictor.")
        trees.append(
            {
                "feature": nodes["feature_idx"].astype(int).tolist(),
                "threshold": nodes["num_threshold"].astype(float).tolist(),
                "missing_left": nodes["missing_go_to_left"].astype(int).tolist(),
                "left": nodes["left"].astype(int).tolist(),
                "right": nodes["right"].astype(int).tolist(),
                "leaf": nodes["is_leaf"].astype(int).tolist(),
                "value": nodes["value"].astype(float).tolist(),
            }
        )
    return {
        "source": "HistGradientBoostingClassifier from model/model.joblib",
        "features": list(clf.feature_names_in_),
        "baseline": float(np.asarray(clf._baseline_prediction).ravel()[0]),
        "trees": trees,
    }


def main():
    model = joblib.load(ROOT / "model" / "model.joblib")
    payload = export(model)
    (ROOT / "docs" / "model.json").write_text(json.dumps(payload, separators=(",", ":")))

    df = F.load_raw(ROOT / "data" / "ai4i2020.csv")
    # Also check inputs outside the dataset: 20,000 random rows over wide ranges,
    # as a user of the demo may type any value.
    rng = np.random.default_rng(0)
    n = 20000
    fuzz = pd.DataFrame(
        {
            F.TYPE: rng.choice(["L", "M", "H"], n),
            F.AIR: rng.uniform(290, 310, n).round(1),
            F.PROCESS: rng.uniform(300, 320, n).round(1),
            F.SPEED: rng.integers(800, 3200, n),
            F.TORQUE: rng.uniform(0, 90, n).round(1),
            F.WEAR: rng.integers(0, 300, n),
        }
    )
    checked = pd.concat([df[[F.TYPE] + F.RAW_NUMERIC], fuzz], ignore_index=True)
    proba = model.predict_proba(checked)[:, 1]
    rows = checked.to_dict(orient="records")

    tmp = ROOT / "results" / "_web_check_input.json"
    tmp.write_text(json.dumps({"rows": rows}))
    out = subprocess.run(
        ["node", str(ROOT / "src" / "check_web_model.js"), str(tmp), str(ROOT / "docs")],
        capture_output=True,
        text=True,
        check=True,
    )
    tmp.unlink()
    result = json.loads(out.stdout)
    js = np.array(result["probabilities"])

    diff = np.abs(js - proba)
    report = {
        "dataset_rows_compared": int(len(df)),
        "random_rows_compared": int(n),
        "max_abs_probability_difference": float(diff.max()),
        "decision_mismatches_at_0_5": int(((js >= 0.5) != (proba >= 0.5)).sum()),
        "celsius_roundtrip_mismatches": int(result["roundtripMismatches"]),
        "celsius_path_probability_differences": int(result["celsiusPathDifferences"]),
    }
    (ROOT / "results" / "web_export_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["decision_mismatches_at_0_5"] or report["max_abs_probability_difference"] > 1e-9:
        raise SystemExit("Browser predictor does not match scikit-learn.")
    if report["celsius_roundtrip_mismatches"] or report["celsius_path_probability_differences"]:
        raise SystemExit("The page's °C conversion changes model inputs.")

    check_cost()

    # The page reads its numbers from these copies, never from hand-typed values.
    for name in ["metrics.json", "web_export_check.json", "caught_by_mode.png"]:
        shutil.copyfile(ROOT / "results" / name, ROOT / "docs" / name)


# Cost pairs (unplanned failure, inspection) for the check, from cheap to
# very expensive failures, plus non-integer values.
COST_CASES = [[1, 1], [2, 1], [5, 1], [10, 1], [20, 1], [50, 1], [100, 1], [1000, 1], [1, 3], [7.5, 2], [250, 40]]


def python_threshold(labels, probs, fail_cost, inspect_cost):
    """Brute force reference: the cheapest threshold over every distinct probability.

    None means "no alarm at all", which is also the starting point. On a tie
    the higher threshold (fewer alarms) is kept.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    failures = int(labels.sum())
    best_t, best_cost = None, failures * fail_cost
    for t in sorted(set(probs.tolist()), reverse=True):
        alarm = probs >= t
        tp = int(((labels == 1) & alarm).sum())
        fp = int(((labels == 0) & alarm).sum())
        cost = (failures - tp) * fail_cost + (tp + fp) * inspect_cost
        if cost < best_cost:
            best_t, best_cost = t, cost
    return best_t


def python_outcome(labels, probs, threshold, fail_cost, inspect_cost):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    alarm = np.zeros(len(probs), bool) if threshold is None else probs >= threshold
    tp = int(((labels == 1) & alarm).sum())
    fp = int(((labels == 0) & alarm).sum())
    fn = int(((labels == 1) & ~alarm).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "cost": fn * fail_cost + (tp + fp) * inspect_cost}


def check_cost():
    """The threshold is chosen on training out-of-fold predictions and applied to the test set."""
    train = json.loads((ROOT / "results" / "train_oof_predictions.json").read_text())
    test = json.loads((ROOT / "results" / "test_predictions.json").read_text())
    for name in ["train_oof_predictions.json", "test_predictions.json"]:
        shutil.copyfile(ROOT / "results" / name, ROOT / "docs" / name)
    cases_file = ROOT / "results" / "_cost_cases.json"
    cases_file.write_text(json.dumps(COST_CASES))
    out = subprocess.run(
        ["node", str(ROOT / "src" / "check_cost.js"), str(ROOT / "docs"), str(cases_file)],
        capture_output=True, text=True, check=True,
    )
    cases_file.unlink()
    differences = 0
    for r in json.loads(out.stdout):
        name, a, b = r["name"], r["failCost"], r["inspectCost"]
        t = python_threshold(train["label"], train[name], a, b)
        ref = python_outcome(test["label"], test[name], t, a, b)
        js = r["chosen"]
        same = (
            r["threshold"] == t
            and all(js[k] == ref[k] for k in ["tp", "fp", "fn"])
            and abs(js["cost"] - ref["cost"]) < 1e-9
        )
        differences += 0 if same else 1
    report = json.loads((ROOT / "results" / "web_export_check.json").read_text())
    report["cost_cases_compared"] = len(COST_CASES) * 2
    report["cost_differences"] = differences
    (ROOT / "results" / "web_export_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"cost_cases_compared": report["cost_cases_compared"], "cost_differences": differences}))
    if differences:
        raise SystemExit("The page's cost calculation does not match the Python reference.")


if __name__ == "__main__":
    main()
