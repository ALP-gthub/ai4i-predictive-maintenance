"""Export the trained model for the browser demo in docs/.

1. Writes docs/model.json with the trees of the gradient boosting model.
2. Compares the browser predictor (docs/model.js, run in Node.js) with
   scikit-learn on all 10,000 dataset rows and 20,000 random inputs, and
   writes the result to results/web_export_check.json. Also checks the page's
   °C entry: every temperature converted to °C and back must be unchanged and
   give the same probability. Stops on any mismatch.
3. Copies results/metrics.json, web_export_check.json and caught_by_mode.png
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

    # The page reads its numbers from these copies, never from hand-typed values.
    for name in ["metrics.json", "web_export_check.json", "caught_by_mode.png"]:
        shutil.copyfile(ROOT / "results" / name, ROOT / "docs" / name)


if __name__ == "__main__":
    main()
