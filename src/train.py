"""Train and evaluate failure prediction models on the AI4I 2020 dataset.

Usage:
    python -m src.train --data data/ai4i2020.csv

Outputs:
    model/model.joblib            the deployed model (gradient boosting, engineered features)
    results/metrics.json          every number reported in the README
    results/test_predictions.json test-set labels and the probabilities of both
                                  gradient boosting models, used by the cost section
    results/train_oof_predictions.json  training-set labels and out-of-fold
                                  probabilities (five-fold), used to choose the
                                  cost-optimal threshold without touching the test set
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import features as F

SEED = 42
TEST_SIZE = 0.2


def make_models():
    return {
        "logistic_raw": make_pipeline(
            FunctionTransformer(F.raw_matrix),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ),
        "boosting_raw": make_pipeline(
            FunctionTransformer(F.raw_matrix),
            HistGradientBoostingClassifier(random_state=SEED),
        ),
        "boosting_engineered": make_pipeline(
            FunctionTransformer(F.engineered_matrix),
            HistGradientBoostingClassifier(random_state=SEED),
        ),
    }


def scores(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if y_score is not None:
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return out


def failure_group(row):
    """Failure-mode label of a row, for example 'HDF', 'TWF+OSF' or 'no mode flag'."""
    modes = [m for m in F.FAILURE_MODES if row[m] == 1]
    return "+".join(modes) if modes else "no mode flag"


def caught_by_group(test_df, y_true, y_pred):
    """For actual failures only: how many of each failure-mode group were caught."""
    failures = test_df[y_true == 1]
    groups = failures.apply(failure_group, axis=1).to_numpy()
    caught = y_pred[y_true == 1]
    out = {}
    for g in sorted(set(groups)):
        mask = groups == g
        out[g] = {"failures": int(mask.sum()), "caught": int(caught[mask].sum())}
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_summary(df):
    any_mode = df[F.FAILURE_MODES].sum(axis=1) > 0
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "type_share": df[F.TYPE].value_counts(normalize=True).round(4).to_dict(),
        "machine_failure": int(df[F.TARGET].sum()),
        "mode_counts": {m: int(df[m].sum()) for m in F.FAILURE_MODES},
        "failure_without_mode": int(((df[F.TARGET] == 1) & ~any_mode).sum()),
        "mode_without_failure": int(((df[F.TARGET] == 0) & any_mode).sum()),
        "rows_with_multiple_modes": int((df[F.FAILURE_MODES].sum(axis=1) > 1).sum()),
    }


def rule_agreement(df):
    rules = F.documented_rules(df)
    return {m: float((rules[m] == df[m]).mean()) for m in rules.columns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ai4i2020.csv")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "model").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    df = F.load_raw(args.data)
    y = df[F.TARGET].to_numpy()
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    results = {
        "data_file_sha256": sha256(args.data),
        "dataset": dataset_summary(df),
        "documented_rule_agreement": rule_agreement(df),
        "split": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_failures": int(y_test.sum()),
        },
        "cross_validation": {},
        "test": {},
        "test_caught_by_failure_mode": {},
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = make_models()
    test_probabilities = {}
    train_oof_probabilities = {}
    for name, model in models.items():
        cvr = cross_validate(
            model, train_df, y_train, cv=cv, scoring=["f1", "average_precision"]
        )
        results["cross_validation"][name] = {
            "f1_mean": float(cvr["test_f1"].mean()),
            "f1_std": float(cvr["test_f1"].std()),
            "pr_auc_mean": float(cvr["test_average_precision"].mean()),
            "pr_auc_std": float(cvr["test_average_precision"].std()),
        }
        if name.startswith("boosting"):
            oof = cross_val_predict(model, train_df, y_train, cv=cv, method="predict_proba")[:, 1]
            train_oof_probabilities[name] = [float(v) for v in oof]
        model.fit(train_df, y_train)
        y_pred = model.predict(test_df)
        y_score = model.predict_proba(test_df)[:, 1]
        results["test"][name] = scores(y_test, y_pred, y_score)
        if name.startswith("boosting"):
            test_probabilities[name] = [float(v) for v in y_score]
        results["test_caught_by_failure_mode"][name] = caught_by_group(test_df, y_test, y_pred)

    # Reference: the documented physical rules, no learning involved.
    rule_pred = F.documented_rules(test_df).max(axis=1).to_numpy()
    results["test"]["documented_rules"] = scores(y_test, rule_pred)
    results["test_caught_by_failure_mode"]["documented_rules"] = caught_by_group(
        test_df, y_test, rule_pred
    )

    joblib.dump(models["boosting_engineered"], out_dir / "model" / "model.joblib")
    (out_dir / "results" / "test_predictions.json").write_text(
        json.dumps({"label": [int(v) for v in y_test], **test_probabilities})
    )
    (out_dir / "results" / "train_oof_predictions.json").write_text(
        json.dumps({"label": [int(v) for v in y_train], **train_oof_probabilities})
    )
    (out_dir / "results" / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
