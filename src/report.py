"""Write the results tables into README.md, and the model card into
model/README.md, from results/metrics.json.

Every number in the README results section comes from this script, so the
README cannot drift from the actual training run.

Usage:
    python -m src.report
"""

import json
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

NAMES = {
    "logistic_raw": "Logistic regression, raw measurements",
    "boosting_raw": "Gradient boosting, raw measurements",
    "boosting_engineered": "Gradient boosting, raw + engineered features",
    "documented_rules": "Documented physical rules (reference, no learning)",
}
GROUP_ORDER = ["HDF", "OSF", "PWF", "TWF", "HDF+PWF", "TWF+OSF", "no mode flag"]


def pct(x):
    return f"{x * 100:.1f}%"


def build(r):
    split = r["split"]
    lines = [
        f"Test set: {split['test_rows']:,} rows held out from training "
        f"({split['test_failures']} actual failures), stratified split, seed {split['seed']}.",
        "",
        "| Approach | Precision | Recall | F1 | PR-AUC | Missed failures | False alarms |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in NAMES.items():
        t = r["test"][key]
        pr_auc = f"{t['pr_auc']:.3f}" if "pr_auc" in t else "n/a"
        lines.append(
            f"| {label} | {pct(t['precision'])} | {pct(t['recall'])} | {t['f1']:.3f} "
            f"| {pr_auc} | {t['fn']} | {t['fp']} |"
        )

    lines += [
        "",
        "Five-fold cross-validation on the training set (mean ± standard deviation):",
        "",
        "| Model | F1 | PR-AUC |",
        "|---|---|---|",
    ]
    for key, cv in r["cross_validation"].items():
        lines.append(
            f"| {NAMES[key]} | {cv['f1_mean']:.3f} ± {cv['f1_std']:.3f} "
            f"| {cv['pr_auc_mean']:.3f} ± {cv['pr_auc_std']:.3f} |"
        )

    groups = r["test_caught_by_failure_mode"]["boosting_engineered"]
    rules = r["test_caught_by_failure_mode"]["documented_rules"]
    raw = r["test_caught_by_failure_mode"]["boosting_raw"]
    lines += [
        "",
        "Actual test failures caught, by failure mode:",
        "",
        "| Failure mode | Failures | Boosting, raw | Boosting, engineered | Documented rules |",
        "|---|---|---|---|---|",
    ]
    for g in [g for g in GROUP_ORDER if g in groups] + [g for g in groups if g not in GROUP_ORDER]:
        lines.append(
            f"| {g} | {groups[g]['failures']} | {raw[g]['caught']} "
            f"| {groups[g]['caught']} | {rules[g]['caught']} |"
        )
    return "\n".join(lines)


MODEL_CARD = """---
library_name: sklearn
pipeline_tag: tabular-classification
tags:
- predictive-maintenance
- tabular-classification
- scikit-learn
---

# AI4I 2020 failure prediction model

Gradient boosting classifier (scikit-learn `HistGradientBoostingClassifier`) that predicts machine failure from the process conditions of a milling operation. It is trained on the public AI4I 2020 Predictive Maintenance Dataset.

* Source code, data, method and full results: https://github.com/ALP-gthub/ai4i-predictive-maintenance
* Browser demo: https://huggingface.co/spaces/ALP-hface/ai4i-failure-prediction

## Inputs

Product quality type (L, M, H), air temperature [K], process temperature [K], rotational speed [rpm], torque [Nm] and tool wear [min]. The pipeline adds three physical quantities itself: process minus air temperature [K], mechanical power [W] and tool wear × torque [min·Nm].

## Results on the held-out test set

{split_line}

| Metric | Value |
|---|---|
| Precision | {precision} |
| Recall | {recall} |
| F1 | {f1:.3f} |
| PR-AUC | {pr_auc:.3f} |
| Missed failures | {fn} |
| False alarms | {fp} |

Five-fold cross-validation on the training set: F1 {cv_f1:.3f} ± {cv_f1_std:.3f}.

## Use

```python
import joblib, pandas as pd
model = joblib.load("model.joblib")  # needs src/features.py from this repository on the import path
row = pd.DataFrame([{{"Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
                     "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100}}])
print(model.predict_proba(row)[0, 1])
```

Saved with scikit-learn {sklearn} under Python {python}. Load it with the same scikit-learn version.

## Limitations

The dataset is synthetic and its failures follow fixed rules. The scores above do not carry over to real machine data, where limits are not sharp and sensors are noisy. Tool wear failures and random failures are assigned at random in the dataset and cannot be predicted.

## Data and citation

AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C. License: CC BY 4.0.

## Author

ALP ATA
"""


def model_card(r):
    t = r["test"]["boosting_engineered"]
    cv = r["cross_validation"]["boosting_engineered"]
    s = r["split"]
    return MODEL_CARD.format(
        split_line=f"{s['test_rows']:,} rows not used in training, {s['test_failures']} actual failures, "
        f"stratified split, seed {s['seed']}.",
        precision=pct(t["precision"]),
        recall=pct(t["recall"]),
        f1=t["f1"],
        pr_auc=t["pr_auc"],
        fn=t["fn"],
        fp=t["fp"],
        cv_f1=cv["f1_mean"],
        cv_f1_std=cv["f1_std"],
        sklearn=r["environment"]["scikit_learn"],
        python=r["environment"]["python"],
    )


def main():
    root = Path(__file__).resolve().parent.parent
    r = json.loads((root / "results" / "metrics.json").read_text())
    readme = root / "README.md"
    text = readme.read_text()
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{build(r)}\n{END}{tail}")
    (root / "model" / "README.md").write_text(model_card(r))
    print(build(r))


if __name__ == "__main__":
    main()
