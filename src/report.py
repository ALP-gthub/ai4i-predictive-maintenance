"""Write the results tables into README.md from results/metrics.json.

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


def main():
    root = Path(__file__).resolve().parent.parent
    r = json.loads((root / "results" / "metrics.json").read_text())
    readme = root / "README.md"
    text = readme.read_text()
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{build(r)}\n{END}{tail}")
    print(build(r))


if __name__ == "__main__":
    main()
