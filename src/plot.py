"""Draw results/caught_by_mode.png from results/metrics.json.

Usage:
    python -m src.plot
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ENGINEERED = "#2a78d6"
RAW = "#eb6834"

MODES = [
    ("HDF", "Heat dissipation"),
    ("OSF", "Overstrain"),
    ("PWF", "Power"),
    ("TWF", "Tool wear (random by design)"),
]


def main():
    root = Path(__file__).resolve().parent.parent
    r = json.loads((root / "results" / "metrics.json").read_text())
    raw = r["test_caught_by_failure_mode"]["boosting_raw"]
    eng = r["test_caught_by_failure_mode"]["boosting_engineered"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bar_h = 0.34
    gap = 0.04
    for i, (code, name) in enumerate(MODES):
        total = eng[code]["failures"]
        y = len(MODES) - 1 - i
        for offset, data, color in [
            (bar_h / 2 + gap / 2, eng, ENGINEERED),
            (-(bar_h / 2 + gap / 2), raw, RAW),
        ]:
            caught = data[code]["caught"]
            share = caught / total * 100
            ax.barh(y + offset, share, height=bar_h, color=color, linewidth=0)
            ax.text(
                max(share, 0) + 1.2,
                y + offset,
                f"{caught} / {total}",
                va="center",
                ha="left",
                fontsize=10,
                color=INK_2,
            )

    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([f"{name}\n({code})" for code, name in reversed(MODES)], color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=INK_2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ENGINEERED),
        plt.Rectangle((0, 0), 1, 1, color=RAW),
    ]
    ax.legend(
        handles,
        ["Raw + engineered features", "Raw measurements only"],
        loc="lower right",
        frameon=False,
        fontsize=10,
        labelcolor=INK,
    )
    fig.suptitle(
        "Share of actual test failures caught, by failure mode",
        x=0.02,
        ha="left",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )
    split = r["split"]
    fig.text(
        0.02,
        0.02,
        f"AI4I 2020 dataset (UCI, CC BY 4.0). Gradient boosting, held-out test set of "
        f"{split['test_rows']:,} rows. Labels show caught / total failures.",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = root / "results" / "caught_by_mode.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
