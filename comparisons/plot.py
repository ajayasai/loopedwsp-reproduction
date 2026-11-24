"""Render an annotated accuracy grid from the recorded GPU predictions summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


TASKS = {"inventory": "Inventory shortfalls", "scheduling": "Booking conflicts", "telemetry": "Sensor alarms"}
METHODS = {"original_1": "Original · 1 pass", "original_4": "Original · 4 passes",
           "original_full": "Original · full depth", "full_feedforward": "Global feedforward",
           "local_w2": "Local · 2 taps", "local_w8": "Local · 8 taps", "local_w32": "Local · 32 taps"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads((args.version / "output/comparison-results.json").read_bytes())
    rows = {(r["task"], r["method"], r["precision"], r["size"]): r["published"] for r in data["rows"]}
    sizes = [8, 16, 32, 64, 128]
    groups = [("float16", "FP16"), ("float32", "FP32"), ("float64", "FP64")]
    cmap = colors.LinearSegmentedColormap.from_list("accuracy", ["#f7f7f2", "#a6d7d3", "#0b645e"])
    cmap.set_bad("#dddddf")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.9))
    fig.subplots_adjust(left=.17, right=.985, top=.875, bottom=.055, hspace=.59)
    fig.text(.02, .978, "Exact records by context, precision and depth", fontsize=20, va="top", weight="bold")
    fig.text(.02, .939, "Each cell shows complete records correct out of 64 on the published test set. "
                       "A dash means every flag failed the numerical screen.", fontsize=10, va="top")
    for ax, (task, title) in zip(axes, TASKS.items()):
        matrix = np.zeros((len(METHODS), len(sizes)*len(groups)))
        for i, method in enumerate(METHODS):
            for g, (dtype, _) in enumerate(groups):
                for j, n in enumerate(sizes):
                    row = rows[task, method, dtype, n]
                    matrix[i, 5*g+j] = np.nan if row["invalid"] == row["n_labels"] else row["exact"]
        ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=64, aspect="auto")
        for (i, j), count in np.ndenumerate(matrix):
            label = "—" if np.isnan(count) else str(int(count))
            ax.text(j, i, label, ha="center", va="center", fontsize=8.4,
                    color="white" if count >= 46 else "#192a2b")
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=23)
        for g, (_, precision) in enumerate(groups):
            ax.text(5*g+2, 1.025, precision, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9.5, weight="bold")
        for j in (4.5, 9.5):
            ax.axvline(j, color="white", lw=3)
        ax.axhline(2.5, color="white", lw=2)
        ax.set_yticks(range(len(METHODS)), list(METHODS.values()), fontsize=9)
        ax.set_xticks(range(15), sizes*3, fontsize=9)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("Items per record", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


if __name__ == "__main__":
    main()
