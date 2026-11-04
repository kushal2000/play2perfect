"""Rebuild the committed research-reference plots without training or a GPU.

These data predate package extraction. They show training metrics, not held-out
success. Their warm clock excludes cold setup; plot_training.py uses the more
complete recorded process clock for new runs.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def trailing_mean(values, restarts, window=20):
    result, start = [], 0
    for index, value in enumerate(values):
        if restarts[index]:
            start = index
        result.append(np.mean(values[max(start, index - window + 1) : index + 1]))
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/reference"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    overview, overview_axes = plt.subplots(
        2, 2, figsize=(10, 6), constrained_layout=True
    )
    for (recipe, info), overview_axis in zip(
        manifest["runs"].items(), overview_axes.flat, strict=True
    ):
        path = args.data / info["data"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {path}")
        with gzip.open(path, "rt") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != info["points"]:
            raise ValueError(f"Reference row count mismatch: {path}")
        columns = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
        restarts = columns["session_restart"].astype(bool)
        xs = [columns["transitions"] / 1e6, columns["warm_seconds"] / 3600]
        ys = [columns["reward"], columns["training_all_goals_hit"]]
        figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for row, values in enumerate(ys):
            smoothed = trailing_mean(values, restarts)
            for column, x in enumerate(xs):
                axis = axes[row, column]
                axis.plot(x, values, color="#216c62", alpha=0.16, linewidth=0.5)
                axis.plot(x, smoothed, color="#216c62", linewidth=1)
                axis.set_xlabel(
                    "Transitions (millions)" if column == 0 else "Warm training hours"
                )
                axis.grid(alpha=0.2)
                if row == 1:
                    axis.set_ylim(0, 1)
            axes[row, 0].set_ylabel(
                "Mean rollout reward"
                if row == 0
                else "Training insertion: all goals hit"
            )
        figure.suptitle(f"{info['title']} · seed 0 · {info['hardware']}")
        figure.savefig(args.output / f"{recipe}.png", dpi=150)
        plt.close(figure)
        overview_axis.plot(
            xs[1], trailing_mean(ys[0], restarts), color="#216c62", linewidth=1
        )
        overview_axis.set_title(info["title"])
        overview_axis.set_xlabel("Warm training hours")
        overview_axis.set_ylabel("Mean rollout reward")
        overview_axis.grid(alpha=0.2)
    overview.suptitle("Direct play-foundation fine-tuning in MJLab · one training seed")
    overview.savefig(args.output / "reward_overview.png", dpi=150)
    plt.close(overview)
    print(args.output)


if __name__ == "__main__":
    main()
