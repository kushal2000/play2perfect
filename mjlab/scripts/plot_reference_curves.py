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

RESTART_DISPLAY_WARMUP = 100


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
    success, success_axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    mobile, mobile_axes = plt.subplots(4, 1, figsize=(6, 12), constrained_layout=True)
    summary = {}
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
        ys = [columns["reward"], 100 * columns["training_all_goals_hit"]]
        figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for row, values in enumerate(ys):
            smoothed = trailing_mean(values, restarts)
            for column, x in enumerate(xs):
                axis = axes[row, column]
                axis.plot(x, values, color="#216c62", alpha=0.16, linewidth=0.5)
                axis.plot(x, smoothed, color="#216c62", linewidth=1)
                axis.set_xlabel(
                    "Transitions (millions)"
                    if column == 0
                    else "Assembly Fine-tuning Time (hours)"
                )
                axis.grid(alpha=0.2)
                if row == 1:
                    axis.set_ylim(0, 100)
            axes[row, 0].set_ylabel(
                "Mean rollout reward"
                if row == 0
                else "Success Rate (%) · training goals"
            )
        figure.suptitle(f"{info['title']} · seed 0 · {info['hardware']}")
        figure.savefig(args.output / f"{recipe}.png", dpi=150)
        plt.close(figure)
        overview_axis.plot(
            xs[1], trailing_mean(ys[0], restarts), color="#216c62", linewidth=1
        )
        overview_axis.set_title(info["title"])
        overview_axis.set_xlabel("Assembly Fine-tuning Time (hours)")
        overview_axis.set_ylabel("Mean rollout reward")
        overview_axis.grid(alpha=0.2)
        smoothed_success = trailing_mean(ys[1], restarts)
        display_success = smoothed_success.copy()
        restart_gaps = []
        for start in map(int, np.flatnonzero(restarts)):
            stop = min(start + RESTART_DISPLAY_WARMUP, len(rows))
            display_success[start:stop] = np.nan
            restart_gaps.append(
                {
                    "first_epoch": int(columns["epoch"][start]),
                    "last_epoch": int(columns["epoch"][stop - 1]),
                    "updates": stop - int(start),
                    "start_hours": float(xs[1][start]),
                    "end_hours": float(xs[1][stop - 1]),
                }
            )
        for success_axis in (
            success_axes.flat[len(summary)],
            mobile_axes[len(summary)],
        ):
            success_axis.plot(xs[1], display_success, color="#216c62", linewidth=1.4)
            success_axis.set(
                title=info["title"].replace("Beam stage", "Multipart assembly · stage"),
                xlabel="Assembly Fine-tuning Time (hours)",
                ylabel="Success Rate (%)",
                ylim=(0, 100),
                xlim=(0, None),
                yticks=[0, 20, 40, 60, 80, 100],
            )
            for index, boundary in enumerate(xs[1][restarts]):
                success_axis.axvline(
                    boundary,
                    color="#888888",
                    linewidth=0.8,
                    linestyle=":",
                    label="Resume · 100-update gap" if index == 0 else None,
                )
            if restarts.any():
                success_axis.legend(loc="lower right", fontsize=8)
            success_axis.grid(alpha=0.2)
        summary[recipe] = {
            "title": info["title"],
            "final_percent": float(smoothed_success[-1]),
            "peak_percent": float(smoothed_success.max()),
            "final_raw_percent": float(ys[1][-1]),
            "training_hours": float(xs[1][-1]),
            "data_sha256": info["sha256"],
            "restart_display_gaps": restart_gaps,
        }
    overview.suptitle("MJLab assembly fine-tuning from the pre-trained play policy")
    overview.savefig(args.output / "reward_overview.png", dpi=150)
    plt.close(overview)
    success.suptitle("MJLab · training goal completion · one seed per task")
    success.savefig(args.output / "success_overview.png", dpi=150)
    plt.close(success)
    mobile.suptitle("MJLab · training goal completion\nOne seed per task")
    mobile.savefig(args.output / "success_overview_mobile.png", dpi=150)
    plt.close(mobile)
    (args.output / "success_summary.json").write_text(
        json.dumps(
            {
                "metric": "Training all-goal completion, not clean-release evaluation",
                "smoothing": "Trailing mean of up to 20 updates; reset per session",
                "overview_display": "The first 100 updates after each recorded "
                "resume are omitted as gaps. This fixed display rule removes "
                "counter warm-up; it does not prove every world has completed an "
                "episode by update 100. Initial training is retained. Detailed "
                "plots, raw data, final values and peaks remain unfiltered.",
                "time": manifest["time_definition"],
                "runs": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
