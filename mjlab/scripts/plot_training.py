"""Plot recorded training rewards and goal metrics against steps and wall time.

Pass --run-dir repeatedly in chronological order for resumed sessions. These
are training-distribution metrics, not frozen-policy evaluation success rates.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play2perfect_mjlab.assets import sha256


def read_history(directories):
    retained, offset, contracts, sources = [], 0.0, [], []
    for directory in directories:
        config_path = directory / "config.json"
        metrics_path = directory / "metrics.jsonl"
        config = json.loads(config_path.read_text())
        rows = []
        lines = metrics_path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(f"Corrupt metrics row {index + 1}: {metrics_path}")
                break  # An active writer can leave an incomplete last line.
        if not rows:
            raise ValueError(f"No complete metrics: {metrics_path}")
        if any(b["transitions"] <= a["transitions"] for a, b in zip(rows, rows[1:])):
            raise ValueError("Transitions must increase within each session")
        contract = config["resume_contract"]
        if contracts and contracts[-1] != contract:
            raise ValueError("Training contracts differ across sessions")
        contracts.append(contract)
        if retained:
            if not config.get("resume"):
                raise ValueError("Only resumed sessions can extend a curve")
            boundary = (
                rows[0]["transitions"] - contract["num_envs"] * contract["horizon"]
            )
            if boundary not in {row["transitions"] for row in retained}:
                raise ValueError("Resume boundary is absent from previous metrics")
            retained = [row for row in retained if row["transitions"] <= boundary]
        setup = config.get("setup_timing", {}).get("main_to_environment_ready_seconds")
        if setup is None:
            raise ValueError("Missing environment setup timing")
        if any(b["wall_seconds"] < a["wall_seconds"] for a, b in zip(rows, rows[1:])):
            raise ValueError("Use separate run directories for restarted timers")
        for row in rows:
            row["recorded_process_seconds"] = offset + setup + row["wall_seconds"]
        retained.extend(rows)
        offset += setup + rows[-1]["wall_seconds"]
        sources.append(
            {
                "run_directory": str(directory),
                "config_sha256": sha256(config_path),
                "metrics_sha256": sha256(metrics_path),
                "base_checkpoint_sha256": contract["base_checkpoint_sha256"],
                "resume": config.get("resume"),
            }
        )
    return retained, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, sources = read_history(args.run_dir)
    xs = [
        np.array([row["transitions"] for row in rows]) / 1e6,
        np.array([row["recorded_process_seconds"] for row in rows]) / 3600,
    ]
    rewards = np.array([row["block_mean_reward"] for row in rows]).mean(axis=1)
    goals = np.array([row["previous_insertion_all_goals_hit"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, (x, label) in enumerate(
        zip(xs, ["Transitions (millions)", "Recorded process time (hours)"])
    ):
        axes[0, column].plot(x, rewards, color="#216c62")
        axes[1, column].plot(x, goals, color="#216c62")
        axes[1, column].set_ylim(0, 1)
        for axis in axes[:, column]:
            axis.set_xlabel(label)
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean rollout reward, six SAPG blocks")
    axes[1, 0].set_ylabel("Training insertion: all goals hit")
    figure.suptitle("MJLab training diagnostics — not held-out policy success")
    figure.savefig(args.output / "training_curves.png", dpi=180)
    figure.savefig(args.output / "training_curves.svg")
    plt.close(figure)
    report = {
        "sources": sources,
        "points": len(rows),
        "last_transitions": rows[-1]["transitions"],
        "time_definition": "Environment construction plus observer wall time; includes "
        "learner setup, compilation and duty-cycle idle. Excludes imports before main, "
        "scheduler wait and downtime. Time spent on discarded post-checkpoint updates "
        "remains charged. The last scalar can precede final checkpoint serialization.",
        "score_definition": "Unsmoothed rollout rewards and previous-episode training "
        "goal metrics; not clean withdrawal or held-out evaluation success.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "training_curves.png")


if __name__ == "__main__":
    main()
