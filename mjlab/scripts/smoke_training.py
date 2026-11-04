"""Check one real update, exact learner restoration and one resumed update.

This tests software and checkpoint portability. Four controls per update do not
measure task success. Prepare public assets, weights and the optional Warp
overlay first. Run from the mjlab directory with the selected pinned environment.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from play2perfect_mjlab.assets import sha256
from play2perfect_mjlab.recipes import RECIPES


def run(command, log):
    with log.open("w") as output:
        subprocess.run(
            [sys.executable, "-u", *command],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )


def check_metrics(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1, "Expected exactly one update"
    row = rows[0]
    assert row["invalid_world_resets"] == 0
    assert not row["physics_peak"]["overflow_flags"]
    for key in ("policy_loss", "actor_value_loss", "entropy", "kl"):
        assert math.isfinite(row[key]), key
    assert all(math.isfinite(value) for value in row["block_mean_reward"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/smoke_training"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--discard-intermediate-checkpoints", action="store_true")
    args = parser.parse_args()
    scripts = Path(__file__).parent
    for name in args.recipe or RECIPES:
        destination = args.output_dir / name
        destination.mkdir(parents=True, exist_ok=False)
        common = [
            str(scripts / "train_recipe.py"),
            "--recipe",
            name,
            "--device",
            args.device,
            "--num-envs",
            "6",
            "--horizon",
            "4",
            "--minibatch-size",
            "24",
            "--max-seconds",
            "180",
            "--torch-threads",
            "1",
        ]
        if args.allow_dirty:
            common.append("--allow-dirty")
        initial = destination / "initial"
        resumed = destination / "resumed"
        before = initial / "checkpoint_00000001.pth"
        restored = resumed / "checkpoint_00000001.pth"
        after = resumed / "checkpoint_00000002.pth"
        run(
            [*common, "--run-dir", str(initial), "--max-epochs", "1"],
            destination / "initial.log",
        )
        run(
            [
                *common,
                "--run-dir",
                str(resumed),
                "--max-epochs",
                "2",
                "--resume",
                str(before),
            ],
            destination / "resumed.log",
        )
        proof = destination / "resume.json"
        run(
            [
                str(scripts / "check_sapg_resume_progress.py"),
                "--before",
                str(before),
                "--restored",
                str(restored),
                "--after",
                str(after),
                "--output",
                str(proof),
            ],
            destination / "resume_check.log",
        )
        reports = [check_metrics(p / "metrics.jsonl") for p in (initial, resumed)]
        options = [
            json.loads((p / "actual_options.json").read_text())
            for p in (initial, resumed)
        ]
        assert options[0] == options[1], "Physics changes on resume"
        actual = options[0]
        recipe = RECIPES[name]
        for key in ("physics_dt", "decimation", "sdf_initpoints", "nconmax"):
            assert actual[key] == recipe[key], key
        assert actual["native"] == actual["warp"]
        assert actual["num_envs"] == 6 and actual["njmax"] == 2048
        assert [row["transitions"] for row in reports] == [24, 48]
        result = {
            "recipe": name,
            "device": args.device,
            "actual_physics": actual,
            "updates": reports,
            "resume_proof": json.loads(proof.read_text()),
            "task_success_measured": False,
            "passed": True,
        }
        assert result["resume_proof"]["passed"]
        (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.discard_intermediate_checkpoints:
            removed = {}
            for checkpoint in destination.glob("**/*.pth"):
                if checkpoint == after or checkpoint.is_symlink():
                    continue
                removed[str(checkpoint.relative_to(destination))] = sha256(checkpoint)
                checkpoint.unlink()
            (destination / "discarded_intermediates.json").write_text(
                json.dumps(removed, indent=2) + "\n"
            )
        print(json.dumps({"recipe": name, "passed": True, "result": str(destination)}))


if __name__ == "__main__":
    main()
