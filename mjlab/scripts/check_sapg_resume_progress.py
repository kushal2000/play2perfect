"""Extend full learner restoration checks to curriculum and random-number state."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--restored", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    base_output = args.output.with_name(args.output.stem + "_learner.json")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("check_sapg_resume.py")),
            "--before",
            str(args.before),
            "--restored",
            str(args.restored),
            "--after",
            str(args.after),
            "--output",
            str(base_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    before, restored, after = [
        torch.load(path, map_location="cpu", weights_only=False)[0]
        for path in [args.before, args.restored, args.after]
    ]
    initial, loaded, final = [s["mjlab"] for s in [before, restored, after]]
    contract = initial["metadata"]["resume_contract"]
    rollout = contract["num_envs"] * contract["horizon"]
    progress = [s["task_progress"] for s in [initial, loaded, final]]
    rng = {
        name: exact(initial[name], loaded[name])
        for name in ["torch_rng", "numpy_rng", "python_rng", "cuda_rng"]
    }
    optimizer_checks = {}
    for name, get_optimizer in [
        ("actor", lambda s: s["optimizer"]),
        ("critic", lambda s: s["mjlab"]["central_value_optimizer"]),
    ]:
        a, b = [get_optimizer(s)["state"] for s in [before, after]]
        assert a.keys() == b.keys() and a
        advances = [float(b[k]["step"]) - float(a[k]["step"]) for k in a]
        optimizer_checks[name] = dict(
            all_updated=all(value > 0 for value in advances),
            step_advances=sorted(set(advances)),
            finite=all(
                torch.isfinite(value).all().item()
                for state in b.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ),
        )
    checks = dict(
        generic_learner_restore_passed=json.loads(base_output.read_text())["passed"],
        task_progress_exact_on_resume=exact(progress[0], progress[1]),
        random_number_states_exact_on_resume=all(rng.values()),
        task_contract_preserved=all(
            s["metadata"]["resume_contract"] == contract for s in [loaded, final]
        ),
        one_update_advances_epoch=after["epoch"] == before["epoch"] + 1,
        one_update_advances_rollout=after["frame"] == before["frame"] + rollout,
        control_counter_advances_by_horizon=(
            progress[2]["common_step_counter"]
            == progress[0]["common_step_counter"] + contract["horizon"]
        ),
        fixed_random_goal_mask_preserved=exact(
            progress[0]["random_goal_env"], progress[2]["random_goal_env"]
        ),
        both_optimizers_update=all(v["all_updated"] for v in optimizer_checks.values()),
        both_optimizers_finite=all(v["finite"] for v in optimizer_checks.values()),
    )
    report = dict(
        scope=__doc__,
        checks=checks,
        random_number_state_checks=rng,
        optimizer_checks=optimizer_checks,
        task_progress=[
            {k: v for k, v in point.items() if k != "random_goal_env"}
            for point in progress
        ],
        random_goal_worlds=int(progress[0]["random_goal_env"].sum()),
        total_worlds=contract["num_envs"],
        mixed_precision=contract["mixed_precision"],
        cuda_rng_was_saved=initial["cuda_rng"] is not None,
        amp_scale_before=before.get("scaler", {}).get("scale"),
        epochs=[s["epoch"] for s in [before, restored, after]],
        transitions=[s["frame"] for s in [before, restored, after]],
        passed=all(checks.values()),
        generic_report=str(base_output),
        input_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [args.before, args.restored, args.after]
        },
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
