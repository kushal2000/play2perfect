"""Verify exact learner restoration, fresh rollout state, and one resumed update."""

import argparse
import json
from pathlib import Path

import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--restored", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    initial, restored, final = [
        torch.load(path, map_location="cpu", weights_only=False)[0]
        for path in (args.before, args.restored, args.after)
    ]
    report = {
        "actor_exact_on_resume": equal(initial["model"], restored["model"]),
        "critic_exact_on_resume": equal(
            initial["assymetric_vf_nets"], restored["assymetric_vf_nets"]
        ),
        "actor_optimizer_exact_on_resume": equal(
            initial["optimizer"], restored["optimizer"]
        ),
        "critic_optimizer_exact_on_resume": equal(
            initial["mjlab"]["central_value_optimizer"],
            restored["mjlab"]["central_value_optimizer"],
        ),
        "grad_scaler_exact_on_resume": equal(
            initial.get("scaler"), restored.get("scaler")
        ),
        "environment_parameters_exact_on_resume": equal(
            initial.get("env_state"), restored.get("env_state")
        ),
        "fresh_observation_on_resume": restored["obs"] is None,
        "fresh_hidden_on_resume": all(not t.any() for t in restored["rnn_states"]),
        "epochs": [w["epoch"] for w in (initial, restored, final)],
        "transitions": [w["frame"] for w in (initial, restored, final)],
        "central_value_epochs": [
            w["mjlab"]["central_value_epoch"] for w in (initial, restored, final)
        ],
        "actor_value_head_changed": not torch.equal(
            initial["model"]["a2c_network.value.weight"],
            final["model"]["a2c_network.value.weight"],
        ),
        "finite_parameters": all(
            torch.isfinite(t).all().item()
            for key in ("model", "assymetric_vf_nets")
            for t in final[key].values()
        ),
        "actor_optimizer_steps": sorted(
            {float(v["step"]) for v in final["optimizer"]["state"].values()}
        ),
        "critic_optimizer_steps": sorted(
            {
                float(v["step"])
                for v in final["mjlab"]["central_value_optimizer"]["state"].values()
            }
        ),
        "grad_scaler_scales": [
            w.get("scaler", {}).get("scale") for w in (initial, restored, final)
        ],
    }
    passed = all(value for value in report.values() if isinstance(value, bool))
    report["passed"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
