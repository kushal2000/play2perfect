"""Capture invalid transitions and reset only the affected simulation worlds."""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch


class TransitionGuard:
    def __init__(self, env, output=None):
        self.env = env
        self.output = Path(output) if output else None
        self.invalid_worlds = 0
        self.events = 0
        self.history = deque(maxlen=2)

    def before_step(self):
        """Retain two control-start states without copying to the host."""
        if self.output is None:
            return
        data = self.env.sim.data
        self.history.append(
            {
                name: getattr(data, name).clone()
                for name in (
                    "qpos",
                    "qvel",
                    "qacc_warmstart",
                    "ctrl",
                    "mocap_pos",
                    "mocap_quat",
                    "qfrc_applied",
                    "xfrc_applied",
                )
            }
        )

    def recover(self, observations, reward, terminated, truncated, infos):
        env, data = self.env, self.env.sim.data
        fields = {f"observation_{k}": v for k, v in observations.items()}
        fields["reward"] = reward
        for name in ("qpos", "qvel", "qacc", "qacc_warmstart"):
            fields[name] = getattr(data, name)
        masks = {
            name: ~torch.isfinite(value).reshape(env.num_envs, -1).all(-1)
            for name, value in fields.items()
        }
        bad = torch.stack(list(masks.values())).any(0)
        infos["invalid_transition"] = bad.clone()
        if not bad.any():
            return observations, reward, terminated, truncated, infos
        ids = bad.nonzero().flatten()
        self.invalid_worlds += len(ids)
        self.events += 1
        event = {
            "event": "invalid_world_reset",
            "control": env.common_step_counter,
            "worlds": ids.cpu().tolist(),
            "components": {k: int(v.sum()) for k, v in masks.items()},
            "cumulative_invalid_worlds": self.invalid_worlds,
            "scope": "State returned by env.step before recovery; failed episodes",
        }
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            if self.events <= 8:
                selected = ids[:5]
                values = {
                    k: v[selected].detach().cpu().numpy() for k, v in fields.items()
                }
                for name in (
                    "ctrl",
                    "mocap_pos",
                    "mocap_quat",
                    "qfrc_applied",
                    "xfrc_applied",
                ):
                    values[name] = getattr(data, name)[selected].cpu().numpy()
                if self.history:
                    for name in self.history[-1]:
                        values["history_" + name] = (
                            torch.stack(
                                [previous[name][selected] for previous in self.history]
                            )
                            .cpu()
                            .numpy()
                        )
                np.savez_compressed(
                    self.output / f"invalid_transition_{self.events:06d}.npz",
                    world_ids=selected.cpu().numpy(),
                    **values,
                )
            with (self.output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        # MJLab partial reset scopes stateful commands and observation histories.
        observations, _ = env.reset(env_ids=ids)
        for values in observations.values():
            if not torch.isfinite(values[ids]).all():
                raise FloatingPointError(
                    "Partial reset did not repair invalid observations"
                )
        reward = reward.clone()
        reward[bad] = 0
        terminated, truncated = terminated.clone(), truncated.clone()
        terminated[bad], truncated[bad] = True, False
        infos["time_outs"] = truncated
        infos["completed"][bad] = False
        infos["recurrent_reset"] |= bad
        infos["reward"] = reward
        return observations, reward, terminated, truncated, infos


def task_progress(env):
    state = env._p2p
    return {
        "common_step_counter": env.common_step_counter,
        **{
            k: state[k]
            for k in (
                "random_goal_tolerance",
                "last_curriculum_update",
                "lift_bonus_active",
            )
            if k in state
        },
        "random_goal_env": state["random_goal_env"].clone(),
    }


def restore_task_progress(env, progress):
    env.common_step_counter = int(progress["common_step_counter"])
    state = env._p2p
    for name in (
        "random_goal_tolerance",
        "last_curriculum_update",
        "lift_bonus_active",
    ):
        if name in progress:
            state[name] = progress[name]
    if "random_goal_env" in progress:
        state["random_goal_env"][:] = progress["random_goal_env"].to(env.device)
    else:
        # One-time migration of legacy checkpoints. Identical CUDA seed/setup
        # reconstructs the fixed mask; fail if even block counts differ.
        actual = (~state["random_goal_env"]).reshape(6, -1).sum(1).cpu().tolist()
        if actual != progress["expected_insertion_world_counts"]:
            raise ValueError("Legacy random-goal mask reconstruction differs")
    state["max_goals"].fill_(len(state["goals"]))
    state["max_goals"][state["random_goal_env"]] = env.cfg.source_randomization.get(
        "random_goal_max_successes", len(state["goals"])
    )
