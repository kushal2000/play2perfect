"""Score first episodes using the public offline reset and scoring protocol.

Native and MJLab use the same manager-compiled model and sampled starts.
All-goals-hit and withdrawal are separate. Incomplete budgets are explicit.
"""

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

wp.config.kernel_cache_dir = os.environ.get(
    "P2P_WARP_CACHE", str(Path(".uv-cache/warp").resolve())
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.recorder_manager import RecorderTermCfg  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply, quat_mul  # noqa: E402

from play2perfect_mjlab.assembly_env_cfg import assembly_cfg, task_contract  # noqa: E402
from play2perfect_mjlab.assets import sha256  # noqa: E402
from play2perfect_mjlab.collision_capacity import configure_epa_horizon  # noqa: E402
from play2perfect_mjlab.evaluation_metrics import (  # noqa: E402
    declared_start_metrics,
    terminal_release_metrics,
)
from play2perfect_mjlab.native import NativeAssembly  # noqa: E402
from play2perfect_mjlab.onnx_actor import OnnxActor  # noqa: E402
from play2perfect_mjlab.rollout_recorder import RolloutRecorder  # noqa: E402
from play2perfect_mjlab.runtime_info import runtime_info, simulation_options  # noqa: E402
from play2perfect_mjlab.torch_actor import PretrainedActor  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--onnx", type=Path)
    p.add_argument("--backend", choices=("native", "mjlab"), default="mjlab")
    p.add_argument(
        "--history",
        choices=("immediate", "source"),
        default="immediate",
        help="Source uses training queue lengths; native supports immediate only",
    )
    p.add_argument(
        "--conditioning",
        choices=("public", "50", "40", "30", "20", "10", "0"),
        default="public",
    )
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Diagnostic: sample the Torch policy's learned Gaussian",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--min-episode-steps", type=int, default=100)
    p.add_argument("--sdf-initpoints", type=int, default=50)
    p.add_argument("--nconmax", type=int, default=512)
    p.add_argument("--njmax", type=int, default=2048)
    p.add_argument("--epa-horizon", type=int, default=24)
    p.add_argument("--solver-iterations", type=int, default=100)
    p.add_argument("--ls-iterations", type=int, default=50)
    p.add_argument("--physics-dt", type=float)
    p.add_argument("--decimation", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--record-every", type=int, default=20)
    args = p.parse_args()
    if args.backend == "native" and args.history != "immediate":
        p.error("Native evaluation supports --history immediate only")
    if args.record_every < 1:
        p.error("--record-every must be positive; use 1 for presentation videos")
    if args.output.exists():
        p.error("Choose a new output file to preserve the existing result")
    if args.trajectory and any(
        args.trajectory.with_suffix(suffix).exists()
        for suffix in (".npz", ".json", ".mjb")
    ):
        p.error("Choose a new trajectory name to preserve the existing recording")
    if args.checkpoint is None and args.onnx is None:
        p.error("Supply a Torch checkpoint or an ONNX actor")
    if args.stochastic and args.onnx:
        p.error("Learned-Gaussian sampling requires a Torch checkpoint")
    if args.trajectory and args.backend != "mjlab":
        p.error("Pose recording currently uses the MJLab recorder hooks")
    if args.trajectory and args.trajectory.suffix != ".npz":
        p.error("Trajectory output must end in .npz")
    if args.trajectory and args.trajectory.with_suffix(".json") == args.output:
        p.error("Result JSON and trajectory metadata must use different filenames")
    script_sha256 = sha256(Path(__file__))
    collision_capacity = configure_epa_horizon(args.epa_horizon)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    tags = (
        torch.linspace(50, 0, args.num_envs)
        if args.conditioning == "public"
        else torch.full((args.num_envs,), float(args.conditioning))
    )
    indices = (
        (tags[:, None] == torch.tensor([50, 40, 30, 20, 10, 0])).float().argmax(-1)
    )
    block_ids = 50 - indices * 10
    if args.onnx:
        actor = OnnxActor(args.onnx, args.device)
        actors = {int(b): actor for b in block_ids.unique()}
    else:
        actors = {
            int(b): PretrainedActor.from_checkpoint(args.checkpoint, int(b))
            .to(args.device)
            .eval()
            for b in block_ids.unique()
        }
    cfg = assembly_cfg(
        args.bundle,
        args.num_envs,
        auto_reset=True,
        sdf_initpoints=args.sdf_initpoints,
        evaluation_preset="public_offline",
        evaluation_history=args.history,
        nconmax=args.nconmax,
        njmax=args.njmax,
        solver_iterations=args.solver_iterations,
        ls_iterations=args.ls_iterations,
        physics_dt=args.physics_dt,
        decimation=args.decimation,
    )
    if args.trajectory:
        cfg.recorders = {
            "rollout": RecorderTermCfg(
                func=RolloutRecorder,
                params={
                    "output": str(args.trajectory),
                    "bundle": str(args.bundle),
                    "every": args.record_every,
                },
            )
        }
    env = ManagerBasedRlEnv(cfg, device=args.device)
    obs, _ = env.reset(seed=args.seed)
    initial_qpos = env.sim.data.qpos[:].cpu().numpy().copy()
    initial_qvel = env.sim.data.qvel[:].cpu().numpy().copy()
    initial_mocap = {
        field: getattr(env.sim.data, field)[:].cpu().numpy().copy()
        for field in ("mocap_pos", "mocap_quat")
    }
    s = env._p2p
    total_goals = len(s["goals"])
    completed = np.zeros(args.num_envs, dtype=bool)
    goals = np.zeros(args.num_envs, dtype=int)
    lengths = np.zeros(args.num_envs, dtype=int)
    ended = np.zeros(args.num_envs, dtype=bool)
    final_error = np.full(args.num_envs, np.nan)
    started = time.monotonic()
    peak_contacts = peak_constraints = 0
    ongoing_goals = np.zeros(args.num_envs, dtype=int)
    ongoing_controls = np.zeros(args.num_envs, dtype=int)
    with torch.no_grad():
        if args.backend == "mjlab":
            hidden = next(iter(actors.values())).initial_state(args.num_envs)
            for step in range(1, args.steps + 1):
                obs["actor"][:, 140] = block_ids.to(args.device)
                action = torch.zeros(args.num_envs, 29, device=args.device)
                for block, actor in actors.items():
                    ids = (block_ids == block).nonzero().flatten().to(args.device)
                    mean, h, c = actor(
                        obs["actor"][ids], hidden[0][:, ids], hidden[1][:, ids]
                    )
                    if args.stochastic:
                        mean = mean + torch.randn_like(mean) * actor.log_std.exp()
                    action[ids], hidden[0][:, ids], hidden[1][:, ids] = mean, h, c
                obs, _, terminated, truncated, _ = env.step(action)
                done = (terminated | truncated).cpu().numpy()
                fresh = done & ~ended
                goals[fresh] = s["episode_goal_count"].cpu().numpy()[fresh]
                completed[fresh] = s["episode_completed"].cpu().numpy()[fresh]
                lengths[fresh] = step
                final_error[fresh] = s["distance"].cpu().numpy()[fresh]
                ended |= done
                ongoing_goals[~ended] = s["goal_index"].cpu().numpy()[~ended]
                ongoing_controls[~ended] = step
                hidden[0][:, terminated | truncated] = 0
                hidden[1][:, terminated | truncated] = 0
                peak_contacts = max(
                    peak_contacts, int(env.sim.wp_data.nacon.numpy()[0])
                )
                peak_constraints = max(
                    peak_constraints, int(env.sim.wp_data.nefc.numpy().max())
                )
                if step % 120 == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "first_episodes_ended": int(ended.sum()),
                                "withdrawn": int(completed.sum()),
                                "first_episode_goals": np.where(
                                    ended, goals, s["goal_index"].cpu().numpy()
                                ).tolist(),
                                "peak_contacts": peak_contacts,
                                "peak_constraints": peak_constraints,
                            }
                        ),
                        flush=True,
                    )
                if ended.all() or time.monotonic() - started > args.max_seconds:
                    break
        else:
            if args.device != "cpu":
                p.error("Native reference uses CPU policy inference")
            native = NativeAssembly(args.bundle, sdf_initpoints=args.sdf_initpoints)
            native.model, native.data = (
                env.sim.mj_model,
                mujoco.MjData(env.sim.mj_model),
            )
            native.palm_id = native.model.body("robot/link7").id
            native.tip_ids = [
                native.model.body(f"robot/palmleft_{f}_DP").id
                for f in ("index", "middle", "ring", "thumb", "pinky")
            ]
            native.goal_id = int(
                native.model.body("goal_object/goal_object").mocapid[0]
            )
            for world in range(args.num_envs):
                native.reset()
                native.metadata.update(cfg.p2p_metadata)
                delta = s["goal_delta_quat"][world].expand(total_goals, -1)
                program = s["goals"]
                positions = (
                    quat_apply(delta, program[:, :3]) + s["goal_delta_pos"][world]
                )
                rotations = quat_mul(delta, program[:, [6, 3, 4, 5]])[:, [1, 2, 3, 0]]
                native.metadata["goals"] = torch.cat(
                    (positions, rotations), -1
                ).tolist()
                native.metadata["goal_position_bias"] = s["bias"][world].tolist()
                native.metadata["block_id"] = int(block_ids[world])
                native.data.qpos[:], native.data.qvel[:] = (
                    initial_qpos[world],
                    initial_qvel[world],
                )
                native.targets = (
                    env.action_manager.get_term("assembly")
                    .targets[world]
                    .numpy()
                    .copy()
                )
                native.data.ctrl[:] = native.targets
                for field, values in initial_mocap.items():
                    getattr(native.data, field)[:] = values[world]
                native.set_goal()
                mujoco.mj_forward(native.model, native.data)
                actor = actors[int(block_ids[world])]
                h, c = actor.initial_state(1)
                for _ in range(args.steps):
                    mean, h, c = actor(
                        torch.from_numpy(native.observation())[None], h, c
                    )
                    if args.stochastic:
                        mean = mean + torch.randn_like(mean) * actor.log_std.exp()
                    native.step(mean[0].numpy())
                    peak_contacts = max(peak_contacts, native.data.ncon)
                    peak_constraints = max(peak_constraints, native.data.nefc)
                    if (
                        native.completed
                        or native.failed
                        or time.monotonic() - started > args.max_seconds
                    ):
                        break
                goals[world], completed[world], lengths[world] = (
                    native.successes,
                    native.completed,
                    native.steps,
                )
                ended[world] = native.completed or native.failed
                ongoing_goals[world] = native.successes
                ongoing_controls[world] = native.steps
                final_error[world] = native.distance
                print(
                    json.dumps(
                        {
                            "world": world,
                            "goals": native.successes,
                            "withdrawn": native.completed,
                            "controls": native.steps,
                        }
                    ),
                    flush=True,
                )
                if time.monotonic() - started > args.max_seconds:
                    break
    valid = ended & (lengths >= args.min_episode_steps)
    result = {
        "scope": (
            "Diagnostic learned-Gaussian actions; public-offline starts and scoring"
            if args.stochastic
            else "Public offline scoring/reset contract; small cohort; "
            "same compiled model for both engines"
        ),
        "action_sampling": "learned_gaussian" if args.stochastic else "mean",
        "backend": args.backend,
        "history_profile": args.history,
        "runtime": runtime_info(),
        "actual_physics": simulation_options(env),
        "collision_capacity": collision_capacity,
        "task_contract": task_contract(cfg),
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "onnx_sha256": sha256(args.onnx) if args.onnx else None,
        "policy_provenance": cfg.p2p_metadata.get("policy_provenance")
        if args.onnx
        else None,
        "bundle_reference_policy": cfg.p2p_metadata.get("policy_provenance"),
        "manifest_sha256": sha256(args.bundle / "manifest.json"),
        "script_sha256": script_sha256,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "conditioning": args.conditioning,
        "effective_block_ids": block_ids.tolist(),
        "initial_qpos": initial_qpos.tolist(),
        "initial_qvel": initial_qvel.tolist(),
        "n_valid_first_episodes": int(valid.sum()),
        "n_early_ends_discarded": int((ended & ~valid).sum()),
        "n_unfinished": int((~ended).sum()),
        "unfinished_goal_counts": {
            str(i): int(ongoing_goals[i]) for i in np.flatnonzero(~ended)
        },
        "unfinished_control_counts": {
            str(i): int(ongoing_controls[i]) for i in np.flatnonzero(~ended)
        },
        "all_goals_hit_rate": float((goals[valid] >= total_goals).mean())
        if valid.any()
        else None,
        "withdrawal_rate": float(completed[valid].mean()) if valid.any() else None,
        "goal_fraction": float((goals[valid] / total_goals).mean())
        if valid.any()
        else None,
        "goal_counts": goals.tolist(),
        "withdrawn": completed.tolist(),
        "controls": lengths.tolist(),
        "final_keypoint_error_m": [
            float(x) if np.isfinite(x) else None for x in final_error
        ],
        "first_episodes_ended": ended.tolist(),
        "wall_seconds": time.monotonic() - started,
        "peak_contacts": peak_contacts,
        "peak_constraints": peak_constraints,
        "contact_scope": (
            "Global Warp contacts / maximum world constraints; after controls"
        ),
        "tf32": False,
    }
    env.close()  # Flush recorder terminal states before scoring physical release.
    if args.trajectory:
        result.update(
            terminal_release_metrics(args.trajectory, completed, lengths, valid)
        )
    result["declared_start_metrics"] = declared_start_metrics(
        goals, completed, valid, total_goals, result.get("contact_free_withdrawn")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("initial_qpos", "initial_qvel", "task_contract", "runtime")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
