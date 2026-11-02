"""Offline (headless) success-rate evaluation of the released checkpoints.

For each released assembly problem, this runs the policy across a large batch
of parallel envs in Isaac Sim and scores only each env's FIRST episode (so the N
envs are N i.i.d. trials). Envs whose first episode ends within --min-episode-steps
are discarded as unstable inits (object fell / ungraspable at reset → the task was
impossible). It reports the success rate over the remaining valid trials.

Because Isaac Sim / Kit cannot cleanly re-initialize a different env in one
process, the driver evaluates each problem in its own subprocess (the same
pattern as ``eval_isaacsim.py``) and prints a summary table.

Usage:
    # Evaluate all released problems with the downloaded checkpoints
    python evaluation/eval_offline.py --policies-dir pretrained_assembly

    # A single problem
    python evaluation/eval_offline.py \
        --problem tight_insertion \
        --checkpoint-path pretrained_assembly/tight_insertion/model.pth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
DEFAULT_AGENT = "rl_games_sapg_cfg_entry_point"
ALL_PROBLEMS = [
    "tight_insertion",
    "beam_assembly_step1",
    "beam_assembly_step2",
    "screwing",
    "fork_insertion",
    "plug_insertion",
]
_RESULT_PREFIX = "RESULT_JSON "


# ─────────────────────────────────────────────────────────────────────────────
# Worker: evaluate ONE problem in this process (after Kit boots).
# ─────────────────────────────────────────────────────────────────────────────
def run_worker(args) -> int:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = False
    app = AppLauncher(launcher_args).app

    import math as _math

    import torch

    import isaacsimenvs  # noqa: F401  triggers gym.register side effects
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner, _load_checkpoint_weights

    # Reuse the eval building blocks so this stays in lock-step with eval_isaacsim.
    from evaluation.eval_isaacsim import (
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
    )

    result = {"problem": args.problem, "checkpoint": args.checkpoint_path}
    env = None
    try:
        cfg = _load_env_cfg(args.task)
        _apply_env_overrides(
            cfg,
            problem=args.problem,
            goal_mode=args.goal_mode,
            random_goal_fraction=0.0,
            insertion_success_tolerance=args.insertion_success_tolerance,
            retract_success_tolerance=args.retract_success_tolerance,
            num_envs=args.num_envs,
            sim_device=args.sim_device,
            sdf=False,
            keep_dr=False,
            extra_overrides={},
        )
        env = _instantiate_env(args.task, cfg)

        agent_cfg = _configure_agent(
            args.task,
            args.agent,
            rl_device=args.rl_device,
            num_envs=args.num_envs,
            deterministic=True,
            games=10**9,
            extra_overrides={},
        )
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", _math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", _math.inf))
        wrapped = register_rlgames_env(
            env, rl_device=args.rl_device, clip_obs=clip_obs, clip_actions=clip_actions
        )

        runner = Runner()
        runner.load(agent_cfg)
        runner.reset()
        player = runner.create_player()
        player.set_weights(_load_checkpoint_weights(player, str(args.checkpoint_path)))
        player.has_batch_dimension = True

        player.reset()
        obs = player.env_reset(wrapped)

        # Batched eval: all envs run in parallel and we score only each env's
        # FIRST episode, so the N envs are N i.i.d. trials. Per-episode success is
        # read from env._prev_episode_successes, which reset_utils copies from
        # env._successes right before each auto-reset (it survives the reset).
        n = args.num_envs
        dev = env.device
        done_once = torch.zeros(n, dtype=torch.bool, device=dev)
        first_len = torch.zeros(n, dtype=torch.long, device=dev)
        first_full = torch.zeros(n, dtype=torch.bool, device=dev)
        first_ratio = torch.zeros(n, dtype=torch.float, device=dev)
        first_retract = torch.zeros(n, dtype=torch.bool, device=dev)
        has_retract = hasattr(env, "retract_succeeded")

        step = 0
        while step < args.steps and not bool(done_once.all()):
            action = player.get_action(obs, is_deterministic=True)
            obs, _rew, dones, _infos = player.env_step(wrapped, action)
            newly = dones.reshape(-1).to(dev).bool() & ~done_once
            ids = torch.nonzero(newly, as_tuple=False).reshape(-1)
            if ids.numel():
                succ = env._prev_episode_successes[ids]
                maxg = env.prev_episode_env_max_goals[ids].clamp_min(1)
                first_len[ids] = step + 1
                first_full[ids] = succ >= env.prev_episode_env_max_goals[ids]
                first_ratio[ids] = (succ.float() / maxg.float()).clamp(0.0, 1.0)
                # retract_succeeded is reset on auto-reset, so read the pre-reset
                # value the env logs each step in extras["episode_final"].
                rs = env.extras.get("episode_final", {}).get("retract_success") if has_retract else None
                if rs is not None:
                    first_retract[ids] = rs[ids].to(dev) > 0.5
                done_once[ids] = True
            step += 1

        # Exclude envs whose first episode ended almost immediately: those starts
        # were unstable / ungraspable (object fell at reset), so the task was
        # impossible — not a fair trial of the policy.
        valid = done_once & (first_len >= args.min_episode_steps)
        nv = int(valid.sum().item())
        n_done = int(done_once.sum().item())
        result.update(
            num_envs=n,
            n_first_episodes=nv,
            n_unstable_init=n_done - nv,
            success_rate=(float(first_full[valid].float().mean().item()) if nv else 0.0),
            mean_goal_ratio=(float(first_ratio[valid].mean().item()) if nv else 0.0),
            retract_rate=(float(first_retract[valid].float().mean().item()) if (has_retract and nv) else None),
        )
        print(_RESULT_PREFIX + json.dumps(result), flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        del app
    os._exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Driver: one subprocess per problem, then print a summary table.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_checkpoint(args, problem: str) -> Path | None:
    if args.checkpoint_path:
        return Path(args.checkpoint_path)
    cand = Path(args.policies_dir) / problem / "model.pth"
    if not cand.is_absolute():
        cand = REPO_ROOT / cand
    return cand if cand.is_file() else None


def run_driver(args) -> int:
    problems = [args.problem] if args.problem else list(ALL_PROBLEMS)
    rows = []
    for problem in problems:
        ckpt = _resolve_checkpoint(args, problem)
        if ckpt is None or not ckpt.is_file():
            print(f"[skip] {problem}: no checkpoint (looked in {args.policies_dir}/{problem}/model.pth)")
            rows.append({"problem": problem, "error": "missing checkpoint"})
            continue
        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
            "--problem", problem,
            "--checkpoint-path", str(ckpt),
            "--task", args.task,
            "--agent", args.agent,
            "--num-envs", str(args.num_envs),
            "--steps", str(args.steps),
            "--min-episode-steps", str(args.min_episode_steps),
            "--rl-device", args.rl_device,
            "--sim-device", args.sim_device,
            "--goal-mode", args.goal_mode,
            "--insertion-success-tolerance", str(args.insertion_success_tolerance),
            "--retract-success-tolerance", str(args.retract_success_tolerance),
        ]
        env = os.environ.copy()
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        print(f"\n===== evaluating {problem} =====", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                              stdout=subprocess.PIPE, stderr=None, text=True)
        row = {"problem": problem, "error": "no result"}
        for line in proc.stdout.splitlines():
            print(line)
            if line.startswith(_RESULT_PREFIX):
                row = json.loads(line[len(_RESULT_PREFIX):])
        rows.append(row)

    print("\n" + "=" * 86)
    print(f"{'problem':28s} {'trials':>8s} {'bad_init':>9s} {'success':>9s} {'goal_ratio':>11s} {'retract':>9s}")
    print("-" * 86)
    for r in rows:
        if "success_rate" not in r:
            print(f"{r['problem']:28s} {'-':>8s} {'-':>9s} {'-':>9s} {'-':>11s} {'-':>9s}  ({r.get('error','?')})")
            continue
        retr = "-" if r.get("retract_rate") is None else f"{r['retract_rate']*100:6.1f}%"
        print(f"{r['problem']:28s} {r['n_first_episodes']:>8d} {r.get('n_unstable_init', 0):>9d} "
              f"{r['success_rate']*100:>8.1f}% {r['mean_goal_ratio']*100:>10.1f}% {retr:>9s}")
    print("=" * 86)
    print("trials = valid first episodes; bad_init = discarded unstable-init starts;")
    print("success = first episode reached all subgoals (over valid trials).")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", default=None, help="Single problem to eval (default: all released).")
    p.add_argument("--policies-dir", default="pretrained_assembly",
                   help="Dir of <problem>/model.pth checkpoints (driver mode).")
    p.add_argument("--checkpoint-path", default=None, help="Explicit checkpoint (single problem).")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--agent", default=DEFAULT_AGENT)
    p.add_argument("--num-envs", type=int, default=1024,
                   help="Parallel envs = number of i.i.d. first-episode trials.")
    p.add_argument("--steps", type=int, default=2500,
                   help="Max control steps; loop ends early once every env finishes episode 1. "
                        "Higher so long tasks (e.g. screwing, 10 subgoals) finish.")
    p.add_argument("--min-episode-steps", type=int, default=100,
                   help="Discard first episodes that end within this many steps as unstable "
                        "inits (object fell/ungraspable at reset → impossible trial).")
    p.add_argument("--goal-mode", default="preInsertAndFinal")
    p.add_argument("--insertion-success-tolerance", type=float, default=0.01)
    p.add_argument("--retract-success-tolerance", type=float, default=0.005)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.worker:
        return run_worker(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
