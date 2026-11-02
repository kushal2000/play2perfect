#!/usr/bin/env python3
"""Interactive (viser) evaluation of a precise-assembly policy in Isaac Sim.

The parent process owns the viser GUI (see ``evaluation/_viser_demo.py``) and
policy/problem selection. Each rollout runs in an Isaac Sim worker process
launched with ``--python`` (default: ``.venv_isaacsim/bin/python``), which streams
poses back to the viewer. Open ``http://localhost:<port>`` to watch the rollout.

Examples:
    # Single checkpoint for one problem
    python evaluation/eval_isaacsim.py \
        --checkpoint-path pretrained_assembly/tight_insertion/model.pth \
        --problem tight_insertion --port 8080

    # All downloaded policies (a GUI dropdown switches between them)
    python evaluation/eval_isaacsim.py --policies-dir pretrained_assembly
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, replace
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
DEFAULT_AGENT = "rl_games_sapg_cfg_entry_point"
DEFAULT_PROBLEM = "tight_insertion"
# Robot -> precise-assembly task, and the directory its per-problem checkpoints live in.
ROBOT_TASKS = {
    "kuka_sharpa": DEFAULT_TASK,
    "flexiv_wuji_right_human_like": "Isaacsimenvs-PreciseAssemblyFlexivWujiRight-Direct-v0",
}
DEFAULT_ROBOT_POLICY_DIRS = {
    "kuka_sharpa": "pretrained_assembly",
    "flexiv_wuji_right_human_like": "pretrained_assembly_flexiv_wuji",
}
DEFAULT_RANDOM_GOAL_SUCCESS_TOLERANCE = 0.01
GOAL_MODES = ("preInsertAndFinal", "finalGoalOnly")
CONTROL_DT = 1.0 / 60.0


SDF_PHYSX_OVERRIDES = {
    # URDF <sdf> collision tags are honored by the asset converter by default.
    # This flag is retained for older launch scripts and now resolves to the
    # same UWLab-style PhysX defaults used by the PreciseAssembly task config.
    "env.sim.physx.min_position_iteration_count": 1,
    "env.sim.physx.max_position_iteration_count": 192,
    "env.sim.physx.min_velocity_iteration_count": 0,
    "env.sim.physx.max_velocity_iteration_count": 1,
    "env.sim.physx.bounce_threshold_velocity": 0.02,
    "env.sim.physx.friction_offset_threshold": 0.01,
    "env.sim.physx.friction_correlation_distance": 0.0005,
    "env.sim.physx.gpu_found_lost_aggregate_pairs_capacity": 4194304,
    "env.sim.physx.gpu_total_aggregate_pairs_capacity": 8388608,
    "env.sim.physx.gpu_max_rigid_contact_count": 8388608,
    "env.sim.physx.gpu_max_rigid_patch_count": 8388608,
    "env.sim.physx.gpu_collision_stack_size": 2147483648,
}


EVAL_DR_OFF_OVERRIDES = {
    # reset_position_noise_x/y/z are intentionally left at training values so
    # eval rollouts see the same start-pose variation the policy was trained
    # under. Set fixed_start_pose if you want a deterministic start pose.
    "env.reset.reset_dof_pos_random_interval_arm": 0.0,
    "env.reset.reset_dof_pos_random_interval_fingers": 0.0,
    "env.reset.reset_dof_vel_random_interval": 0.0,
    "env.reset.table_reset_z_range": 0.0,
    "env.domain_randomization.use_obs_delay": False,
    "env.domain_randomization.use_action_delay": False,
    "env.domain_randomization.use_object_state_delay_noise": False,
    "env.domain_randomization.object_state_xyz_noise_std": 0.0,
    "env.domain_randomization.object_state_rotation_noise_degrees": 0.0,
    "env.domain_randomization.object_scale_noise_multiplier_range": [1.0, 1.0],
    "env.domain_randomization.joint_velocity_obs_noise_std": 0.0,
    "env.domain_randomization.force_scale": 0.0,
    "env.domain_randomization.torque_scale": 0.0,
}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    checkpoint: Path
    config: Path | None
    metadata: dict
    task: str = DEFAULT_TASK

    @property
    def problem(self) -> str | None:
        problem = self.metadata.get("problem")
        if not problem or str(problem) == "pretrained_policy":
            return None
        return str(problem)

    @property
    def dir_name(self) -> str:
        """The policy's subdir name (``pretrained_assembly/<problem>/`` names it after its problem)."""
        return self.checkpoint.parent.name


class PopenAdapter:
    """Small adapter so the viser demo's viewer lifecycle can manage Popen."""

    def __init__(self, popen: subprocess.Popen):
        self.popen = popen
        self.pid = popen.pid

    def is_alive(self) -> bool:
        return self.popen.poll() is None

    def join(self, timeout: float | None = None) -> None:
        try:
            self.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def kill(self) -> None:
        self.popen.kill()


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _default_isaacsim_python() -> str:
    current = Path(sys.executable)
    if "venv_isaacsim" in str(current):
        return str(current)
    candidate = REPO_ROOT / ".venv_isaacsim" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return str(current)


def _default_policies_dir() -> Path | None:
    """Default policy directory: the downloaded assembly checkpoints, if present.

    ``download_checkpoints.py`` populates ``pretrained_assembly/`` with one
    subfolder per problem (each containing ``model.pth`` + ``config.yaml``), which is
    exactly the layout ``discover_policies`` expects.
    """
    root = REPO_ROOT / "pretrained_assembly"
    return root if root.is_dir() else None


def _load_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def discover_policies(policies_dir: Path) -> dict[str, PolicySpec]:
    policies = {}
    for subdir in sorted(p for p in policies_dir.iterdir() if p.is_dir()):
        checkpoint = subdir / "model.pth"
        if not checkpoint.is_file():
            continue
        config = subdir / "config.yaml"
        metadata = _load_metadata(subdir / "metadata.json")
        policies[subdir.name] = PolicySpec(
            name=subdir.name,
            checkpoint=checkpoint,
            config=config if config.is_file() else None,
            metadata=metadata,
        )
    return policies


def _choose_policy(policies: dict[str, PolicySpec], name: str | None) -> PolicySpec:
    if name is not None:
        if name not in policies:
            raise KeyError(
                f"Unknown policy {name!r}. Available policies: {sorted(policies)}"
            )
        return policies[name]
    if "pretrained_policy" in policies:
        return policies["pretrained_policy"]
    if len(policies) == 1:
        return next(iter(policies.values()))
    return policies[sorted(policies)[0]]


def _iter_policy_lines(policies: Iterable[PolicySpec]) -> Iterable[str]:
    for policy in policies:
        problem = policy.problem or "-"
        yield f"{policy.name:48s} problem={problem} checkpoint={policy.checkpoint}"


def _coerce_override_value(raw):
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    if raw == "True":
        return True
    if raw == "False":
        return False
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _set_attr_path(root, key: str, value) -> None:
    path = key[4:] if key.startswith("env.") else key
    target = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], _coerce_override_value(value))


def _set_dict_path(root: dict, key: str, value) -> None:
    path = key[6:] if key.startswith("agent.") else key
    target = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = _coerce_override_value(value)


def _wxyz_to_xyzw_np(quat):
    return quat[..., [1, 2, 3, 0]]


def _tensor_bool(value) -> bool:
    return bool(value.detach().cpu().item())


def _tensor_float(value) -> float:
    return float(value.detach().cpu().item())


def _tensor_int(value) -> int:
    return int(value.detach().cpu().item())


def _done0(dones) -> bool:
    if hasattr(dones, "detach"):
        return bool(dones.reshape(-1)[0].detach().cpu().item())
    return bool(dones[0])


def _load_env_cfg(task: str):
    import gymnasium as gym
    import yaml
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    spec = gym.spec(task)
    yaml_path = Path(spec.kwargs["env_cfg_yaml_entry_point"])
    with yaml_path.open() as f:
        overlay = yaml.safe_load(f) or {}
    overlay.pop("clip_observations", None)
    overlay.pop("clip_actions", None)
    cfg.from_dict(overlay)
    return cfg


def _instantiate_env(task: str, cfg):
    import importlib
    import gymnasium as gym

    spec = gym.spec(task)
    mod_name, cls_name = spec.entry_point.split(":")
    env_cls = getattr(importlib.import_module(mod_name), cls_name)
    return env_cls(cfg=cfg)


def _apply_env_overrides(
    cfg,
    *,
    problem: str,
    goal_mode: str,
    random_goal_fraction: float,
    insertion_success_tolerance: float | None,
    retract_success_tolerance: float,
    num_envs: int,
    sim_device: str,
    sdf: bool,
    keep_dr: bool,
    extra_overrides: dict[str, object],
) -> None:
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = sim_device
    cfg.precise_assembly.problem = problem
    cfg.precise_assembly.goal_mode = goal_mode
    cfg.precise_assembly.random_goal_fraction = float(random_goal_fraction)
    if insertion_success_tolerance is not None:  # None: the problem's own default
        cfg.precise_assembly.insertion_success_tolerance = float(insertion_success_tolerance)
    cfg.precise_assembly.retract_success_tolerance = float(retract_success_tolerance)
    if float(random_goal_fraction) > 0.0:
        cfg.termination.success_tolerance = DEFAULT_RANDOM_GOAL_SUCCESS_TOLERANCE
        cfg.termination.target_success_tolerance = DEFAULT_RANDOM_GOAL_SUCCESS_TOLERANCE
    cfg.termination.force_consecutive_near_goal_steps = True

    merged = {}
    if not keep_dr:
        merged.update(EVAL_DR_OFF_OVERRIDES)
    if sdf:
        merged.update(SDF_PHYSX_OVERRIDES)
    merged.update(extra_overrides)
    for key, value in merged.items():
        if key.startswith("agent.") or key.startswith("hydra."):
            continue
        _set_attr_path(cfg, key, value)


def _configure_agent(
    task: str,
    agent: str,
    *,
    rl_device: str,
    num_envs: int,
    deterministic: bool,
    games: int,
    extra_overrides: dict[str, object],
) -> dict:
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(task, agent)
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    agent_cfg["params"]["config"]["num_actors"] = int(num_envs)
    player_cfg = agent_cfg["params"]["config"].setdefault("player", {})
    player_cfg["games_num"] = int(games)
    player_cfg["deterministic"] = bool(deterministic)
    player_cfg["print_stats"] = False
    for key, value in extra_overrides.items():
        if key.startswith("agent."):
            _set_dict_path(agent_cfg, key, value)
    return agent_cfg


def _sim_get_state(env, done_pending: bool = False):
    import numpy as np
    import torch
    from isaaclab.utils.math import quat_apply
    from isaacsimenvs.tasks.play.utils.obs_utils import compute_intermediate_values

    compute_intermediate_values(env)
    env_id = 0
    origin = env.scene.env_origins[env_id]

    perm = env._perm_lab_to_canon
    joint_pos = env.robot.data.joint_pos[env_id, perm].detach().cpu().numpy()

    obj_pos = env.object.data.root_pos_w[env_id] - origin
    obj_quat = env.object.data.root_quat_w[env_id]
    goal_pos = env.goal_viz.data.root_pos_w[env_id] - origin
    goal_quat = env.goal_viz.data.root_quat_w[env_id]

    # Match the viser demo's visual convention: it streams obj_keypoint_pos and
    # goal_keypoint_pos, which are the object-size keypoints even when reward /
    # success use the fixed-size keypoint distance.
    kp_offsets = env._keypoint_offsets[env_id]
    if hasattr(env, "_object_scale_multiplier"):
        kp_offsets = kp_offsets * env._object_scale_multiplier[env_id].unsqueeze(0)
    num_kp = kp_offsets.shape[0]
    obj_kps = obj_pos.unsqueeze(0) + quat_apply(
        obj_quat.unsqueeze(0).expand(num_kp, -1), kp_offsets
    )
    goal_kps = goal_pos.unsqueeze(0) + quat_apply(
        goal_quat.unsqueeze(0).expand(num_kp, -1), kp_offsets
    )

    retract_phase = _tensor_bool(env.retract_phase[env_id])
    if retract_phase:
        success_tolerance = (
            float(env.cfg.precise_assembly.retract_success_tolerance)
            * float(env.cfg.reward.keypoint_scale)
        )
    elif hasattr(env, "_keypoint_success_tolerance_m"):
        success_tolerance = _tensor_float(env._keypoint_success_tolerance_m()[env_id])
    else:
        success_tolerance = (
            _tensor_float(env._current_success_tolerance)
            * float(env.cfg.reward.keypoint_scale)
        )

    if hasattr(env, "hole_pos"):
        hole_pos_t = env.hole_pos[env_id]
        if hasattr(env, "hole_quat_wxyz"):
            hole_quat_wxyz_t = env.hole_quat_wxyz[env_id]
        else:
            hole_quat_wxyz_t = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=hole_pos_t.device, dtype=hole_pos_t.dtype
            )
        hole_quat_xyzw_t = _wxyz_to_xyzw_np(hole_quat_wxyz_t.unsqueeze(0))[0]
        hole_pose = torch.cat([hole_pos_t, hole_quat_xyzw_t]).detach().cpu().numpy()
    else:
        hole_pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    is_random_goal = (
        _tensor_bool(env.is_random_goal_env[env_id])
        if hasattr(env, "is_random_goal_env")
        else False
    )

    obj_pose = torch.cat([obj_pos, _wxyz_to_xyzw_np(obj_quat.unsqueeze(0))[0]])
    goal_pose = torch.cat([goal_pos, _wxyz_to_xyzw_np(goal_quat.unsqueeze(0))[0]])

    return (
        joint_pos,                                            # 0
        obj_pose.detach().cpu().numpy(),                      # 1
        goal_pose.detach().cpu().numpy(),                     # 2
        obj_kps.detach().cpu().numpy(),                       # 3
        goal_kps.detach().cpu().numpy(),                      # 4
        retract_phase,                                        # 5
        _tensor_bool(env.retract_succeeded[env_id]),          # 6
        float(env._curr_fingertip_distances[env_id].mean().detach().cpu().item()),
        _tensor_float(env._keypoints_max_dist[env_id]),       # 8
        float(success_tolerance),                             # 9
        _tensor_int(env._near_goal_steps[env_id]),            # 10
        _tensor_int(env.episode_length_buf[env_id]),          # 11
        int(env.max_episode_length),                          # 12
        bool(done_pending),                                   # 13
        np.zeros(3, dtype=np.float32),                        # 14
        hole_pose,                                            # 15 (x,y,z,qx,qy,qz,qw)
        is_random_goal,                                       # 16
    )


def _sim_episode(conn, env, wrapped, player, *, deterministic: bool):
    player.reset()
    obs = player.env_reset(wrapped)

    retract_ok = False
    peak_successes = 0
    max_goals_seen = max(1, _tensor_int(env.env_max_goals[0]))
    step = 0
    done = False
    paused = False

    while not done:
        while conn.poll(0):
            cmd = conn.recv()
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return
            elif cmd == "quit":
                return "quit"

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()
        action = player.get_action(obs, is_deterministic=deterministic)
        obs, _rew, dones, _infos = player.env_step(wrapped, action)
        done = _done0(dones)
        step += 1

        cur_succ = _tensor_int(env._successes[0])
        cur_max = _tensor_int(env.env_max_goals[0])
        peak_successes = max(peak_successes, cur_succ)
        max_goals_seen = max(max_goals_seen, cur_max)
        retract_ok = retract_ok or _tensor_bool(env.retract_succeeded[0])
        state = _sim_get_state(env, done_pending=done)
        conn.send(("state", state, cur_succ, cur_max, step, retract_ok))

        sleep = CONTROL_DT - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

    goal_pct = 100.0 * peak_successes / max(1, max_goals_seen)
    conn.send(("done", goal_pct, step, retract_ok))
    return None


def sim_worker(
    conn,
    *,
    checkpoint_path: str,
    task: str,
    agent: str,
    problem: str,
    goal_mode: str,
    random_goal_fraction: float,
    insertion_success_tolerance: float | None,
    retract_success_tolerance: float,
    num_envs: int,
    games: int,
    rl_device: str,
    sim_device: str,
    deterministic: bool,
    headless: bool,
    sdf: bool,
    keep_dr: bool,
    extra_overrides: dict[str, object],
) -> None:
    app = None
    env = None
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

        from isaaclab.app import AppLauncher

        launcher_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(launcher_parser)
        launcher_args, _ = launcher_parser.parse_known_args([])
        launcher_args.headless = bool(headless)
        launcher_args.enable_cameras = False
        app = AppLauncher(launcher_args).app

        import math as _math

        import isaacsimenvs  # noqa: F401  triggers gym.register side effects
        from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
        from rl_games.torch_runner import Runner, _load_checkpoint_weights

        cfg = _load_env_cfg(task)
        _apply_env_overrides(
            cfg,
            problem=problem,
            goal_mode=goal_mode,
            random_goal_fraction=random_goal_fraction,
            insertion_success_tolerance=insertion_success_tolerance,
            retract_success_tolerance=retract_success_tolerance,
            num_envs=num_envs,
            sim_device=sim_device,
            sdf=sdf,
            keep_dr=keep_dr,
            extra_overrides=extra_overrides,
        )
        env = _instantiate_env(task, cfg)

        agent_cfg = _configure_agent(
            task,
            agent,
            rl_device=rl_device,
            num_envs=num_envs,
            deterministic=deterministic,
            games=games,
            extra_overrides=extra_overrides,
        )
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", _math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", _math.inf))
        wrapped = register_rlgames_env(
            env, rl_device=rl_device, clip_obs=clip_obs, clip_actions=clip_actions
        )

        runner = Runner()
        runner.load(agent_cfg)
        runner.reset()
        player = runner.create_player()
        weights = _load_checkpoint_weights(player, str(checkpoint_path))
        player.set_weights(weights)
        print(f"=> initialized player weights from '{checkpoint_path}'", flush=True)
        player.has_batch_dimension = True

        player.reset()
        obs = player.env_reset(wrapped)
        del obs
        conn.send(("ready", _sim_get_state(env)))

        while True:
            cmd = conn.recv()
            if cmd == "run":
                result = _sim_episode(
                    conn, env, wrapped, player, deterministic=deterministic
                )
                if result == "quit":
                    break
            elif cmd == "quit":
                break
    except Exception as exc:
        try:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
        except Exception:
            pass
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            if app is not None:
                del app
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()


def _worker_main(args) -> int:
    conn = Client(
        (args.worker_host, int(args.worker_port)),
        authkey=bytes.fromhex(args.worker_authkey),
    )
    extra_overrides = json.loads(args.override_json or "{}")
    sim_worker(
        conn,
        checkpoint_path=args.checkpoint_path,
        task=args.task,
        agent=args.agent,
        problem=args.problem,
        goal_mode=args.goal_mode,
        random_goal_fraction=args.random_goal_fraction,
        insertion_success_tolerance=args.insertion_success_tolerance,
        retract_success_tolerance=args.retract_success_tolerance,
        num_envs=args.num_envs,
        games=args.games,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        deterministic=args.deterministic,
        headless=args.worker_headless,
        sdf=args.sdf,
        keep_dr=args.keep_dr,
        extra_overrides=extra_overrides,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--python", default=None, help="Python executable for the Isaac Sim worker.")
    parser.add_argument("--policies-dir", default=None, help="Policies for --task (one robot).")
    parser.add_argument(
        "--robot-policies",
        nargs="+",
        default=None,
        metavar="ROBOT=DIR",
        help=f"Per-robot policy dirs (robots: {', '.join(ROBOT_TASKS)}); a Robot dropdown "
        "switches between them. Default, with no policy args: whichever of "
        f"{', '.join(DEFAULT_ROBOT_POLICY_DIRS.values())} exist.",
    )
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--policy", default=None, help="Alias for --initial-policy.")
    parser.add_argument("--initial-policy", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="preInsertAndFinal")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--random-goal-fraction", type=float, default=0.0)
    parser.add_argument("--insertion-success-tolerance", type=float, default=None,
                        help="default: the problem's own (0.01; screwing 0.005)")
    parser.add_argument("--retract-success-tolerance", type=float, default=0.005)
    parser.add_argument("--eval-success-tolerance", type=float, default=None)
    parser.add_argument("--sdf", action="store_true")
    parser.add_argument("--keep-dr", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--list-policies", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--override",
        nargs=2,
        action="append",
        default=[],
        metavar=("KEY", "VALUE"),
        help="Extra env/agent override, e.g. --override env.reward.lifting_bonus 0.0",
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--worker-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-authkey", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--override-json", default="{}", help=argparse.SUPPRESS)
    return parser


def _robot_policy_dirs(args) -> dict[str, Path] | None:
    """Robot -> policy dir for the multi-robot viewer, or None for the single-task modes."""
    if args.robot_policies:
        dirs = {}
        for item in args.robot_policies:
            robot, sep, path = item.partition("=")
            if not sep or robot not in ROBOT_TASKS:
                raise ValueError(f"--robot-policies wants ROBOT=DIR with ROBOT in {sorted(ROBOT_TASKS)}; got {item!r}")
            dirs[robot] = _resolve_path(path)
        return dirs
    if args.policies_dir or args.checkpoint_path:
        return None
    dirs = {
        robot: REPO_ROOT / rel
        for robot, rel in DEFAULT_ROBOT_POLICY_DIRS.items()
        if (REPO_ROOT / rel).is_dir()
    }
    return dirs if len(dirs) > 1 else None


def _collect_policies(args) -> tuple[dict[str, PolicySpec], PolicySpec | None]:
    selected_policy = None
    policies: dict[str, PolicySpec] = {}
    robot_dirs = _robot_policy_dirs(args)
    if robot_dirs is not None:
        for robot, robot_dir in robot_dirs.items():
            if not robot_dir.is_dir():
                raise FileNotFoundError(f"policies dir not found: {robot_dir}")
            for name, spec in discover_policies(robot_dir).items():
                key = f"{robot}/{name}"
                policies[key] = replace(spec, name=key, task=ROBOT_TASKS[robot])
        if args.policies:
            keep = set(args.policies)
            policies = {name: spec for name, spec in policies.items() if name in keep}
        if not policies:
            raise FileNotFoundError(f"no policy subdirs with model.pth in {sorted(map(str, robot_dirs.values()))}")
        initial = args.initial_policy or args.policy or next(
            (name for name in policies if name.endswith(f"/{DEFAULT_PROBLEM}")), None
        )
        return policies, _choose_policy(policies, initial)

    policies_dir = _resolve_path(args.policies_dir) if args.policies_dir else None
    if policies_dir is None and args.checkpoint_path is None:
        policies_dir = _default_policies_dir()

    if policies_dir is not None:
        if not policies_dir.is_dir():
            raise FileNotFoundError(f"policies dir not found: {policies_dir}")
        policies = {
            name: replace(spec, task=args.task)
            for name, spec in discover_policies(policies_dir).items()
        }
        if args.policies:
            keep = set(args.policies)
            policies = {name: spec for name, spec in policies.items() if name in keep}
        if not policies:
            raise FileNotFoundError(f"no policy subdirs with model.pth in {policies_dir}")
        selected_policy = _choose_policy(
            policies, args.initial_policy or args.policy
        )
    elif args.checkpoint_path is not None:
        checkpoint = _resolve_path(args.checkpoint_path)
        name = checkpoint.parent.name or "policy"
        spec = PolicySpec(name=name, checkpoint=checkpoint, config=None, metadata={}, task=args.task)
        policies[name] = spec
        selected_policy = spec
    return policies, selected_policy


def _run_viewer(args) -> int:
    import evaluation._viser_demo as gym_eval

    policies, selected_policy = _collect_policies(args)
    if args.list_policies:
        if not policies:
            print("No --policies-dir supplied and no hardware rollout policy dir found.")
            return 1
        robot_dirs = _robot_policy_dirs(args)
        if robot_dirs is not None:
            print("Policies in " + ", ".join(f"{r}={d}" for r, d in robot_dirs.items()) + ":")
        else:
            policy_dir = Path(args.policies_dir) if args.policies_dir else _default_policies_dir()
            print(f"Policies in {policy_dir}:")
        for line in _iter_policy_lines(policies[name] for name in sorted(policies)):
            print("  " + line)
        return 0
    if not policies or selected_policy is None:
        raise ValueError("provide --checkpoint-path or --policies-dir")
    if not selected_policy.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {selected_policy.checkpoint}")

    from evaluation.problems import PROBLEM_REGISTRY

    # pretrained_assembly/<problem>/ subdirs are named after their problem.
    policy_problem = selected_policy.problem or (
        selected_policy.dir_name if selected_policy.dir_name in PROBLEM_REGISTRY else None
    )
    initial_problem = args.problem or policy_problem or DEFAULT_PROBLEM
    extra_overrides = {
        key: _coerce_override_value(value)
        for key, value in args.override
    }
    if args.eval_success_tolerance is not None:
        extra_overrides["env.termination.eval_success_tolerance"] = float(
            args.eval_success_tolerance
        )

    viewer_policies = {
        name: (str(spec.config or ""), str(spec.checkpoint))
        for name, spec in policies.items()
    }

    class IsaacSimPegDynamicDemo(gym_eval.PegDynamicDemo):
        def __init__(self, *demo_args, **demo_kwargs):
            self.task = args.task
            self.agent = args.agent
            self.worker_python = args.python or _default_isaacsim_python()
            self.num_envs = int(args.num_envs)
            self.games = int(args.games)
            self.rl_device = args.rl_device
            self.sim_device = args.sim_device
            self.deterministic = bool(args.deterministic)
            self.sdf = bool(args.sdf)
            self.keep_dr = bool(args.keep_dr)
            super().__init__(*demo_args, **demo_kwargs)
            self._insertion_tol_override = args.insertion_success_tolerance
            self._on_problem_change()
            self._sl_retract_tol.value = float(args.retract_success_tolerance)

        def _load_env(self):
            problem_name = self._dd_problem.value
            goal_mode = self._dd_goal_mode.value
            rgf = self._sl_rgf.value
            insertion_tol = float(self._sl_insertion_tol.value)
            retract_tol = float(self._sl_retract_tol.value)
            policy_name = self.policy_name
            if policy_name is None:
                self._md_status.content = (
                    f"**Status:** No policy for {self.viewer_robot.name} / {problem_name}"
                )
                return
            _config_path, checkpoint_path = self.policies[policy_name]

            try:
                self._set_problem_assets(problem_name)
                self._reload_problem_meshes()
            except Exception as exc:
                self._md_status.content = (
                    f"**Status:** Problem load error -- {str(exc)[:200]}"
                )
                print(
                    f"[isaacsim launcher] Problem load error for {problem_name!r}:\n"
                    f"{traceback.format_exc()}"
                )
                return

            self._kill_subprocess()

            label = (
                f"{problem_name} | {policy_name} | goals: {goal_mode} | "
                f"rgf: {rgf:.1f} | ins: {insertion_tol * 1000:.1f}mm | "
                f"ret: {retract_tol * 1000:.1f}mm"
            )
            self._md_status.content = f"**Status:** Loading Isaac Sim *{label}* ..."
            self._md_task.content = f"**Task:** Isaac Sim | {label}"
            self._md_retract.content = "**Retract:** --"
            self._md_hole.content = "**Hole pos:** --"
            self._md_object_pose.content = "**Object pose:** --"
            self._md_goal_pose.content = "**Goal pose:** --"
            self._md_pose_delta.content = "**Object-goal z dist:** --"
            self.ep_count = 0
            self._peak_force = 0.0
            self._md_stats.content = "**Stats:** No episodes yet"

            self.set_robot_joints(self.viewer_robot.default_dof_pos)
            self._setup_scene_objects()

            worker_overrides = dict(self.extra_overrides)
            authkey = os.urandom(16)
            listener = Listener(("127.0.0.1", 0), authkey=authkey)
            host, port = listener.address
            listener._listener._socket.settimeout(20.0)

            cmd = [
                self.worker_python,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-host",
                str(host),
                "--worker-port",
                str(port),
                "--worker-authkey",
                authkey.hex(),
                "--task",
                self.task,
                "--agent",
                self.agent,
                "--checkpoint-path",
                str(checkpoint_path),
                "--problem",
                problem_name,
                "--goal-mode",
                goal_mode,
                "--num-envs",
                str(self.num_envs),
                "--games",
                str(self.games),
                "--random-goal-fraction",
                str(float(rgf)),
                "--insertion-success-tolerance",
                str(insertion_tol),
                "--retract-success-tolerance",
                str(retract_tol),
                "--rl-device",
                self.rl_device,
                "--sim-device",
                self.sim_device,
                "--override-json",
                json.dumps(worker_overrides),
            ]
            if self.deterministic:
                cmd.append("--deterministic")
            if self.sdf:
                cmd.append("--sdf")
            if self.keep_dr:
                cmd.append("--keep-dr")
            if self.headless:
                cmd.append("--worker-headless")

            env = os.environ.copy()
            env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            try:
                popen = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
                self._proc = PopenAdapter(popen)
                self._conn = listener.accept()
            except Exception as exc:
                listener.close()
                if self._proc is not None:
                    self._proc.kill()
                    self._proc.join(timeout=2)
                    self._proc = None
                self._md_status.content = (
                    f"**Status:** Isaac Sim worker launch error -- {str(exc)[:200]}"
                )
                print(
                    "[isaacsim launcher] Worker launch error:\n"
                    f"{traceback.format_exc()}"
                )
                return
            finally:
                try:
                    listener.close()
                except Exception:
                    pass

            print(
                f"[isaacsim launcher] Spawned pid={self._proc.pid} problem={problem_name} "
                f"goal_mode={goal_mode} rgf={rgf} policy={policy_name}"
            )

        def run(self):
            print()
            print(f"  Peg-in-Hole Dynamic Isaac Sim Eval   http://localhost:{self.port}")
            print()
            try:
                while True:
                    self._poll()
                    time.sleep(1.0 / 120.0)
            except KeyboardInterrupt:
                print("\n[isaacsim launcher] Shutting down...")
                self._kill_subprocess()

    if args.dry_run:
        print(f"[eval_isaacsim] worker python: {args.python or _default_isaacsim_python()}")
        print(f"[eval_isaacsim] selected policy: {selected_policy.name}")
        print(f"[eval_isaacsim] task: {selected_policy.task}")
        print(f"[eval_isaacsim] checkpoint: {selected_policy.checkpoint}")
        print(f"[eval_isaacsim] initial problem: {initial_problem}")
        print(f"[eval_isaacsim] port: {args.port}")
        return 0

    IsaacSimPegDynamicDemo(
        policies=viewer_policies,
        port=args.port,
        headless=not args.no_headless,
        goal_mode=args.goal_mode,
        random_goal_fraction=args.random_goal_fraction,
        initial_policy=selected_policy.name,
        extra_overrides=extra_overrides,
        initial_problem=initial_problem,
        policy_tasks={name: spec.task for name, spec in policies.items()},
        policy_problems={name: spec.problem or spec.dir_name for name, spec in policies.items()},
    ).run()
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.worker:
        return _worker_main(args)
    return _run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
