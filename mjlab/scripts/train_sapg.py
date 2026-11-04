"""Fine-tune the original six-block SAPG learner in the MJLab assembly scene."""

import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

from warp_source import activate_source_overlay, training_contract

# The source override must precede imports of MJLab and mujoco_warp. The normal
# parser below still validates every argument and documents the opt-in flag.
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--warp-overlay", type=Path)
_overlay_path = _bootstrap.parse_known_args()[0].warp_overlay
_overlay_proof = activate_source_overlay(_overlay_path) if _overlay_path else None

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.config.kernel_cache_dir = os.environ.get(
    "P2P_WARP_CACHE", str(Path(".uv-cache/warp").resolve())
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.nan_guard import NanGuardCfg  # noqa: E402
from source_identity import source_identity  # noqa: E402

from play2perfect_mjlab.assembly_env_cfg import (  # noqa: E402
    TRAINING_PRESET_TASKS,
    assembly_cfg,
    task_contract,
    training_bundle,
)
from play2perfect_mjlab.assets import sha256  # noqa: E402
from play2perfect_mjlab.collision_capacity import configure_epa_horizon  # noqa: E402
from play2perfect_mjlab.runtime_info import (  # noqa: E402
    runtime_info,
    simulation_options,
)
from play2perfect_mjlab.source_sapg import (  # noqa: E402
    TRAINER_COMMIT,
    TRAINER_TREE,
    AssemblyVecEnv,
    TrainingObserver,
    create_agent,
    make_config,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--warp-overlay", type=Path, help="Verified experimental stabilization checkout"
    )
    p.add_argument("--bundle", type=Path)
    p.add_argument("--training-preset", choices=tuple(TRAINING_PRESET_TASKS))
    p.add_argument(
        "--kinematics-only-rewards",
        action="store_true",
        help="Refresh reward kinematics without a redundant dynamics solve",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--task",
        choices=("tight_insertion", "screwing", "fabrica"),
        default="tight_insertion",
    )
    p.add_argument("--num-envs", type=int, default=1536)
    p.add_argument("--sdf-initpoints", type=int)
    p.add_argument("--nconmax", type=int, default=512)
    p.add_argument("--njmax", type=int, default=2048)
    p.add_argument("--epa-horizon", type=int, default=24)
    p.add_argument("--solver-iterations", type=int, default=100)
    p.add_argument("--ls-iterations", type=int, default=50)
    p.add_argument("--cuda-memory-fraction", type=float)
    p.add_argument("--duty-cycle", type=float, default=1.0)
    p.add_argument("--min-free-gpu-gb", type=float, default=0.0)
    p.add_argument("--required-free-gpu-gb", type=float, default=0.0)
    p.add_argument("--yield-to-file", type=Path)
    p.add_argument("--physics-dt", type=float)
    p.add_argument("--decimation", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--initial-std-max", type=float)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--minibatch-size", type=int, default=6144)
    p.add_argument("--max-epochs", type=int, default=1000000)
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--checkpoint-seconds", type=float, default=300)
    p.add_argument("--keep-checkpoints", type=int, default=12)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()
    if (args.run_dir / "config.json").exists():
        p.error("Use a new run directory, including when resuming a saved learner")
    if args.bundle is None and args.training_preset:
        args.bundle = training_bundle(args.training_preset)
    entered_main = time.monotonic()
    if args.keep_checkpoints < 0 or args.checkpoint_seconds <= 0:
        p.error("Checkpoint interval must be positive and retention nonnegative")
    if (
        not 0 < args.duty_cycle <= 1
        or args.min_free_gpu_gb < 0
        or args.required_free_gpu_gb < 0
    ):
        p.error("Duty cycle must be in(0,1] and GPU headroom nonnegative")
    if args.required_free_gpu_gb:
        if not args.device.startswith("cuda"):
            p.error("The startup GPU memory requirement needs a CUDA device")
        free, _ = torch.cuda.mem_get_info(args.device)
        if free < args.required_free_gpu_gb * 2**30:
            p.error("Insufficient free GPU memory for this run's startup budget")
    if any(
        value is not None and value <= 0
        for value in (args.learning_rate, args.initial_std_max)
    ):
        p.error("Learning rate and noise cap must be positive when specified")
    torch.set_num_threads(args.torch_threads)
    if args.cuda_memory_fraction is not None:
        if not args.device.startswith("cuda") or not 0 < args.cuda_memory_fraction <= 1:
            p.error("CUDA memory fraction needs a CUDA device and a value in(0,1]")
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, args.device
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    source = source_identity()
    if source.get("dirty") and not args.allow_dirty:
        p.error(
            "Commit source changes, or record a local smoke check with --allow-dirty"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    collision_capacity = configure_epa_horizon(args.epa_horizon)
    cfg = make_config(
        Path("configs/source_sapg.yaml"),
        num_envs=args.num_envs,
        device=args.device,
        run_dir=args.run_dir,
        seed=args.seed,
        max_epochs=args.max_epochs,
        horizon=args.horizon,
        minibatch_size=args.minibatch_size,
    )
    if args.learning_rate is not None:
        cfg["params"]["config"]["learning_rate"] = args.learning_rate
    env_cfg = assembly_cfg(
        args.bundle or Path("data/browser") / args.task,
        args.num_envs,
        auto_reset=True,
        sdf_initpoints=args.sdf_initpoints,
        training_preset=args.training_preset,
        kinematics_only_rewards=args.kinematics_only_rewards,
        nconmax=args.nconmax,
        njmax=args.njmax,
        solver_iterations=args.solver_iterations,
        ls_iterations=args.ls_iterations,
        physics_dt=args.physics_dt,
        decimation=args.decimation,
    )
    env_cfg.seed = args.seed
    env_cfg.sim.nan_guard = NanGuardCfg(
        enabled=True,
        buffer_size=2,
        output_dir=str(args.run_dir / "physics_nan"),
        max_envs_to_dump=3,
    )
    contract = {
        "trainer_commit": TRAINER_COMMIT,
        "trainer_tree": TRAINER_TREE,
        "task": task_contract(env_cfg),
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "minibatch_size": args.minibatch_size,
        "seed": args.seed,
        "base_checkpoint_sha256": sha256(args.checkpoint),
        "source_config_sha256": sha256(Path("configs/source_sapg.yaml")),
        "manifest_sha256": sha256(
            (args.bundle or Path("data/browser") / args.task) / "manifest.json"
        ),
        "mixed_precision": args.device.startswith("cuda"),
        "continuous_stage_memory_version": 1,
        "initial_std_max": args.initial_std_max,
        "initial_actor_learning_rate": cfg["params"]["config"]["learning_rate"],
    }
    if _overlay_proof is not None:
        contract["effective_warp_overlay"] = training_contract(_overlay_proof)
    metadata = {
        "run_directory": str(args.run_dir.resolve()),
        "numerical_recovery": {
            "invalid_transitions": "capture, reset affected worlds, "
            "zero reward, terminate",
            "physics_history_substeps": 2,
            "policy_mean": "capture and reject before sampling",
        },
        "scope": "source six-block SAPG; "
        + (args.training_preset or "fixed scene, no randomization"),
        "source": source,
        "resume_contract": contract,
        "runtime": runtime_info(),
        "collision_capacity": collision_capacity,
        "resource_controls": {
            "cuda_torch_memory_fraction": args.cuda_memory_fraction,
            "target_duty_cycle": args.duty_cycle,
            "minimum_free_gpu_gib": args.min_free_gpu_gb,
            "required_startup_free_gpu_gib": args.required_free_gpu_gb,
            "yield_to_file": str(args.yield_to_file) if args.yield_to_file else None,
        },
        "configuration": cfg,
    }
    if args.resume:
        metadata["resume"] = {
            "checkpoint": str(args.resume),
            "sha256": sha256(args.resume),
            "episode_state": "fresh physical episode and zero recurrent memory",
        }
    (args.run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Avoid a recursive dict: Runner deep-copies config and inserts runtime objects.
    cfg["params"]["config"]["mjlab_metadata"] = {
        key: value for key, value in metadata.items() if key != "configuration"
    }
    environment_start = time.monotonic()
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    metadata["actual_physics"] = simulation_options(env)
    cfg["params"]["config"]["mjlab_metadata"]["actual_physics"] = metadata[
        "actual_physics"
    ]
    (args.run_dir / "actual_options.json").write_text(
        json.dumps(metadata["actual_physics"], indent=2) + "\n"
    )
    metadata["setup_timing"] = {
        "environment_setup_seconds": time.monotonic() - environment_start,
        "main_to_environment_ready_seconds": time.monotonic() - entered_main,
        "training_wall_seconds": "observer timer starts after environment setup",
    }
    if args.device.startswith("cuda"):
        metadata["hardware"] = {"gpu": torch.cuda.get_device_name(args.device)}
    observer = TrainingObserver(
        args.run_dir,
        args.checkpoint_seconds,
        args.max_seconds,
        args.keep_checkpoints,
        duty_cycle=args.duty_cycle,
        min_free_gpu_gb=args.min_free_gpu_gb,
        yield_to_file=args.yield_to_file,
    )
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(
            sig, lambda signum, frame: setattr(observer, "stop_requested", True)
        )
    try:
        agent = create_agent(
            cfg,
            AssemblyVecEnv(env, args.run_dir / "numerical_diagnostics"),
            observer,
            args.resume or args.checkpoint,
            resume=bool(args.resume),
            initial_std_max=args.initial_std_max,
        )
        metadata.update(agent.config["mjlab_metadata"])
        (args.run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        observer.save()
        agent.train()
    except FloatingPointError as error:
        if "agent" in locals() and all(
            torch.isfinite(v).all()
            for model in (agent.model, agent.central_value_net)
            for v in model.state_dict().values()
        ):
            agent.config["mjlab_metadata"]["aborted_rollout"] = str(error)
            observer.save()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
