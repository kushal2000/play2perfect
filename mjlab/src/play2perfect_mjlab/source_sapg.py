"""Small MJLab bridge to Play2Perfect's pinned SAPG implementation.

The optimizer, losses, experience sharing, and recurrent replay stay upstream.
Our additions supply the environment, metrics, and complete learner checkpoints.
Resume starts a fresh physical episode; it does not replay stale observations.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np
import torch
import warp as wp
import yaml
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from play2perfect_mjlab.numerical_guard import (
    TransitionGuard,
    restore_task_progress,
    task_progress,
)

TRAINER_COMMIT = "ff2fc62873f6c6cd93164123c8358e9ce2d65c9b"
TRAINER_TREE = "dc653791ca2b606205c5314c4b9cebd6e5b8ce37"
CHECKPOINT_VERSION = 1


class AssemblyVecEnv:
    """Expose raw 140/162 observations; SAPG appends its own per-world tag."""

    def __init__(self, env, diagnostics_dir=None):
        self.unwrapped = env
        self.env = SimpleNamespace(device=env.device)
        self.num_envs = env.num_envs
        self.numerics = TransitionGuard(env, diagnostics_dir)

    def get_env_info(self):
        return {
            "observation_space": gym.spaces.Box(-np.inf, np.inf, (140,)),
            "state_space": gym.spaces.Box(-np.inf, np.inf, (162,)),
            "action_space": gym.spaces.Box(-1.0, 1.0, (29,)),
            "agents": 1,
            "value_size": 1,
        }

    @staticmethod
    def observations(obs):
        return {"obs": obs["actor"][..., :140], "states": obs["critic"]}

    def reset(self):
        obs, _ = self.unwrapped.reset()
        return self.observations(obs)

    def step(self, actions):
        self.numerics.before_step()
        obs, reward, terminated, truncated, _ = self.unwrapped.step(actions)
        state = self.unwrapped._p2p
        infos = {
            "time_outs": truncated,
            "completed": state["episode_completed"].clone(),
            "goal_count": state["episode_goal_count"].clone(),
            "recurrent_reset": state["recurrent_reset"].clone(),
            "reward": reward,
            "max_goals": sum(
                len(stage["goals"]) for stage in state["metadata"]["stages"]
            )
            if state["metadata"].get("stages")
            else state["max_goals"].clone(),
            "insertion_world": ~state["random_goal_env"],
            "fall": state["episode_fell"].clone(),
        }
        obs, reward, terminated, truncated, infos = self.numerics.recover(
            obs, reward, terminated, truncated, infos
        )
        return self.observations(obs), reward, terminated | truncated, infos

    def get_number_of_agents(self):
        return 1

    def set_train_info(self, *args, **kwargs):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        if state is not None:
            raise ValueError("This bridge resumes with fresh physical episodes")


def make_config(
    path,
    *,
    num_envs,
    device,
    run_dir,
    seed=0,
    max_epochs=1000000,
    horizon=16,
    minibatch_size=6144,
):
    if num_envs % 6 or num_envs < 6:
        raise ValueError("Six pretrained SAPG blocks require num_envs divisible by6")
    batch = num_envs * horizon
    if minibatch_size > batch or batch % minibatch_size or minibatch_size % horizon:
        raise ValueError("Minibatch must divide the rollout and contain full sequences")
    params = yaml.safe_load(Path(path).read_text())["params"]
    params["seed"] = seed
    cfg = params["config"]
    cfg.update(
        name="0_mjlab_sapg",
        full_experiment_name="0_mjlab_sapg",
        env_name="mjlab",
        device=device,
        device_name=device,
        num_actors=num_envs,
        train_dir=str(Path(run_dir).resolve() / "rl_games"),
        max_epochs=max_epochs,
        horizon_length=horizon,
        seq_length=horizon,
        minibatch_size=minibatch_size,
        expl_coef_block_size=num_envs // 6,
        mixed_precision=device.startswith("cuda"),
        save_frequency=0,
    )
    cfg["central_value_config"]["minibatch_size"] = minibatch_size
    return {"params": params}


def unwrap_checkpoint(path):
    weights = torch.load(path, map_location="cpu", weights_only=False)
    return weights[0] if 0 in weights else weights


class RecurrentBoundaryBuffer:
    """Store memory boundaries for replay, while keeping episode masks for GAE."""

    def __init__(self, buffer, agent):
        self.buffer, self.agent = buffer, agent

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def update_data(self, name, index, value):
        if name == "dones":
            self.agent.episode_dones[index].copy_(value)
            value = value.bool() | self.agent.recurrent_boundary
        self.buffer.update_data(name, index, value)


class PhysicsRolloutMonitor:
    """Sample every physics substep, before auto-reset can clear overflow flags."""

    def __init__(self, sim):
        from mujoco_warp import OverflowType

        self.sim = sim
        self.overflow_types = list(OverflowType)
        data = sim.wp_data
        device = wp.to_torch(data.nefc).device
        self.peaks = torch.zeros(3, dtype=torch.int32, device=device)
        self.flags = torch.zeros(
            len(self.overflow_types), dtype=torch.bool, device=device
        )
        self.flag_bits = torch.tensor(
            [int(flag) for flag in self.overflow_types], device=device
        )
        self.substeps = 0
        self.original_step = sim.step
        sim.step = self.step

    def step(self):
        self.original_step()
        data = self.sim.wp_data
        sample = torch.stack(
            (
                wp.to_torch(data.nacon)[0],
                wp.to_torch(data.nefc).max(),
                wp.to_torch(data.solver_niter).max(),
            )
        )
        torch.maximum(self.peaks, sample, out=self.peaks)
        overflow = wp.to_torch(data.overflow)
        self.flags |= ((overflow[:, None] & self.flag_bits) != 0).any(0)
        self.substeps += 1

    def snapshot(self):
        contacts, constraints, iterations = self.peaks.tolist()
        return {
            "contacts_total": contacts,
            "max_world_constraints": constraints,
            "max_solver_iterations": iterations,
            "overflow_flags": [
                flag.name
                for flag, seen in zip(self.overflow_types, self.flags.tolist())
                if seen
            ],
            "physics_substeps": self.substeps,
            "contact_capacity_total": self.sim.wp_data.naconmax,
            "constraint_capacity_per_world": self.sim.wp_data.njmax,
            "scope": "peak over every physics substep in this rollout",
        }

    def validate(self):
        report = self.snapshot()
        # Solver convergence limits are logged, but do not discard constraints.
        capacity_flags = set(report["overflow_flags"]) - {"ITERATIONS", "LS_ITERATIONS"}
        if (
            capacity_flags
            or report["contacts_total"] > report["contact_capacity_total"]
            or report["max_world_constraints"] > report["constraint_capacity_per_world"]
        ):
            raise RuntimeError(
                "Physics capacity overflow; reject rollout before optimizer update: "
                + json.dumps(report)
            )

    def reset(self):
        self.peaks.zero_()
        self.flags.zero_()
        # Warp flags persist until data reset. Clear diagnostics at rollout start.
        self.sim.wp_data.overflow.zero_()
        self.substeps = 0


class AssemblySapgAgent(A2CAgent):
    """Keep source optimization; extend resume and continuous-stage memory masks."""

    def init_tensors(self):
        super().init_tensors()
        self.episode_dones = torch.empty(
            self.horizon_length, self.num_actors, device=self.ppo_device
        )
        self.recurrent_boundary = torch.zeros(
            self.num_actors, dtype=torch.bool, device=self.ppo_device
        )
        self.experience_buffer = RecurrentBoundaryBuffer(self.experience_buffer, self)
        sim = self.vec_env.unwrapped.sim
        if not hasattr(sim, "_p2p_rollout_monitor"):
            sim._p2p_rollout_monitor = PhysicsRolloutMonitor(sim)
        self.physics_monitor = sim._p2p_rollout_monitor
        if not hasattr(self, "_mu_guard_handle"):
            self._mu_guard_handle = self.model.a2c_network.mu.register_forward_hook(
                self._check_policy_output
            )
        self._inference_inputs = None

    def _check_policy_output(self, module, inputs, output):
        if self._inference_inputs is None or torch.isfinite(output).all():
            return
        ids = (~torch.isfinite(output).all(-1)).nonzero().flatten()
        directory = (
            Path(self.config["mjlab_metadata"].get("run_directory", self.nn_dir))
            / "numerical_diagnostics"
        )
        directory.mkdir(parents=True, exist_ok=True)
        obs, recurrent = self._inference_inputs
        torch.save(
            {
                "scope": "Non-finite policy mean before Gaussian sampling",
                "worlds": ids.cpu(),
                "observations": {k: v[ids].detach().cpu() for k, v in obs.items()},
                "recurrent": [v[:, ids].detach().cpu() for v in recurrent]
                if recurrent is not None
                else None,
                "mean": output[ids].detach().cpu(),
                "epoch": self.epoch_num,
            },
            directory / "policy_output_failure.pth",
        )
        raise FloatingPointError(
            f"Non-finite policy mean in {len(ids)} worlds; captured before sampling"
        )

    def get_action_values(self, obs, rnn_states=None):
        if not all(torch.isfinite(v).all() for v in obs.values()):
            raise FloatingPointError(
                "Non-finite policy input escaped transition recovery"
            )
        self._inference_inputs = (obs, rnn_states)
        try:
            return super().get_action_values(obs, rnn_states)
        finally:
            self._inference_inputs = None

    def env_step(self, actions):
        result = super().env_step(actions)
        self.recurrent_boundary = result[-1]["recurrent_reset"]
        # Source rollout updates rnn_states before env_step. Clear at handoff
        # now, so both the next action and saved sequence states start at zero.
        for recurrent in self.rnn_states:
            recurrent[:, self.recurrent_boundary] = 0
        return result

    def discount_values(self, fdones, last_values, mb_fdones, mb_values, rewards):
        return super().discount_values(
            fdones, last_values, self.episode_dones, mb_values, rewards
        )

    def play_steps(self):
        self.physics_monitor.reset()
        batch, extras = super().play_steps()
        self.physics_monitor.validate()
        # The relabeled off-policy returns also use physical episode masks.
        # batch['dones'] retains the extra boundaries for recurrent replay.
        extras["dones"] = self.episode_dones.clone()
        return batch, extras

    def train_epoch(self):
        result = super().train_epoch()
        self.latest_metrics = {
            "play_seconds": result[1],
            "update_seconds": result[2],
            "policy_loss": float(torch.stack(result[4]).mean().detach()),
            "actor_value_loss": float(torch.stack(result[5]).mean().detach()),
            "entropy": float(torch.stack(result[7]).mean().detach()),
            "kl": float(torch.stack(result[8]).mean().detach()),
        }
        if not all(np.isfinite(v) for v in self.latest_metrics.values()):
            raise FloatingPointError("Non-finite SAPG training metrics")
        return result

    def get_full_state_weights(self):
        weights = super().get_full_state_weights()
        cv = self.central_value_net
        weights["mjlab"] = {
            "version": CHECKPOINT_VERSION,
            "metadata": self.config["mjlab_metadata"],
            "central_value_optimizer": cv.optimizer.state_dict(),
            "central_value_epoch": cv.epoch_num,
            "central_value_frame": cv.frame,
            "central_value_lr": cv.lr,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if self.ppo_device.startswith("cuda")
            else None,
            "episode_state": "fresh reset on resume",
            "task_progress": task_progress(self.vec_env.unwrapped),
        }
        return weights

    def set_full_state_weights(self, weights, set_epoch=True):
        extension = weights.get("mjlab", {})
        if extension.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Resume requires a complete MJLab SAPG checkpoint")
        expected = self.config["mjlab_metadata"]["resume_contract"]
        if extension["metadata"]["resume_contract"] != expected:
            raise ValueError("Resume contract differs from the saved learner")
        # Upstream can restore cached observations without restoring physics.
        # Remove these fields together, then explicitly initialize a new episode.
        weights = {
            k: v
            for k, v in weights.items()
            if k
            not in {
                "rnn_states",
                "dones",
                "obs",
                "current_rewards",
                "current_shaped_rewards",
                "current_lengths",
            }
        }
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        cv = self.central_value_net
        cv.optimizer.load_state_dict(extension["central_value_optimizer"])
        cv.epoch_num = extension["central_value_epoch"]
        cv.frame = extension["central_value_frame"]
        cv.lr = extension["central_value_lr"]
        for name in (
            "obs",
            "rnn_states",
            "dones",
            "current_rewards",
            "current_shaped_rewards",
            "current_lengths",
        ):
            setattr(self, name, None)
        torch.set_rng_state(extension["torch_rng"].cpu())
        np.random.set_state(extension["numpy_rng"])
        random.setstate(extension["python_rng"])
        if extension["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(extension["cuda_rng"])
        if "task_progress" in extension:
            restore_task_progress(self.vec_env.unwrapped, extension["task_progress"])


def prepare_best_checkpoint_link(agent):
    """Seed the source pointer so its first safe_symlink avoids31s of retries.

    Upstream first tries to remove the pointer with exponential retries, even
    when it does not exist. A dangling symlink is removable before the first
    best checkpoint is saved. Preserve any pointer or file already present.
    """
    directory = Path(agent.experiment_dir) / "best"
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / "model.pth"
    if not os.path.lexists(link):
        target = Path(agent.nn_dir) / (agent.config["name"] + ".pth")
        link.symlink_to(os.path.relpath(target, start=directory))


def create_agent(
    config, vec_env, observer, checkpoint, *, resume=False, initial_std_max=None
):
    runner = Runner(algo_observer=observer)
    runner.load(config)
    runner.set_vec_env(vec_env)
    agent = AssemblySapgAgent(base_name="run", params=runner.params)
    weights = unwrap_checkpoint(checkpoint)
    if resume:
        agent.set_full_state_weights(weights)
        if "exploration_initialization" in weights["mjlab"]["metadata"]:
            agent.config["mjlab_metadata"]["exploration_initialization"] = weights[
                "mjlab"
            ]["metadata"]["exploration_initialization"]
    else:
        agent.set_weights(weights)
        agent.central_value_net.load_state_dict(weights["assymetric_vf_nets"])
        sigma = agent.model.a2c_network.sigma
        inherited = sigma.detach().exp().cpu().tolist()
        if initial_std_max is not None:
            with torch.no_grad():
                sigma.clamp_(max=float(np.log(initial_std_max)))
        if "mjlab_metadata" in agent.config:
            agent.config["mjlab_metadata"]["exploration_initialization"] = {
                "initial_std_max": initial_std_max,
                "inherited_std_by_block": inherited,
                "initial_std_by_block": sigma.detach().exp().cpu().tolist(),
                "optimizer_initialization": "fresh actor and critic Adam",
            }
    agent.init_tensors()
    prepare_best_checkpoint_link(agent)
    return agent


class TrainingObserver(AlgoObserver):
    def __init__(
        self,
        run_dir,
        checkpoint_seconds=300,
        max_seconds=None,
        keep=12,
        *,
        duty_cycle=1.0,
        min_free_gpu_gb=0.0,
        yield_to_file=None,
    ):
        self.run_dir = Path(run_dir)
        self.checkpoint_seconds = checkpoint_seconds
        self.max_seconds = max_seconds
        self.keep = keep
        self.started = self.last_save = time.monotonic()
        self.stop_requested = False
        self.episodes = self.completed = 0
        self.block_episodes = [0] * 6
        self.block_completed = [0] * 6
        self.goal_counts = [0] * 6
        self.duty_cycle = duty_cycle
        self.min_free_gpu_bytes = min_free_gpu_gb * 2**30
        self.yield_to_file = yield_to_file
        self.last_epoch_end = self.started
        self.last_resource_poll = self.started - 60
        self.process_gpu_mib = None
        self.stop_reason = None

    def resource_status(self, now):
        status = {}
        if self.algo.ppo_device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(self.algo.ppo_device)
            status.update(gpu_free_bytes=free, gpu_total_bytes=total)
            if free < self.min_free_gpu_bytes:
                self.stop_requested = True
                self.stop_reason = "GPU memory headroom requested by other work"
        if now - self.last_resource_poll >= 60:
            self.last_resource_poll = now
            if self.algo.ppo_device.startswith("cuda"):
                try:
                    output = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=True,
                    ).stdout
                    self.process_gpu_mib = next(
                        int(memory.strip())
                        for line in output.splitlines()
                        for pid, memory in [line.split(",", 1)]
                        if int(pid.strip()) == os.getpid()
                    )
                except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
                    self.process_gpu_mib = None
            if self.yield_to_file is not None:
                # One exact known mount file; no directory scan or SSH polling.
                result = subprocess.run(
                    ["timeout", "10", "test", "-f", str(self.yield_to_file)],
                    check=False,
                )
                if result.returncode == 0:
                    self.stop_requested = True
                    self.stop_reason = "Cluster allocation started"
        status.update(
            process_gpu_mib=self.process_gpu_mib, stop_reason=self.stop_reason
        )
        return status

    def after_init(self, algo):
        self.algo = algo
        self.reward_totals = torch.zeros(6, device=algo.ppo_device)
        self.reward_steps = 0
        self.previous_goal_fraction = torch.zeros(
            algo.num_actors, device=algo.ppo_device
        )
        self.previous_all_goals = torch.zeros_like(self.previous_goal_fraction)
        self.previous_fall = torch.zeros_like(self.previous_goal_fraction)
        self.insertion_world = torch.ones_like(
            self.previous_goal_fraction, dtype=torch.bool
        )

    def process_infos(self, infos, done_indices, **kwargs):
        if "insertion_world" in infos:
            self.insertion_world = infos["insertion_world"]
        if "reward" in infos:
            self.reward_totals += infos["reward"].reshape(6, -1).mean(dim=1)
            self.reward_steps += 1
        ids = done_indices.flatten()
        if not len(ids):
            return
        if "max_goals" in infos:
            maximum = infos["max_goals"]
            maximum = maximum[ids] if isinstance(maximum, torch.Tensor) else maximum
            self.previous_goal_fraction[ids] = (
                infos["goal_count"][ids].float() / maximum
            )
            self.previous_all_goals[ids] = (infos["goal_count"][ids] >= maximum).float()
            self.previous_fall[ids] = infos["fall"][ids].float()
        blocks = ids // self.algo.intr_coef_block_size
        for block in range(6):
            selected = ids[blocks == block]
            self.block_episodes[block] += len(selected)
            self.block_completed[block] += int(infos["completed"][selected].sum())
            self.goal_counts[block] += int(infos["goal_count"][selected].sum())

    def save(self):
        epoch = self.algo.epoch_num
        target = self.run_dir / f"checkpoint_{epoch:08d}.pth"
        temporary = target.with_suffix(".pth.tmp")
        torch.save({0: self.algo.get_full_state_weights()}, temporary)
        temporary.replace(target)
        latest = self.run_dir / "latest.json"
        temp_latest = latest.with_suffix(".tmp")
        temp_latest.write_text(
            json.dumps(
                {
                    "checkpoint": str(target.resolve()),
                    "epoch": epoch,
                    "transitions": self.algo.frame,
                }
            )
            + "\n"
        )
        temp_latest.replace(latest)
        if self.keep:
            recent = sorted(self.run_dir.glob("checkpoint_*.pth"))
            for old in recent[: -self.keep]:
                if old.name != "checkpoint_00000000.pth":
                    old.unlink()
        self.last_save = time.monotonic()
        print(f"MJLab checkpoint: {target}", flush=True)

    def after_print_stats(self, frame, epoch_num, total_time):
        algo = self.algo
        now = time.monotonic()
        # Finite parameters and losses are checked at the first update; later
        # checkpoints and held-out rollouts provide the useful ongoing evidence.
        if epoch_num == 1:
            for model in (algo.model, algo.central_value_net):
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite SAPG parameters")
        record = {
            "epoch": epoch_num,
            "transitions": algo.frame,
            "wall_seconds": now - self.started,
            "learning_rate": algo.last_lr,
            "block_episodes": self.block_episodes,
            "block_completed": self.block_completed,
            "block_goal_counts": self.goal_counts,
            # Match source previous-episode metrics; withdrawal stays separate.
            "block_previous_goal_fraction": self.previous_goal_fraction.reshape(6, -1)
            .mean(1)
            .tolist(),
            "block_previous_all_goals_hit": self.previous_all_goals.reshape(6, -1)
            .mean(1)
            .tolist(),
            "block_previous_fall_ratio": self.previous_fall.reshape(6, -1)
            .mean(1)
            .tolist(),
            "actor_aux_value_loss": algo.has_value_loss,
            "block_mean_reward": (
                self.reward_totals / max(1, self.reward_steps)
            ).tolist(),
            "central_value_learning_rate": (
                algo.central_value_net.optimizer.param_groups[0]["lr"]
            ),
        }
        insertion = self.insertion_world.reshape(6, -1)
        record["block_insertion_world_counts"] = insertion.sum(1).tolist()
        denominator = insertion.sum(1).clamp(min=1)
        for name, values in (
            ("goal_fraction", self.previous_goal_fraction),
            ("all_goals_hit", self.previous_all_goals),
            ("fall_ratio", self.previous_fall),
        ):
            record[f"block_previous_insertion_{name}"] = (
                (values.reshape(6, -1) * insertion).sum(1) / denominator
            ).tolist()
            selected = values[self.insertion_world]
            record[f"previous_insertion_{name}"] = (
                float(selected.mean()) if len(selected) else None
            )
        if algo.ppo_device.startswith("cuda"):
            record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                algo.ppo_device
            )
            record["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                algo.ppo_device
            )
        record.update(getattr(algo, "latest_metrics", {}))
        record["invalid_world_resets"] = algo.vec_env.numerics.invalid_worlds
        progress = task_progress(algo.vec_env.unwrapped)
        record["task_progress"] = {
            k: v for k, v in progress.items() if k != "random_goal_env"
        }
        record["resource_status"] = self.resource_status(now)
        idle_seconds = max(0, now - self.last_epoch_end) * (1 / self.duty_cycle - 1)
        record["target_duty_cycle"] = self.duty_cycle
        record["planned_idle_seconds"] = idle_seconds
        env = algo.vec_env.unwrapped
        iterations = env.sim.wp_data.solver_niter.numpy()
        record["physics_sample"] = {
            "contacts_total": int(env.sim.wp_data.nacon.numpy()[0]),
            "max_world_constraints": int(env.sim.wp_data.nefc.numpy().max()),
            "max_solver_iterations": int(iterations.max()),
            "solver_limit_world_fraction": float(
                (iterations >= env.cfg.sim.mujoco.iterations).mean()
            ),
            "scope": "last control of this rollout; not a substep peak",
        }
        record["physics_peak"] = algo.physics_monitor.snapshot()
        with (self.run_dir / "metrics.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        self.block_episodes = [0] * 6
        self.block_completed = [0] * 6
        self.goal_counts = [0] * 6
        self.reward_totals.zero_()
        self.reward_steps = 0
        stopped = self.stop_requested or (
            self.max_seconds is not None and now - self.started >= self.max_seconds
        )
        final = algo.max_epochs != -1 and epoch_num >= algo.max_epochs
        if stopped or final or now - self.last_save >= self.checkpoint_seconds:
            self.save()
        if stopped:
            raise SystemExit(0)
        if final:
            return
        idle_deadline = time.monotonic() + idle_seconds
        while time.monotonic() < idle_deadline:
            if self.stop_requested:
                self.save()
                raise SystemExit(0)
            time.sleep(max(0, min(1, idle_deadline - time.monotonic())))
        self.last_epoch_end = time.monotonic()
