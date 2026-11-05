"""Pose/state recording through MJLab's lifecycle hooks, before auto-reset."""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.managers.recorder_manager import RecorderTerm

from play2perfect_mjlab.assembly_mdp import geometry
from play2perfect_mjlab.assets import sha256


class RolloutRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.output = Path(cfg.params["output"])
        self.every = cfg.params.get("every", 20)
        self.alive = np.ones(env.num_envs, dtype=bool)
        self.rows = []

    def capture(self, terminal=None):
        env = self._env
        fields = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "eq_active")
        _, obj, goal = geometry(env, env._p2p["reward_offsets"])
        distance = torch.linalg.vector_norm(obj - goal, dim=-1).amax(-1)
        wd, model = env.sim.wp_data, env.sim.mj_model
        count = min(int(wd.nacon.numpy()[0]), wd.naconmax)
        pairs = wd.contact.geom.numpy()[:count]
        worlds = wd.contact.worldid.numpy()[:count]
        distances = wd.contact.dist.numpy()[:count]
        robot = np.array(
            [
                (model.geom(i).name or "").startswith("robot/")
                for i in range(model.ngeom)
            ]
        )
        objects = np.array(
            [
                (model.geom(i).name or "").startswith(("object/", "second/"))
                for i in range(model.ngeom)
            ]
        )
        selected = (
            (
                (robot[pairs[:, 0]] & objects[pairs[:, 1]])
                | (objects[pairs[:, 0]] & robot[pairs[:, 1]])
            )
            & (distances <= 0)
            & (worlds >= 0)
            & (worlds < env.num_envs)
        )
        hand_contacts = np.bincount(worlds[selected], minlength=env.num_envs)
        penetration = np.zeros(env.num_envs)
        np.maximum.at(penetration, worlds[selected], -distances[selected])
        self.rows.append(
            {
                "control": np.array(env.common_step_counter),
                "active_worlds": self.alive.copy(),
                "terminal_worlds": np.zeros_like(self.alive)
                if terminal is None
                else terminal,
                "goal_index": env._p2p["goal_index"].cpu().numpy().copy(),
                **{
                    name: env._p2p[name].cpu().numpy().copy()
                    for name in ("stage_index", "handoff_step", "release_remaining")
                    if name in env._p2p
                },
                "distance_m": distance.cpu().numpy().copy(),
                "reward_target_distance_m": env._p2p["distance"].cpu().numpy().copy(),
                "robot_object_penetrating_contacts": hand_contacts,
                "robot_object_max_penetration_m": penetration,
                **{
                    field: getattr(env.sim.data, field)[:].cpu().numpy().copy()
                    for field in fields
                },
            }
        )

    def record_post_reset(self, env_ids):
        if self._env.common_step_counter == 0 and not self.rows:
            self.capture()

    def record_pre_reset(self, env_ids):
        terminal = np.zeros_like(self.alive)
        terminal[env_ids.cpu().numpy()] = True
        terminal &= self.alive
        if terminal.any():
            self.capture(terminal)
        self.alive &= ~terminal

    def record_post_step(self):
        if self._env.common_step_counter % self.every == 0 and self.alive.any():
            self.capture()

    def finish(self, terminal=None):
        """Freeze the last rollout state before a caller performs a manual reset."""
        if self.alive.any():
            self.capture(terminal)
        self.alive[:] = False

    def close(self):
        if not self.rows:
            return
        if (
            self.alive.any()
            and self.rows[-1]["control"] != self._env.common_step_counter
        ):
            self.capture()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        model_path = self.output.with_suffix(".mjb")
        mujoco.mj_saveModel(self._env.sim.mj_model, str(model_path))
        np.savez_compressed(
            self.output,
            **{key: np.stack([r[key] for r in self.rows]) for key in self.rows[0]},
        )
        self.output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "scope": self.cfg.params.get(
                        "scope",
                        "Warp first-episode poses; terminal states before reset",
                    ),
                    "model_layout": "mjlab_compiled",
                    "frames": len(self.rows),
                    "num_envs": self._env.num_envs,
                    "bundle": self.cfg.params["bundle"],
                    "sdf_initpoints": self._env.cfg.sdf_initpoints,
                    "record_every_controls": self.every,
                    "compiled_model": model_path.name,
                    "compiled_model_sha256": sha256(model_path),
                    "mujoco_version": mujoco.__version__,
                },
                indent=2,
            )
            + "\n"
        )
