"""Native MuJoCo reference for the published browser assembly controllers.

Coordinates and metadata quaternions use xyzw. MuJoCo stores wxyz.
Success means goal completion AND withdrawal, including release follow-through.
This module is a parity reference, not the randomized training MDP.
"""

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from play2perfect_mjlab.policy import action_targets


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, q[[3, 0, 1, 2]])
    return matrix.reshape(3, 3)


def multiply(a, b):
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, np.asarray(a)[[3, 0, 1, 2]], np.asarray(b)[[3, 0, 1, 2]])
    return result[[1, 2, 3, 0]]


def load_model(bundle: Path, metadata):
    root = ET.fromstring((bundle / "scene.xml").read_text())
    for name, pose in metadata.get("initial_root_poses_xyzw", {}).items():
        body = root.find(f"worldbody/body[@name='{name}']")
        if body is None:
            raise ValueError(f"Unknown initialization root body: {name}")
        body.set("pos", " ".join(map(str, pose[:3])))
        body.set("quat", " ".join(map(str, [pose[6], *pose[3:6]])))
    if metadata.get("stages"):
        # Browser worker injects these welds before compiling; the saved XML
        # alone is therefore not the complete runtime model.
        equality = ET.SubElement(root, "equality")
        for index, part in enumerate(("object", "second")):
            anchor = f"freeze_anchor_{index}"
            ET.SubElement(root.find("worldbody"), "body", name=anchor, mocap="true")
            ET.SubElement(
                equality,
                "weld",
                name=f"freeze_{index}",
                body1=part,
                body2=anchor,
                active="false",
                relpose="0 0 0 1 0 0 0",
                anchor="0 0 0",
                solref="0.005 1",
                solimp="0.9995 0.9999 0.0001",
            )
    assets = {name: (bundle / name).read_bytes() for name in metadata["files"]}
    return mujoco.MjModel.from_xml_string(ET.tostring(root).decode(), assets)


class NativeAssembly:
    def __init__(
        self,
        bundle: Path,
        start: int = 0,
        sdf_initpoints: int | None = None,
        initialization: dict | None = None,
    ):
        self.base_metadata = json.loads((bundle / "manifest.json").read_text())
        if initialization is not None:
            self.base_metadata.update(initialization)
        self.model = load_model(bundle, self.base_metadata)
        if sdf_initpoints is not None:
            if sdf_initpoints < 1:
                raise ValueError("SDF sample count must be positive")
            self.model.opt.sdf_initpoints = sdf_initpoints
        self.data = mujoco.MjData(self.model)
        self.robot_actuator_ids = np.asarray(
            self.base_metadata.get("robot_actuator_ids", list(range(29)))
        )
        self.palm_id = self.model.body("link7").id
        self.tip_ids = [
            self.model.body(f"palmleft_{finger}_DP").id
            for finger in ("index", "middle", "ring", "thumb", "pinky")
        ]
        self.goal_id = self.model.body("goal_object").mocapid[0]
        self.reset(start)

    def reset(self, start=0):
        self.metadata = copy.deepcopy(self.base_metadata)
        self.stage_index = 0
        if self.metadata.get("stages"):
            self.metadata.update(self.metadata["stages"][0])
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.metadata["initial_qpos"]
        if self.metadata.get("initialization_pool"):
            self.data.qpos[29:] = self.metadata["initialization_pool"][start][
                "part_qpos"
            ]
        elif start:
            raise ValueError("This bundle has only canonical start 0")
        if "initial_qvel" in self.metadata:
            self.data.qvel[:] = self.metadata["initial_qvel"]
        self.targets = np.array(self.metadata["initial_ctrl"], dtype=np.float32)
        self.data.ctrl[self.robot_actuator_ids] = self.targets
        self.steps = self.successes = self.near = self.episode = 0
        self.retract = self.succeeded = self.failed = self.completed = False
        self.release_remaining = None
        self.handoffs = []
        self.distance = None
        self.set_goal()
        mujoco.mj_forward(self.model, self.data)

    @property
    def success_tolerance(self):
        if "withdrawal_tolerance_m" in self.metadata:
            return self.metadata["withdrawal_tolerance_m"]
        if self.base_metadata.get("stages") and self.stage_index == 0:
            return 0.0075
        return self.metadata.get("insertion_tolerance", 0.0075)

    @property
    def address(self):
        return self.metadata.get("object_qpos_address", 29)

    def set_goal(self):
        goals = self.metadata["goals"]
        self.goal = np.asarray(goals[min(self.successes, len(goals) - 1)])
        self.data.mocap_pos[self.goal_id] = self.goal[:3]
        self.data.mocap_quat[self.goal_id] = self.goal[[6, 3, 4, 5]]

    def geometry(self, offsets=None, yaw=0.0):
        if offsets is None:
            offsets = self.metadata["keypoint_offsets"]
        pose = self.data.qpos[self.address : self.address + 7]
        q = pose[[4, 5, 6, 3]]
        obj = np.asarray(offsets) @ rotation(q).T + pose[:3]
        yaw_q = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])
        goal = np.asarray(offsets) @ (rotation(yaw_q) @ rotation(self.goal[3:])).T
        return pose[:3], q, obj, goal + self.goal[:3]

    def observation(self):
        d, m, md = self.data, self.model, self.metadata
        limits = m.jnt_range[:29].astype(np.float32)
        joints = d.qpos[:29].astype(np.float32)
        normalized = 2 * (joints - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1
        palm = d.xpos[self.palm_id] + d.xmat[self.palm_id].reshape(3, 3) @ [
            0,
            -0.02,
            0.16,
        ]
        _, q, obj, goal = self.geometry(yaw=md.get("goal_yaw_bias", 0))
        tips = [
            d.xpos[i] + d.xmat[i].reshape(3, 3) @ [0.02, 0.002, 0] - palm
            for i in self.tip_ids
        ]
        values = np.concatenate(
            [
                normalized,
                d.qvel[:29],
                self.targets,
                palm,
                d.xquat[self.palm_id][[1, 2, 3, 0]],
                q,
                np.asarray(tips).ravel(),
                (obj - palm).ravel(),
                (obj - goal - md["goal_position_bias"]).ravel(),
                md["object_scales"],
            ]
        ).astype(np.float32)
        if values.shape != (140,):
            raise ValueError(f"Bad actor observation: {values.shape}")
        return np.append(np.clip(values, -10, 10), np.float32(md["block_id"]))

    def _handoff(self):
        md = self.metadata
        stage = copy.deepcopy(self.base_metadata["stages"][1])
        achieved_pos, achieved_q, _, _ = self.geometry()
        target = np.asarray(md["goals"][-1])
        conjugate = target[3:] * [-1, -1, -1, 1]
        delta = multiply(achieved_q, conjugate)
        offset = achieved_pos - rotation(delta) @ target[:3]
        stage["goals"] = [
            [*(rotation(delta) @ np.asarray(g[:3]) + offset), *multiply(delta, g[3:])]
            for g in stage["goals"]
        ]
        body = self.model.jnt_bodyid[
            np.flatnonzero(self.model.jnt_qposadr == self.address)[0]
        ]
        anchor = self.model.body("freeze_anchor_0").mocapid[0]
        self.data.mocap_pos[anchor] = self.data.xpos[body]
        self.data.mocap_quat[anchor] = self.data.xquat[body]
        self.data.eq_active[0] = 1
        self.metadata = {**copy.deepcopy(self.base_metadata), **stage}
        self.stage_index = 1
        self.successes = self.near = self.episode = 0
        self.retract = self.succeeded = self.failed = False
        self.handoffs.append(self.steps)
        self.set_goal()

    def step(self, action):
        md = self.metadata
        self.targets = action_targets(self.targets, action, self.model.jnt_range[:29])
        self.data.ctrl[self.robot_actuator_ids] = self.targets
        self.advance_physics()
        _, _, obj, goal = self.geometry(md.get("reward_offsets"))
        self.distance = float(np.linalg.norm(obj - goal, axis=1).max())
        pos = self.data.qpos[self.address : self.address + 3]
        tips = np.linalg.norm(self.data.xpos[self.tip_ids] - pos, axis=1)
        self.steps += 1
        self.episode += 1
        near = self.distance <= md.get("insertion_tolerance", 0.015)
        if md.get("near_goal_counter", "consecutive") == "accumulated":
            self.near += int(near)
        else:
            self.near = self.near + 1 if near else 0
        if self.near >= md.get("success_steps", 10) and not self.retract:
            self.successes += 1
            self.episode = 0
            if self.successes == len(md["goals"]):
                self.retract = True
            else:
                self.near = 0
        self.succeeded |= (
            self.retract
            and tips.mean() > 0.1
            and self.distance <= self.success_tolerance
        )
        self.failed |= (
            pos[2] < 0.1
            or (not self.retract and tips.max() > 1.5)
            or self.episode >= 601
        )
        self.set_goal()
        if (
            self.succeeded
            and self.base_metadata.get("stages")
            and self.stage_index == 0
        ):
            self._handoff()
        if self.succeeded and not self.failed:
            if self.release_remaining is None:
                self.release_remaining = md.get("release_follow_through_controls", 0)
            else:
                self.release_remaining = max(0, self.release_remaining - 1)
            if self.release_remaining == 0:
                self.completed = bool(
                    self.distance <= self.success_tolerance and tips.mean() > 0.1
                )
                self.failed |= not self.completed
        if not np.isfinite(self.data.qpos).all():
            raise FloatingPointError("Nonfinite physics state")

    def advance_physics(self):
        mujoco.mj_step(self.model, self.data, nstep=self.metadata["decimation"])
        mujoco.mj_forward(self.model, self.data)
