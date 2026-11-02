#!/usr/bin/env python3
"""Viser GUI base class for interactive precise-assembly evaluation.

`PegDynamicDemo` owns the viser scene (robot URDF, table, object/goal meshes,
keypoints, GUI widgets, and the pose-streaming loop). It is backend-agnostic:
a subclass implements `_load_env` to launch a rollout and stream poses back.
The Isaac Sim backend lives in `evaluation/eval_isaacsim.py`.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole"

CONTROL_DT = 1.0 / 60.0

# The Kuka + Sharpa's resting pose in the viewer (29 joints, canonical order). Other robots use
# their RobotSpec's default arm pose (see load_viewer_robot).
_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
DEFAULT_DOF_POS = np.zeros(29)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT

GOAL_MODES = ["preInsertAndFinal", "finalGoalOnly"]
DEFAULT_PROBLEM = "tight_insertion"

DEFAULT_INSERTION_SUCCESS_TOLERANCE = 0.01
DEFAULT_RETRACT_SUCCESS_TOLERANCE = 0.005

# Viewer geometry per task: the robot and table come from the task's YAML overlay, the robot's
# joint names and default pose from isaacsimenvs/tasks/play/robots.py (Kit-free).
DEFAULT_TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
TASK_YAMLS = {
    DEFAULT_TASK: "PreciseAssembly.yaml",
    "Isaacsimenvs-PreciseAssemblyFlexivWujiRight-Direct-v0": "PreciseAssemblyFlexivWujiRight.yaml",
}


@dataclass(frozen=True)
class ViewerRobot:
    """Everything the viewer draws that depends on the robot / its table setup."""

    name: str
    robot_urdf: Path
    joint_names: Tuple[str, ...]          # canonical (policy) order, as the sim streams them
    default_dof_pos: np.ndarray           # canonical order
    table_dims: Tuple[float, float, float]
    table_center: Tuple[float, float, float]  # env-local
    target_volume_mins: Tuple[float, float, float]
    target_volume_maxs: Tuple[float, float, float]


def _load_robot_specs() -> dict:
    """``robots.ROBOT_SPECS`` without importing isaacsimenvs (which pulls in Isaac Lab)."""
    import importlib.util

    path = REPO_ROOT / "isaacsimenvs" / "tasks" / "play" / "robots.py"
    spec = importlib.util.spec_from_file_location("_p2p_robots", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.ROBOT_SPECS


def load_viewer_robot(task: str) -> ViewerRobot:
    import xml.etree.ElementTree as ET

    import yaml

    if task not in TASK_YAMLS:
        raise KeyError(f"no viewer geometry for task {task!r}; known: {sorted(TASK_YAMLS)}")
    cfg = yaml.safe_load((REPO_ROOT / "isaacsimenvs" / "cfg" / "task" / TASK_YAMLS[task]).read_text())
    spec = _load_robot_specs()[cfg.get("robot", "kuka_sharpa")]
    if spec.name == "kuka_sharpa":
        default = DEFAULT_DOF_POS.copy()  # the Kuka's viewer pose, as released
    else:
        default = np.zeros(spec.num_joints)
        default[:7] = [spec.arm_default_joint_pos[n] for n in spec.arm_joint_names]
    table_urdf = REPO_ROOT / cfg["assets"]["table_urdf"]
    box = ET.parse(table_urdf).getroot().find("link/collision/geometry/box")
    rs = cfg["reset"]
    tx, ty = rs.get("table_reset_xy", (0.0, 0.0))
    return ViewerRobot(
        name=spec.name,
        robot_urdf=REPO_ROOT / cfg["assets"]["robot_urdf"],
        joint_names=tuple(spec.joint_names_canonical),
        default_dof_pos=default,
        table_dims=tuple(float(v) for v in box.get("size").split()),
        table_center=(float(tx), float(ty), float(rs["table_reset_z"])),
        target_volume_mins=tuple(rs["target_volume_mins"]),
        target_volume_maxs=tuple(rs["target_volume_maxs"]),
    )


# Problems whose URDF visuals carry real materials (textured scan, multi-material
# GLBs) and should be drawn with them. Every other problem keeps the flat-colour
# collision-geometry rendering.
MATERIAL_VIZ_PROBLEMS = frozenset({"fork_insertion", "plug_insertion"})


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


def _load_mesh_for_viz(asset_path: Path, prefer_visual: bool = False) -> trimesh.Trimesh:
    """Load a mesh for viser visualization. Handles URDFs by composing
    each link's parent-joint origin with each collision/visual ``<origin>``,
    so multi-link fixture URDFs (joint-origin-driven layouts like our
    fabrica/fmb insertion_fixtures) render at the correct assembled pose."""
    return trimesh.util.concatenate(_load_viz_parts(asset_path, prefer_visual))


def _load_scene_for_viz(asset_path: Path) -> Optional[trimesh.Scene]:
    """Visual geometry as a Scene with per-part materials intact.

    ``trimesh.util.concatenate`` keeps only one material, so a multi-material
    receptacle would collapse to grey; shipping the Scene as GLB keeps them.
    """
    try:
        parts = _load_viz_parts(asset_path, prefer_visual=True)
    except Exception:
        return None
    if not parts:
        return None
    scene = trimesh.Scene()
    for i, m in enumerate(parts):
        scene.add_geometry(m, geom_name=f"part_{i:02d}")
    return scene


def _load_viz_parts(asset_path: Path, prefer_visual: bool = False) -> list:
    """Per-part meshes of an asset (URDF or mesh file), transforms applied."""
    if asset_path.suffix in (".obj", ".stl", ".ply"):
        return [trimesh.load(str(asset_path), force="mesh")]

    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    tree = ET.parse(str(asset_path))
    root = tree.getroot()

    def _origin_T(elem):
        T = np.eye(4)
        if elem is None:
            return T
        xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
        T[:3, 3] = xyz
        if any(v != 0 for v in rpy):
            T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        return T

    # Build child_link -> (parent_T, parent_link). Most of our fixture
    # URDFs are flat (parent = "root"), so a single chain is sufficient.
    parent_T_of: dict = {}
    parent_link_of: dict = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        parent_T_of[child.get("link")] = _origin_T(joint.find("origin"))
        parent_link_of[child.get("link")] = parent.get("link")

    def _link_world_T(link_name):
        T = np.eye(4)
        cur = link_name
        guard = 0
        while cur in parent_T_of and guard < 32:
            T = parent_T_of[cur] @ T
            cur = parent_link_of.get(cur, "")
            guard += 1
        return T

    def _collect_geom(link_elem, prefer_tag):
        out = []
        link_T = _link_world_T(link_elem.get("name", ""))
        for elem in link_elem.findall(prefer_tag):
            geom = elem.find("geometry")
            if geom is None:
                continue
            local_T = _origin_T(elem.find("origin"))
            box = geom.find("box")
            mesh_elem = geom.find("mesh")
            if box is not None:
                m = trimesh.creation.box(
                    extents=[float(v) for v in box.get("size").split()]
                )
            elif mesh_elem is not None:
                mesh_file = asset_path.parent / mesh_elem.get("filename")
                if not mesh_file.exists():
                    continue
                if not prefer_visual:
                    m = trimesh.load(str(mesh_file), force="mesh")
                else:
                    # Keep a Scene's sub-geometries separate: merging them
                    # drops all but one material.
                    loaded = trimesh.load(str(mesh_file))
                    sub = ([g for g in loaded.geometry.values()
                            if isinstance(g, trimesh.Trimesh)]
                           if isinstance(loaded, trimesh.Scene) else [loaded])
                    for g in sub:
                        g = g.copy()
                        g.apply_transform(link_T @ local_T)
                        out.append(g)
                    continue
            else:
                continue
            m.apply_transform(link_T @ local_T)
            out.append(m)
        return out

    order = ("visual", "collision") if prefer_visual else ("collision", "visual")
    meshes = []
    for tag in order:
        for link in root.findall("link"):
            meshes.extend(_collect_geom(link, tag))
        if meshes:
            return meshes
    return [trimesh.creation.box(extents=(0.08, 0.08, 0.01))]


def _asset_abs(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / "assets" / path


def _asset_rel(path_like) -> str:
    path = Path(path_like)
    if path.is_absolute():
        try:
            return path.relative_to(REPO_ROOT / "assets").as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _registered_problem_names():
    from evaluation.problems import PROBLEM_REGISTRY

    return sorted(PROBLEM_REGISTRY)


def _resolve_problem_assets(problem_name: str) -> Tuple[str, str]:
    from evaluation.objects import NAME_TO_OBJECT
    from evaluation.problems import PROBLEM_REGISTRY

    if problem_name not in PROBLEM_REGISTRY:
        raise KeyError(
            f"Unknown problem {problem_name!r}; known: {sorted(PROBLEM_REGISTRY)}"
        )
    problem = PROBLEM_REGISTRY[problem_name]
    obj = NAME_TO_OBJECT.get(problem.insertion_object_name)
    if obj is None:
        raise KeyError(
            f"insertion_object_name {problem.insertion_object_name!r} not in NAME_TO_OBJECT"
        )
    return _asset_rel(obj.urdf_path), _asset_rel(problem.receptive_urdf)


# ===================================================================
# SUBPROCESS -- IsaacGym simulation
# ===================================================================

class PegDynamicDemo:
    def __init__(self, policies: Dict[str, Tuple[str, str]],
                 port: int = 8043, headless: bool = True,
                 goal_mode: str = "preInsertAndFinal",
                 random_goal_fraction: float = 0.0,
                 initial_policy: Optional[str] = None,
                 extra_overrides: Optional[dict] = None,
                 initial_problem: Optional[str] = None,
                 object_urdf: Optional[str] = None,
                 hole_urdf: Optional[str] = None,
                 policy_tasks: Optional[Dict[str, str]] = None,
                 policy_problems: Optional[Dict[str, str]] = None):
        """``policy_tasks`` maps a policy name to the task (robot) it was trained on; policies
        it omits use ``self.task``. With more than one task the GUI gets a Robot dropdown.
        ``policy_problems`` maps a policy name to its problem (default: the name itself)."""
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}")
        if not policies:
            raise ValueError("policies dict must be non-empty")
        self.problem_names = _registered_problem_names()
        if not self.problem_names:
            raise ValueError("PROBLEM_REGISTRY is empty")
        self.policies = policies
        # Subclasses may set self.task before calling this __init__.
        default_task = getattr(self, "task", DEFAULT_TASK)
        self.policy_tasks = {name: (policy_tasks or {}).get(name, default_task) for name in policies}
        self.policy_problems = {name: (policy_problems or {}).get(name, name) for name in policies}
        self.initial_policy = initial_policy if initial_policy in policies else next(iter(policies))
        self.task = self.policy_tasks[self.initial_policy]
        self._viewer_robots = {
            task: load_viewer_robot(task) for task in dict.fromkeys(self.policy_tasks.values())
        }
        self.viewer_robot = self._viewer_robots[self.task]
        if initial_problem in self.problem_names:
            self.problem_name = initial_problem
        elif DEFAULT_PROBLEM in self.problem_names:
            self.problem_name = DEFAULT_PROBLEM
        else:
            self.problem_name = self.problem_names[0]
        self.port = port
        self.headless = headless
        self.goal_mode = goal_mode
        self.random_goal_fraction = random_goal_fraction
        self.extra_overrides = extra_overrides or {}
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        # Asset paths are relative to assets/ root unless absolute. CLI asset
        # overrides only affect the initial render; choosing a problem from the
        # dropdown re-derives both meshes from PROBLEM_REGISTRY.
        self._object_urdf_rel = ""
        self._hole_urdf_rel = ""
        self._object_urdf_abs = REPO_ROOT / "assets"
        self._hole_urdf_abs = REPO_ROOT / "assets"
        if object_urdf is not None or hole_urdf is not None:
            problem_object, problem_hole = _resolve_problem_assets(self.problem_name)
            self._set_asset_paths(object_urdf or problem_object, hole_urdf or problem_hole)
        else:
            self._set_problem_assets(self.problem_name)

        self._proc = None
        self._conn = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

        self.ep_count = 0
        self._peak_force = 0.0

        self.robot = None
        self._dyn = []
        self._obj_frame = None
        self._goal_frame = None
        self._hole_frame = None
        self._object_viz = None
        self._goal_viz = None
        self._hole_viz = None
        self._obj_keypoints = []
        self._goal_keypoints = []

        self._reload_problem_meshes()
        self._target_vol_box = None

        self._build_gui()
        self._setup_static_scene()

    def _set_asset_paths(self, object_urdf: str, hole_urdf: str) -> None:
        self._object_urdf_rel = _asset_rel(object_urdf)
        self._hole_urdf_rel = _asset_rel(hole_urdf)
        self._object_urdf_abs = _asset_abs(self._object_urdf_rel)
        self._hole_urdf_abs = _asset_abs(self._hole_urdf_rel)

    def _set_problem_assets(self, problem_name: str) -> None:
        object_urdf, hole_urdf = _resolve_problem_assets(problem_name)
        self.problem_name = problem_name
        self._set_asset_paths(object_urdf, hole_urdf)

    def _reload_problem_meshes(self) -> None:
        self._material_viz = self.problem_name in MATERIAL_VIZ_PROBLEMS
        self._hole_mesh = _load_mesh_for_viz(self._hole_urdf_abs, self._material_viz)
        self._object_mesh = _load_mesh_for_viz(self._object_urdf_abs, self._material_viz)
        # Exported once here: the fork scan's GLB is ~4 MB.
        self._hole_glb = self._object_glb = None
        if self._material_viz:
            hole_scene = _load_scene_for_viz(self._hole_urdf_abs)
            object_scene = _load_scene_for_viz(self._object_urdf_abs)
            if hole_scene is not None:
                self._hole_glb = hole_scene.export(file_type="glb")
            if object_scene is not None:
                self._object_glb = object_scene.export(file_type="glb")

    def _task_policy_names(self, task: str) -> list:
        return [name for name, t in self.policy_tasks.items() if t == task]

    def _resolve_policy(self, task: str, problem_name: str) -> Optional[str]:
        """The policy is fixed by (robot, problem): the robot's policy for that problem, or its
        only policy (e.g. a single --checkpoint-path); None if neither exists."""
        names = self._task_policy_names(task)
        for name in names:
            if self.policy_problems[name] == problem_name:
                return name
        return names[0] if len(names) == 1 else None

    def _on_problem_change(self) -> None:
        self.policy_name = self._resolve_policy(self.task, self._dd_problem.value)
        # Insertion tolerance follows the problem unless pinned on the command line.
        if not hasattr(self, "_sl_insertion_tol"):  # still building the GUI
            return
        override = getattr(self, "_insertion_tol_override", None)
        if override is not None:
            self._sl_insertion_tol.value = float(override)
        else:
            from evaluation.problems import default_insertion_success_tolerance

            self._sl_insertion_tol.value = default_insertion_success_tolerance(
                self._dd_problem.value
            )
        if self.policy_name is None:
            self._md_policy.content = f"**Policy:** none for {self.viewer_robot.name} / {self._dd_problem.value}"
        else:
            self._md_policy.content = f"**Policy:** `{self.policies[self.policy_name][1]}`"

    def _on_robot_change(self) -> None:
        """Swap the drawn robot/table and the policy; the next Load uses the new robot."""
        task = next(t for t, vr in self._viewer_robots.items() if vr.name == self._dd_robot.value)
        if task == self.task:
            return
        self._kill_subprocess()
        self._clear_dynamic()
        self.task = task
        self.viewer_robot = self._viewer_robots[task]
        self._build_robot_viz()
        self._on_problem_change()
        self._md_status.content = (
            f"**Status:** Robot set to {self.viewer_robot.name}; press Load / reload env"
        )

    def _build_gui(self):
        self.server.gui.add_markdown(
            "# Peg-in-Hole Dynamic Eval\n"
            "### Pretrained policy with dynamic hole placement"
        )

        with self.server.gui.add_folder("Task Selection", expand_by_default=True):
            if len(self._viewer_robots) > 1:
                self._dd_robot = self.server.gui.add_dropdown(
                    "Robot", options=[vr.name for vr in self._viewer_robots.values()],
                    initial_value=self.viewer_robot.name,
                )
                self._dd_robot.on_update(lambda _: self._on_robot_change())
            self._dd_problem = self.server.gui.add_dropdown(
                "Problem", options=self.problem_names, initial_value=self.problem_name,
            )
            self._md_policy = self.server.gui.add_markdown("**Policy:** --")
            self._on_problem_change()
            self._dd_problem.on_update(lambda _: self._on_problem_change())
            self._dd_goal_mode = self.server.gui.add_dropdown(
                "Goal mode", options=GOAL_MODES, initial_value=self.goal_mode,
            )
            self._sl_rgf = self.server.gui.add_slider(
                "Random goal frac", min=0.0, max=1.0, step=0.1,
                initial_value=self.random_goal_fraction,
            )
            self._sl_insertion_tol = self.server.gui.add_slider(
                "Insertion tol (m)", min=0.001, max=0.02, step=0.001,
                initial_value=DEFAULT_INSERTION_SUCCESS_TOLERANCE,
            )
            self._sl_retract_tol = self.server.gui.add_slider(
                "Retract tol (m)", min=0.001, max=0.01, step=0.001,
                initial_value=DEFAULT_RETRACT_SUCCESS_TOLERANCE,
            )
            self._btn_load = self.server.gui.add_button("Load / reload env")
            self._btn_load.on_click(lambda _: self._load_env())
            self._md_status = self.server.gui.add_markdown("**Status:** Ready")

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop")
            self._btn_stop.on_click(lambda _: self._cmd_stop())

        with self.server.gui.add_folder("Display", expand_by_default=True):
            self._cb_keypoints = self.server.gui.add_checkbox("Show keypoints", initial_value=False)
            self._cb_keypoints.on_update(lambda _: self._apply_keypoint_visibility())
            self._cb_goal = self.server.gui.add_checkbox("Show goal", initial_value=True)
            self._cb_goal.on_update(lambda _: self._apply_goal_visibility())
            self._sl_goal_opacity = self.server.gui.add_slider(
                "Goal opacity", min=0.0, max=1.0, step=0.05, initial_value=0.5,
            )
            self._sl_goal_opacity.on_update(lambda _: self._apply_goal_visibility())
            self._sl_fixture_opacity = self.server.gui.add_slider(
                "Fixture opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0,
            )
            self._sl_fixture_opacity.on_update(lambda _: self._apply_fixture_opacity())
            self._sl_object_opacity = self.server.gui.add_slider(
                "Object opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0,
            )
            self._sl_object_opacity.on_update(lambda _: self._apply_object_opacity())
            self._cb_target_vol = self.server.gui.add_checkbox("Show target volume", initial_value=False)
            self._cb_target_vol.on_update(lambda _: self._toggle_target_volume())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_hole = self.server.gui.add_markdown("**Hole pos:** --")
            self._md_object_pose = self.server.gui.add_markdown("**Object pose:** --")
            self._md_goal_pose = self.server.gui.add_markdown("**Goal pose:** --")
            self._md_pose_delta = self.server.gui.add_markdown("**Object-goal z dist:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_diag = self.server.gui.add_markdown("**Goal dist:** --")
            self._md_retract = self.server.gui.add_markdown("**Retract:** --")
            self._md_force = self.server.gui.add_markdown("**Table force:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")

    def _setup_static_scene(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self._table_frame_default_wxyz = (1.0, 0.0, 0.0, 0.0)
        self._table_frame = self.server.scene.add_frame(
            "/table",
            position=self.viewer_robot.table_center,
            wxyz=self._table_frame_default_wxyz,
            show_axes=False,
        )
        self._table_wood = None
        self._build_robot_viz()

    def _build_robot_viz(self):
        """(Re)draw everything that depends on ``self.viewer_robot``: robot, table, target volume."""
        vr = self.viewer_robot
        if self.robot is not None:
            self.robot.remove()
        self.robot = ViserUrdf(self.server, vr.robot_urdf, root_node_name="/robot")
        # The sim streams joints in the policy's canonical order; ViserUrdf wants its own.
        self._viser_from_canon = np.array(
            [vr.joint_names.index(n) for n in self.robot.get_actuated_joint_names()]
        )
        self.set_robot_joints(vr.default_dof_pos)

        self._table_frame_default_pos = vr.table_center
        self._table_frame.position = vr.table_center
        self._table_frame.wxyz = self._table_frame_default_wxyz
        # Base table dims = the URDF box (X, Y, Z) before any scale variants.
        # Stored so subclasses can re-issue the box with scaled dimensions.
        self._table_base_dims: tuple[float, float, float] = vr.table_dims
        if self._table_wood is not None:
            self._table_wood.remove()
        self._table_wood = self.server.scene.add_box(
            "/table/wood", color=(180, 130, 70),
            dimensions=self._table_base_dims, position=(0, 0, 0),
            side="double", opacity=0.9,
        )
        if self._target_vol_box is not None:
            self._target_vol_box.remove()
            self._target_vol_box = None
            self._toggle_target_volume()

    def set_robot_joints(self, joint_pos_canonical) -> None:
        self.robot.update_cfg(np.asarray(joint_pos_canonical)[self._viser_from_canon])

    def _update_table_viz(
        self,
        scale_x_range: tuple[float, float] = (1.0, 1.0),
        scale_y_range: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """Re-issue ``/table/wood`` sized to the max-extent of the scale range.

        Per-env scale variants are baked at scene-init in the sim worker; viser
        can only show one mesh, so we draw the largest box the scale dropdown
        permits as an envelope. Identity scale (default) reproduces the
        original mesh exactly.
        """
        sx = max(scale_x_range)
        sy = max(scale_y_range)
        bx, by, bz = self._table_base_dims
        try:
            self._table_wood.remove()
        except Exception:
            pass
        self._table_wood = self.server.scene.add_box(
            "/table/wood", color=(180, 130, 70),
            dimensions=(bx * sx, by * sy, bz), position=(0, 0, 0),
            side="double", opacity=0.9,
        )

    def _update_table_pose(
        self,
        pos_local: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> None:
        """Move the ``/table`` frame to the displayed env's actual table pose.

        ``pos_local`` is the env-local position (world - env_origin). The
        table height offset is already baked into the URDF box visual position
        (z=0.15), so the frame z should be the table's actual world z minus
        the box-half-height inherent in ``_table_base_dims[2] / 2``; in
        practice the sim publishes ``root_pos_w`` at the center of the box,
        and viser's frame is the box parent — we use the published z as-is.
        """
        self._table_frame.position = tuple(float(v) for v in pos_local)
        self._table_frame.wxyz = tuple(float(v) for v in quat_wxyz)

    def _clear_dynamic(self):
        for h in reversed(self._dyn):
            try:
                h.remove()
            except Exception:
                pass
        self._dyn.clear()
        self._obj_frame = self._goal_frame = self._hole_frame = None
        self._object_viz = self._goal_viz = None
        self._hole_viz = None
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()

    def _add_object_viz(self, node_name: str, color, opacity=1.0):
        """Add object mesh to scene from the preloaded trimesh.

        Reconstructing ViserUrdf objects inside the GUI load callback can block
        reloads for detailed URDFs. A single composed mesh is enough for this
        eval overlay and lets the goal ghost opacity update live.
        """
        rgb = color[:3] if len(color) >= 3 else color
        # The object (not the goal ghost) is drawn with its own materials when
        # opaque; add_glb has no opacity, so translucent falls back to flat.
        if (node_name == "/object" and opacity >= 0.999
                and getattr(self, "_object_glb", None) is not None):
            return self.server.scene.add_glb(f"{node_name}/mesh", glb_data=self._object_glb)
        verts = np.array(self._object_mesh.vertices, dtype=np.float32)
        faces = np.array(self._object_mesh.faces, dtype=np.uint32)
        return self.server.scene.add_mesh_simple(
            f"{node_name}/mesh", vertices=verts, faces=faces,
            color=rgb, opacity=opacity, side="double",
            cast_shadow=False, receive_shadow=False,
        )

    def _set_viz_visible(self, viz, visible):
        if viz is None:
            return
        if hasattr(viz, "show_visual"):
            viz.show_visual(visible)
            return
        try:
            viz.visible = visible
        except AttributeError:
            pass

    def _set_viz_opacity(self, viz, opacity):
        if viz is None:
            return
        handles = getattr(viz, "_meshes", [viz])
        for handle in handles:
            try:
                handle.opacity = opacity
            except AttributeError:
                pass

    def _setup_scene_objects(self):
        self._clear_dynamic()

        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.05, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)
        self._object_viz = self._add_object_viz(
            "/object", (204, 40, 40), opacity=self._sl_object_opacity.value,
        )
        self._dyn.append(self._object_viz)

        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.05, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)
        self._goal_viz = self._add_object_viz(
            "/goal", (0, 255, 0), opacity=self._sl_goal_opacity.value,
        )
        self._dyn.append(self._goal_viz)

        self._hole_frame = self.server.scene.add_frame(
            "/hole",
            position=(
                self.viewer_robot.table_center[0],
                self.viewer_robot.table_center[1],
                self.viewer_robot.table_center[2] + self.viewer_robot.table_dims[2] / 2,
            ),
            wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self._dyn.append(self._hole_frame)
        self._hole_viz = self._add_hole_viz()
        self._dyn.append(self._hole_viz)
        self._apply_goal_visibility()
        self._apply_fixture_opacity()
        self._apply_object_opacity()

    def _add_hole_viz(self):
        opacity = float(self._sl_fixture_opacity.value)
        if opacity >= 0.999 and getattr(self, "_hole_glb", None) is not None:
            return self.server.scene.add_glb("/hole/mesh", glb_data=self._hole_glb)
        return self.server.scene.add_mesh_simple(
            "/hole/mesh",
            vertices=np.array(self._hole_mesh.vertices, dtype=np.float32),
            faces=np.array(self._hole_mesh.faces, dtype=np.uint32),
            color=(120, 120, 120),
            opacity=opacity,
        )

    def _update_hole_viz(self, hole_pose):
        # Accept either legacy 3-vec position-only or 7-vec pose (x,y,z,qx,qy,qz,qw).
        if self._hole_frame is None:
            return
        hp = np.asarray(hole_pose, dtype=np.float32).reshape(-1)
        if hp[2] < 0:
            self._hole_frame.visible = False
            return
        self._hole_frame.visible = True
        self._hole_frame.position = (float(hp[0]), float(hp[1]), float(hp[2]))
        if hp.shape[0] >= 7:
            qx, qy, qz, qw = hp[3], hp[4], hp[5], hp[6]
            self._hole_frame.wxyz = (float(qw), float(qx), float(qy), float(qz))

    def _setup_keypoints(self, num_keypoints):
        for kp in self._obj_keypoints + self._goal_keypoints:
            try:
                kp.remove()
            except Exception:
                pass
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()
        for i in range(num_keypoints):
            self._obj_keypoints.append(
                self.server.scene.add_icosphere(f"/obj_kp/{i}", radius=0.005, color=(255, 0, 0))
            )
            self._goal_keypoints.append(
                self.server.scene.add_icosphere(f"/goal_kp/{i}", radius=0.005, color=(0, 255, 0), opacity=0.5)
            )
        self._apply_keypoint_visibility()

    def _apply_keypoint_visibility(self):
        visible = self._cb_keypoints.value
        for kp in self._obj_keypoints:
            kp.visible = visible
        self._apply_goal_visibility()

    def _apply_goal_visibility(self):
        goal_visible = self._cb_goal.value and self._sl_goal_opacity.value > 0.0
        opacity = float(self._sl_goal_opacity.value)
        if self._goal_frame is not None:
            self._goal_frame.visible = goal_visible
        self._set_viz_visible(self._goal_viz, goal_visible)
        self._set_viz_opacity(self._goal_viz, opacity)
        for kp in self._goal_keypoints:
            kp.visible = goal_visible and self._cb_keypoints.value
            try:
                kp.opacity = opacity
            except AttributeError:
                pass

    def _apply_fixture_opacity(self):
        if self._hole_viz is not None and getattr(self, "_hole_glb", None) is not None:
            # GLB handles have no opacity: rebuild (GLB when opaque, flat otherwise).
            try:
                self._hole_viz.remove()
            except Exception:
                pass
            try:
                self._dyn.remove(self._hole_viz)
            except ValueError:
                pass
            self._hole_viz = self._add_hole_viz()
            self._dyn.append(self._hole_viz)
            return
        if self._hole_viz is not None:
            self._set_viz_opacity(
                self._hole_viz, float(self._sl_fixture_opacity.value)
            )

    def _apply_object_opacity(self):
        if self._object_viz is not None:
            try:
                self._object_viz.remove()
            except Exception:
                pass
            try:
                self._dyn.remove(self._object_viz)
            except ValueError:
                pass
            self._object_viz = self._add_object_viz(
                "/object", (204, 40, 40),
                opacity=float(self._sl_object_opacity.value),
            )
            self._dyn.append(self._object_viz)

    def _toggle_target_volume(self):
        show = self._cb_target_vol.value
        if show and self._target_vol_box is None:
            tv_min = np.array(self.viewer_robot.target_volume_mins)
            tv_max = np.array(self.viewer_robot.target_volume_maxs)
            center = (tv_min + tv_max) / 2
            dims = tv_max - tv_min
            self._target_vol_box = self.server.scene.add_box(
                "/target_volume",
                color=(100, 255, 100),
                dimensions=tuple(dims.tolist()),
                position=tuple(center.tolist()),
                side="double",
                opacity=0.08,
            )
        if self._target_vol_box is not None:
            self._target_vol_box.visible = show

    # ── Subprocess management ────────────────────────────────────

    def _kill_subprocess(self):
        if self._conn is not None:
            try:
                self._conn.send("quit")
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
            self._proc = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

    def _load_env(self):
        # The base demo has no rollout backend in this release; the Isaac Sim
        # subclass (evaluation/eval_isaacsim.py) overrides this to launch a
        # worker subprocess and stream poses back to the viewer.
        raise NotImplementedError(
            "PegDynamicDemo._load_env must be overridden by a backend subclass."
        )

    def _send(self, msg):
        if self._conn is not None:
            try:
                self._conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    def _cmd_run(self):
        if not self._env_ready:
            self._md_status.content = "**Status:** Load an environment first."
            return
        if self._episode_running:
            return
        self._episode_running = True
        self._is_paused = False
        self._btn_pause.name = "Pause"
        self._md_status.content = "**Status:** Running episode..."
        self._md_retract.content = "**Retract:** --"
        self._peak_force = 0.0
        self._send("run")

    def _cmd_pause(self):
        if not self._episode_running:
            return
        self._is_paused = not self._is_paused
        self._send("pause" if self._is_paused else "resume")
        self._btn_pause.name = "Resume" if self._is_paused else "Pause"

    def _cmd_stop(self):
        if self._episode_running:
            self._send("stop")

    @staticmethod
    def _pose_status(label, pose):
        pose = np.asarray(pose, dtype=np.float32)
        xyz = ", ".join(f"{v:+.4f}" for v in pose[:3])
        quat = ", ".join(f"{v:+.4f}" for v in pose[3:7])
        return f"**{label}:** xyz [{xyz}]  \nquat_xyzw [{quat}]"

    def _update_viz(self, state_tuple):
        joint_pos, obj_pose, goal_pose = state_tuple[0], state_tuple[1], state_tuple[2]
        self.set_robot_joints(joint_pos)

        if self._obj_frame is not None:
            self._obj_frame.position = tuple(obj_pose[:3])
            self._obj_frame.wxyz = quat_xyzw_to_wxyz(obj_pose[3:7])
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(goal_pose[:3])
            self._goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])
        self._md_object_pose.content = self._pose_status("Object pose", obj_pose)
        self._md_goal_pose.content = self._pose_status("Goal pose", goal_pose)
        z_delta = float(obj_pose[2] - goal_pose[2])
        self._md_pose_delta.content = (
            f"**Object-goal z dist:** {z_delta * 1000:+.2f} mm "
            f"(abs {abs(z_delta) * 1000:.2f} mm)"
        )

        if len(state_tuple) > 3:
            obj_kps, goal_kps = state_tuple[3], state_tuple[4]
            for handle, pos in zip(self._obj_keypoints, obj_kps):
                handle.position = tuple(pos)
            for handle, pos in zip(self._goal_keypoints, goal_kps):
                handle.position = tuple(pos)

        if len(state_tuple) > 15:
            hole_pos = state_tuple[15]
            self._update_hole_viz(hole_pos)
            is_rg = state_tuple[16] if len(state_tuple) > 16 else False
            mode_str = "RANDOM GOAL" if is_rg else "INSERTION"
            if hole_pos[2] < 0:
                self._md_hole.content = f"**Hole pos:** hidden (random-goal mode) | **Mode:** {mode_str}"
            else:
                self._md_hole.content = (
                    f"**Hole pos:** ({hole_pos[0]:.3f}, {hole_pos[1]:.3f}, {hole_pos[2]:.3f})"
                    f" | **Mode:** {mode_str}"
                )

    def _handle(self, msg):
        tag = msg[0]
        if tag == "ready":
            init_state = msg[1]
            if len(init_state) > 3:
                self._setup_keypoints(init_state[3].shape[0])
            self._update_viz(init_state)
            self._env_ready = True
            self._md_status.content = "**Status:** Ready -- click **Run Episode**"
            print("[launcher] Environment ready")

        elif tag == "table_scale":
            # Worker reports the actual (sx, sy) of the displayed env's table.
            sx, sy = float(msg[1]), float(msg[2])
            self._update_table_viz(
                scale_x_range=(sx, sx), scale_y_range=(sy, sy),
            )
            print(f"[launcher] table_scale received: sx={sx:.3f} sy={sy:.3f}")

        elif tag == "table_pose":
            # Worker reports env_id's env-local table pose (after reset).
            px, py, pz = float(msg[1]), float(msg[2]), float(msg[3])
            qw, qx, qy, qz = (float(msg[4]), float(msg[5]),
                              float(msg[6]), float(msg[7]))
            self._update_table_pose((px, py, pz), (qw, qx, qy, qz))
            print(f"[launcher] table_pose received: "
                  f"pos=({px:.3f},{py:.3f},{pz:.3f}) "
                  f"quat=({qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f})")

        elif tag == "state":
            state, successes, max_succ, step = msg[1], msg[2], msg[3], msg[4]
            latched_retract = msg[5] if len(msg) > 5 else False
            self._update_viz(state)
            pct = 100 * successes / max_succ if max_succ > 0 else 0
            self._md_prog.content = (
                f"**Time:** {step / 60.0:.1f}s &nbsp;|&nbsp; "
                f"**Goal:** {successes}/{max_succ} ({pct:.0f}%)"
            )
            if len(state) >= 8:
                retract_phase, retract_ok, mean_ft_dist = state[5], state[6], state[7]
                retract_ok = retract_ok or latched_retract
                if retract_ok:
                    self._md_retract.content = f"**Retract:** SUCCESS (hand dist {mean_ft_dist:.3f}m)"
                elif retract_phase:
                    self._md_retract.content = f"**Retract:** IN PROGRESS (hand dist {mean_ft_dist:.3f}m)"
                else:
                    self._md_retract.content = f"**Retract:** not yet (hand dist {mean_ft_dist:.3f}m)"
            if len(state) >= 14:
                kp_max_dist = state[8]
                tol_m = state[9]
                near_steps = state[10]
                progress = state[11]
                max_ep_len = state[12]
                reset_pending = state[13]
                in_tol = "Y" if kp_max_dist <= tol_m else "N"
                self._md_diag.content = (
                    f"**Goal dist:** {kp_max_dist*1000:.1f} mm {in_tol}  "
                    f"&nbsp;(tol {tol_m*1000:.1f} mm)  "
                    f"&nbsp;near-goal-steps: **{near_steps}**  \n"
                    f"**progress_buf:** {progress}/{max_ep_len}  "
                    f"&nbsp;reset_buf: **{reset_pending}**"
                )
            if len(state) >= 15:
                force_vec = np.asarray(state[14], dtype=np.float32)
                force_mag = float(np.linalg.norm(force_vec))
                if force_mag > self._peak_force:
                    self._peak_force = force_mag
                self._md_force.content = (
                    f"**Table force:** {force_mag:.2f} N  "
                    f"&nbsp;(peak: **{self._peak_force:.2f} N**)  \n"
                    f"&nbsp;[fx, fy, fz] = "
                    f"[{force_vec[0]:+.2f}, {force_vec[1]:+.2f}, {force_vec[2]:+.2f}] N"
                )

        elif tag == "done":
            goal_pct, steps, retract_ok = msg[1], msg[2], msg[3]
            self._episode_running = False
            self.ep_count += 1
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Last goal:** {goal_pct:.0f}% &nbsp;|&nbsp; "
                f"**Last time:** {steps / 60.0:.1f}s"
            )
            self._md_status.content = (
                f"**Status:** Done -- {steps / 60.0:.1f}s, {goal_pct:.0f}% goals"
                f" | Retract {'OK' if retract_ok else 'FAIL'}"
            )
            self._md_retract.content = f"**Retract:** {'SUCCESS' if retract_ok else 'FAILED'}"
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals, {steps / 60.0:.1f}s")

        elif tag == "stopped":
            self._episode_running = False
            self._md_status.content = "**Status:** Episode stopped."

        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
            self._md_status.content = f"**Status:** Error -- {msg[1][:200]}"
            print(f"[launcher] Subprocess error:\n{msg[1]}")

    def _poll(self):
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            self._conn = None
            if self._proc is not None and not self._proc.is_alive():
                self._md_status.content = "**Status:** Subprocess exited unexpectedly."
                self._proc = None
                self._env_ready = False
                self._episode_running = False

    def run(self):
        print()
        print(f"  Peg-in-Hole Dynamic Eval   http://localhost:{self.port}")
        print()
        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_subprocess()

