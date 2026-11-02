"""Peg-in-hole scene setup."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane

from isaacsimenvs.tasks.play.utils.scene_utils import (
    _apply_self_collision_filters,
    _bake_usd,
    _convert_urdf_to_usd,
    _log_scene_step,
    _materialize_env_prims,
    _robot_joint_drive_cfg,
    build_robot_articulation_usd_cfg,
    hide_goal_viz_for_student_camera,
    setup_student_camera,
    _urdf_box_size,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _asset_path(path: str | Path) -> str:
    asset_path = Path(path)
    if not asset_path.is_absolute():
        asset_path = REPO_ROOT / asset_path
    return str(asset_path)


def _pih_rigid_object_cfg(prim_path: str, usd_path: str) -> RigidObjectCfg:
    """Single-USD RigidObject spawn using `UsdFileCfg` (not `MultiUsdFileCfg`).

    Why bypass `MultiUsdFileCfg` even though our list has just one entry:
    `MultiUsdFileCfg` sets the global carb flag
    `/isaaclab/spawn/multi_assets=True`, which triggers IsaacLab's
    "varying assets" warning under `replicate_physics=True` and may take a
    slower clone path under the hood. `UsdFileCfg` keeps the regex
    prim_path (so the Articulation/RigidObject discovers all envs) but
    avoids the multi-asset carb branch.
    """
    return RigidObjectCfg(prim_path=prim_path, spawn=UsdFileCfg(usd_path=usd_path))


def setup_scene(env) -> None:
    """Build robot, narrow table, dynamic hole fixture, object, goal, and light."""
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(setup_t0, f"peg-in-hole setup start num_envs={env.num_envs}")

    env._tmp_asset_dir = tempfile.mkdtemp(prefix="precise_assembly_assets_")
    env._object_urdf_paths = [_asset_path(env._pih_object_urdf_abs)]
    env._hole_urdf_paths = [_asset_path(env._pih_receptive_urdf_abs)]
    env._table_urdf_paths = [_asset_path(assets_cfg.table_urdf)]

    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    object_raw_usd = _convert_urdf_to_usd(
        _asset_path(env._pih_object_urdf_abs), usd_work_dir, fix_base=False
    )
    object_usd_path = _bake_usd(
        object_raw_usd,
        bake_root,
        "object",
        props=dict(
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=1000.0,
            rb_solver_position_iterations=4,
            rb_solver_velocity_iterations=0,
            articulation_enabled=False,
        ),
    )
    goalviz_usd_path = _bake_usd(
        object_raw_usd,
        bake_root,
        "goalviz",
        props=dict(
            kinematic_enabled=True,
            disable_gravity=True,
            articulation_enabled=False,
            rb_solver_position_iterations=4,
            rb_solver_velocity_iterations=0,
        ),
        collision_enabled=False,
    )

    hole_usd_path = _bake_usd(
        _convert_urdf_to_usd(
            _asset_path(env._pih_receptive_urdf_abs),
            usd_work_dir / "hole",
            fix_base=False,
        ),
        bake_root,
        "hole",
        props=dict(
            kinematic_enabled=True,
            disable_gravity=True,
            articulation_enabled=False,
            rb_solver_position_iterations=4,
            rb_solver_velocity_iterations=0,
        ),
    )

    # Robot self-collision follows the Play task the policy was pretrained on: off for the
    # Kuka, on (minus RobotSpec.self_collision_filter) for the Flexiv + Wuji.
    self_collision = bool(getattr(env.cfg, "robot_self_collision", False))
    robot_converted_usd = _convert_urdf_to_usd(
        _asset_path(assets_cfg.robot_urdf),
        usd_work_dir,
        fix_base=True,
        self_collision=self_collision,
        joint_drive=_robot_joint_drive_cfg(),
    )
    if self_collision:
        _apply_self_collision_filters(robot_converted_usd, env.robot_spec.self_collision_filter)
    robot_usd_path = _bake_usd(
        robot_converted_usd,
        bake_root,
        "robot",
        props=dict(
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
            enabled_self_collisions=self_collision,
            solver_position_iterations=8,
            solver_velocity_iterations=0,
        ),
        apply_physx_articulation=True,
    )
    table_usd_path = _bake_usd(
        _convert_urdf_to_usd(
            _asset_path(assets_cfg.table_urdf),
            usd_work_dir,
            fix_base=False,
        ),
        bake_root,
        "table",
        props=dict(
            kinematic_enabled=True,
            disable_gravity=True,
            articulation_enabled=False,
        ),
    )
    _log_scene_step(setup_t0, "converted object/hole/table URDFs")

    _materialize_env_prims(env)

    # Use UsdFileCfg (single USD path) instead of MultiUsdFileCfg.
    # precise_assembly is a single-geometry task (every env uses the same peg + hole +
    # table), so the multi-asset carb path is pure overhead — and it triggers
    # the noisy "Varying assets ... however replicate_physics is enabled"
    # warning under `replicate_physics=True`. The regex prim_path is kept so
    # Articulation / RigidObject discover all envs after clone.
    env.robot = Articulation(build_robot_articulation_usd_cfg(robot_usd_path, spec=env.robot_spec))
    env.table = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Table", table_usd_path))
    env.hole = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Hole", hole_usd_path))
    env.object = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_path))
    env.goal_viz = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_path))
    _log_scene_step(setup_t0, "spawned robot/table/hole/object/goalviz")

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    env._object_scale_per_env = torch.tensor(
        env._pih_object_scale,
        device=env.device,
        dtype=torch.float32,
    ).expand(env.num_envs, -1).contiguous()
    env._object_asset_index_per_env = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )
    # Half the table slab's height: 0.15 for the Kuka's table, 0.02 for the Flexiv's.
    env._table_half_height = _urdf_box_size(_asset_path(assets_cfg.table_urdf))[2] / 2

    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["hole"] = env.hole
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    hide_goal_viz_for_student_camera(env)
    _log_scene_step(setup_t0, "registered assets with scene")

    # When replicate_physics=True, InteractiveScene.__init__ leaves
    # `_default_env_origins=None` and expects `clone_environments()` to
    # populate it later. That auto-call only fires inside
    # `_add_entities_from_cfg` (config-driven scenes). Our scene is
    # manually built, so we call it ourselves here. With
    # replicate_physics=False this is a no-op for env_origins (already
    # set in __init__).
    if env.scene._default_env_origins is None:
        env.scene.clone_environments(copy_from_source=False)
        _log_scene_step(setup_t0, "cloned environments (replicate_physics=True path)")

    # Student camera is set up AFTER the clone so every env path exists at
    # sensor-construction time. The TiledCamera/Camera spawn-cfg regex still
    # creates per-env prims correctly post-clone, and the RayCaster branch
    # can pre-create its Xform parent on each env explicitly.
    setup_student_camera(env)


__all__ = ["setup_scene"]
