"""Scene construction, asset conversion, and runtime material setup."""

from __future__ import annotations

import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_angle_axis, quat_mul
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

from ..robots import KUKA_SHARPA, RobotSpec
from .generate_objects import generate_handle_head_urdfs
from .goal_table_clearance import collision_corners


# ----------------------------------------------------------------------------
# Joint names / regexes / body names / gains
# ----------------------------------------------------------------------------
# The robot lives in ``robots.py`` (one RobotSpec per embodiment, picked by ``cfg.robot``). These
# module names are the Kuka + Sharpa's, kept because other tasks and the Newton patches read them.

ARM_JOINT_REGEX = KUKA_SHARPA.arm_joint_regex
HAND_JOINT_REGEX = KUKA_SHARPA.hand_joint_regex
# Legacy policy order; Isaac Lab tensors are permuted at action/obs boundaries.
JOINT_NAMES_CANONICAL: tuple[str, ...] = KUKA_SHARPA.joint_names_canonical
PALM_BODY_NAME = KUKA_SHARPA.palm_body_name
FINGERTIP_BODY_REGEX = KUKA_SHARPA.fingertip_body_regex
FINGERTIP_LINK_NAMES: tuple[str, ...] = KUKA_SHARPA.fingertip_link_names
# Per-joint PD gains and dynamics (verified with pretrained checkpoint).
ARM_JOINT_STIFFNESS: dict[str, float] = KUKA_SHARPA.arm_stiffness
ARM_JOINT_DAMPING: dict[str, float] = KUKA_SHARPA.arm_damping
HAND_JOINT_STIFFNESS: dict[str, float] = KUKA_SHARPA.hand_stiffness
HAND_JOINT_DAMPING: dict[str, float] = KUKA_SHARPA.hand_damping
HAND_JOINT_ARMATURE: dict[str, float] = KUKA_SHARPA.hand_armature
HAND_JOINT_FRICTION: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.132, "left_thumb_CMC_AA": 0.132,
    "left_thumb_MCP_FE": 0.07456, "left_thumb_MCP_AA": 0.07456, "left_thumb_IP": 0.01276,
    "left_2_index_MCP_FE": 0.07456, "left_index_MCP_AA": 0.07456,
    "left_index_PIP": 0.01276, "left_index_DIP": 0.00378738,
    "left_3_middle_MCP_FE": 0.07456, "left_middle_MCP_AA": 0.07456,
    "left_middle_PIP": 0.01276, "left_middle_DIP": 0.00378738,
    "left_4_ring_MCP_FE": 0.07456, "left_ring_MCP_AA": 0.07456,
    "left_ring_PIP": 0.01276, "left_ring_DIP": 0.00378738,
    "left_5_pinky_CMC": 0.012, "left_pinky_MCP_FE": 0.07456,
    "left_pinky_MCP_AA": 0.07456, "left_pinky_PIP": 0.01276, "left_pinky_DIP": 0.00378738,
}

assert len(HAND_JOINT_FRICTION) == 22

# Proven-working default arm pose.
ARM_DEFAULT_JOINT_POS: dict[str, float] = KUKA_SHARPA.arm_default_joint_pos

_CONTACT_OFFSET = 0.002
_REST_OFFSET = 0.0

# group: "rb" (RigidBodyAPI) or "art" (ArticulationRootAPI).
# attr_name: USD attribute path. vtype_str: matched against pxr.Sdf.ValueTypeNames.
_PHYSICS_SPECS: dict[str, tuple[str, str, str]] = {
    "kinematic_enabled": ("rb", "physics:kinematicEnabled", "Bool"),
    "disable_gravity": ("rb", "physxRigidBody:disableGravity", "Bool"),
    "max_depenetration_velocity": ("rb", "physxRigidBody:maxDepenetrationVelocity", "Float"),
    "rb_solver_position_iterations": ("rb", "physxRigidBody:solverPositionIterationCount", "Int"),
    "rb_solver_velocity_iterations": ("rb", "physxRigidBody:solverVelocityIterationCount", "Int"),
    "articulation_enabled": ("art", "physics:articulationEnabled", "Bool"),
    "enabled_self_collisions": ("art", "physxArticulation:enabledSelfCollisions", "Bool"),
    "solver_position_iterations": ("art", "physxArticulation:solverPositionIterationCount", "Int"),
    "solver_velocity_iterations": ("art", "physxArticulation:solverVelocityIterationCount", "Int"),
}


def build_robot_articulation_usd_cfg(
    usd_path: str, spec: RobotSpec = KUKA_SHARPA
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.8, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                **spec.arm_default_joint_pos,
                **{name: 0.0 for name in spec.hand_joint_names},
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[spec.arm_joint_regex],
                stiffness=spec.arm_stiffness,
                damping=spec.arm_damping,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[spec.hand_joint_regex],
                stiffness=spec.hand_stiffness,
                damping=spec.hand_damping,
                armature=spec.hand_armature,
                effort_limit_sim=spec.hand_effort_limit,
                velocity_limit_sim=spec.hand_velocity_limit,
                friction=spec.hand_friction,
            ),
        },
    )


def build_rigid_object_cfg(prim_path: str, usd_paths: list[str]) -> RigidObjectCfg:
    """Spawn a RigidObject from one or more pre-baked USDs (round-robin)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=MultiUsdFileCfg(usd_path=list(usd_paths), random_choice=False),
    )


def _log_scene_step(start_time: float, message: str) -> None:
    print(f"[scene_utils][+{time.perf_counter() - start_time:.2f}s] {message}", flush=True)


def _student_camera_data_types(modality: str) -> list[str]:
    modality = str(modality).lower()
    if modality == "depth":
        return ["distance_to_image_plane"]
    if modality == "rgb":
        return ["rgb"]
    if modality == "rgbd":
        return ["rgb", "distance_to_image_plane"]
    raise ValueError(
        "cfg.student_obs.image_modality must be one of "
        f"('depth', 'rgb', 'rgbd'), got {modality!r}."
    )


def hide_goal_viz_for_student_camera(env) -> None:
    cfg = getattr(env.cfg, "student_obs", None)
    if cfg is None or not cfg.enabled or not cfg.hide_goal_viz:
        return

    from pxr import UsdGeom

    stage = get_current_stage()
    goal_viz_paths = find_matching_prim_paths("/World/envs/env_.*/GoalViz")
    for prim_path in goal_viz_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    _log_scene_step(
        time.perf_counter(),
        f"hid {len(goal_viz_paths)} GoalViz prims from render products",
    )


def _quat_wxyz_to_rotmat(quat_wxyz: tuple) -> torch.Tensor:
    """Standard (w, x, y, z) -> 3x3 rotation matrix.

    Column 0 of the result is the camera's local +X axis in world coords,
    which for a ROS-convention camera ('+X = image right') is the
    direction toward the RIGHT eye of a stereo pair.
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    return torch.tensor([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),         2 * (x * z + w * y)],
        [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),         2 * (y * z - w * x)],
        [    2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def _stereo_right_pose_from_left(
    left_pos: tuple, left_quat_wxyz: tuple, baseline_m: float
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Right-eye pose from left-eye pose + ZED-style horizontal baseline.

    Rectified stereo shares orientation, so the right cam's quat is the
    same as the left's; only the position is offset by `baseline_m` along
    the left camera's local +X axis (image-right under ROS convention).
    """
    R = _quat_wxyz_to_rotmat(left_quat_wxyz)
    delta = float(baseline_m) * R[:, 0]
    right_pos = tuple(float(left_pos[i] + delta[i].item()) for i in range(3))
    right_quat = tuple(float(v) for v in left_quat_wxyz)
    return right_pos, right_quat


def _diagnose_target_expr(expr: str) -> None:
    """Echo what `_obtain_trackable_prim_view` will see for this expr.

    Walks parents until hitting RigidBodyAPI / ArticulationRootAPI exactly
    like the raycaster does, then prints `mesh_prims` and `view_prims`
    counts. If they disagree, the worker will error with
    "1 mesh prim vs N physics prims" and this print pinpoints which target.
    """
    from pxr import Usd, UsdPhysics
    from isaaclab.sim.utils import find_matching_prims, find_first_matching_prim

    try:
        mesh_prim = find_first_matching_prim(expr)
        if mesh_prim is None or not mesh_prim.IsValid():
            print(f"[raycaster][diag]   {expr!r}: NO mesh prim match", flush=True)
            return
        cur_prim = mesh_prim
        cur_expr = expr
        depth = 0
        while True:
            depth += 1
            if cur_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                kind = "ArticulationRoot"
                break
            if cur_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                kind = "RigidBody"
                break
            parent = cur_prim.GetParent()
            cur_expr = cur_expr.rsplit("/", 1)[0]
            if not parent.IsValid() or depth > 10:
                kind = "XForm(fallback)"
                break
            cur_prim = parent

        mesh_paths = [str(p.GetPath()) for p in find_matching_prims(expr)]
        view_paths = [str(p.GetPath()) for p in find_matching_prims(cur_expr)]
        print(
            f"[raycaster][diag]   target={expr!r}\n"
            f"     -> walked up to {cur_prim.GetPath()} ({kind}), "
            f"path_expr={cur_expr!r}\n"
            f"     -> meshes ({len(mesh_paths)}): {mesh_paths}\n"
            f"     -> views  ({len(view_paths)}): {view_paths}",
            flush=True,
        )
    except Exception as exc:
        print(f"[raycaster][diag]   {expr!r}: diag failed: {exc!r}", flush=True)


def _expand_link_wildcard(expr: str, num_envs: int) -> list[str]:
    """Resolve a `.../<wildcard>/visuals` raycast prim expression at setup
    time.

    The raycaster needs a 1-to-1 mapping between mesh prims (the leaves
    matching the prim_expr) and physics-body view prims (the parent it
    walks up to with RigidBodyAPI / ArticulationRootAPI). When the regex
    component just before `/visuals` is a wildcard like `.*`, the walk-up
    builds a `.../...*` view-prim expression that also matches sibling
    non-link xforms (`Looks`, `joints`, ...), and the raycaster errors:
        "The number of mesh prims (N) does not match the number of physics
         prims (M)"

    Fix: at setup time, list the actual link children that (a) match the
    wildcard component, AND (b) have a `/visuals` subgroup with at least
    one Gprim child. Return one explicit prim_expr per concrete link name,
    preserving the `/visuals` tail. Callers who pass an expression without
    a single wildcard between two slashes just get it back unchanged.

    Example:
        `/World/envs/env_.*/Object/.*/visuals`  ->
            ["/World/envs/env_.*/Object/peg/visuals"]      # peg.tol0p5mm
        or  ["/World/envs/env_.*/Object/lpeg/visuals"]     # Lpeg.tol0p5mm

    For multi-link object URDFs (e.g. fabrica beam parts) we emit one
    expression per link.
    """
    if "/.*/" not in expr:
        return [expr]
    # Split at the first wildcard component so we can list its concrete
    # candidates against env_0 (clones share structure, so env_0 is enough).
    head, tail = expr.split("/.*/", 1)
    # Materialise `env_.*` in the head against env_0 for the stage lookup.
    head_env0 = head.replace("/env_.*", "/env_0")
    stage = get_current_stage()
    parent_prim = stage.GetPrimAtPath(head_env0)
    if not parent_prim.IsValid():
        print(
            f"[raycaster][expand] parent {head_env0!r} INVALID; keeping {expr!r}",
            flush=True,
        )
        return [expr]

    # Walk children of `parent_prim`. Keep ones that look like URDF link
    # subgroups (have a `/visuals` child), drop the materials / joints /
    # sensor xforms the URDF importer also creates.
    SKIP_NAMES = {"Looks", "joints", "Sensors", "joint_drives"}
    concrete: list[str] = []
    skipped: list[tuple[str, str]] = []
    for child in parent_prim.GetChildren():
        name = child.GetName()
        if name in SKIP_NAMES:
            skipped.append((name, "in SKIP_NAMES"))
            continue
        visuals_prim = stage.GetPrimAtPath(f"{head_env0}/{name}/visuals")
        if not visuals_prim.IsValid():
            skipped.append((name, "no /visuals child"))
            continue
        concrete.append(f"{head}/{name}/{tail}")
    print(
        f"[raycaster][expand] {expr!r} -> kept {len(concrete)}: "
        f"{[c.rsplit('/', 2)[0].rsplit('/', 1)[1] for c in concrete]}; "
        f"skipped {skipped}",
        flush=True,
    )
    return concrete if concrete else [expr]


def setup_student_camera(env) -> None:
    """Create the optional per-env student camera sensor.

    The camera is registered with ``env.scene.sensors`` so DirectRLEnv owns its
    lifecycle. Student observations remain opt-in through
    ``env.unwrapped.get_student_obs()``; the normal teacher/critic observation
    path does not touch this sensor.
    """
    cfg = getattr(env.cfg, "student_obs", None)
    env.student_camera = None
    if cfg is None or not cfg.enabled or not cfg.image_enabled:
        return

    backend = str(cfg.camera_backend).lower()
    if backend not in ("tiled", "standard", "raycaster", "foundation_stereo"):
        raise ValueError(
            "cfg.student_obs.camera_backend must be 'tiled', 'standard', "
            f"'raycaster', or 'foundation_stereo', got {backend!r}."
        )

    camera_mount = str(cfg.camera_mount).lower()
    if camera_mount != "world":
        raise NotImplementedError(
            "Only world-mounted student cameras are wired into DirectRLEnv right "
            f"now; got cfg.student_obs.camera_mount={camera_mount!r}."
        )

    t0 = time.perf_counter()

    if backend in ("tiled", "standard"):
        from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg

        camera_cfg_cls = TiledCameraCfg if backend == "tiled" else CameraCfg
        camera_cls = TiledCamera if backend == "tiled" else Camera
        camera_cfg = camera_cfg_cls(
            prim_path="/World/envs/env_.*/StudentCamera",
            update_period=float(getattr(cfg, "camera_update_period_s", 0.0)),
            update_latest_camera_pose=True,
            height=int(cfg.image_height),
            width=int(cfg.image_width),
            data_types=_student_camera_data_types(cfg.image_modality),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(cfg.focal_length),
                focus_distance=float(cfg.focus_distance),
                horizontal_aperture=float(cfg.horizontal_aperture),
                horizontal_aperture_offset=float(
                    getattr(cfg, "horizontal_aperture_offset", 0.0)
                ),
                vertical_aperture_offset=float(
                    getattr(cfg, "vertical_aperture_offset", 0.0)
                ),
                clipping_range=tuple(float(x) for x in cfg.clipping_range),
            ),
            offset=camera_cfg_cls.OffsetCfg(
                pos=tuple(float(x) for x in cfg.camera_pos),
                rot=tuple(float(x) for x in cfg.camera_quat_wxyz),
                convention=str(cfg.camera_convention),
            ),
        )
        env.student_camera = camera_cls(cfg=camera_cfg)
        env.scene.sensors["student_camera"] = env.student_camera
        size_str = f"{int(cfg.image_width)}x{int(cfg.image_height)}"
    elif backend == "raycaster":
        # CUDA mesh raycaster, no rasterizer / Replicator.
        # Restricted to depth modality — RGB / semantic outputs aren't supported.
        if str(cfg.image_modality).lower() != "depth":
            raise ValueError(
                "cfg.student_obs.camera_backend='raycaster' only supports "
                f"image_modality='depth' (got {cfg.image_modality!r}). "
                "Use camera_backend='tiled' for RGB/semantic modalities."
            )
        from isaaclab.sensors.ray_caster import (
            MultiMeshRayCasterCamera,
            MultiMeshRayCasterCameraCfg,
            patterns,
        )

        pattern_cfg = patterns.PinholeCameraPatternCfg(
            focal_length=float(cfg.focal_length),
            horizontal_aperture=float(cfg.horizontal_aperture),
            horizontal_aperture_offset=float(
                getattr(cfg, "horizontal_aperture_offset", 0.0)
            ),
            vertical_aperture_offset=float(
                getattr(cfg, "vertical_aperture_offset", 0.0)
            ),
            width=int(cfg.image_width),
            height=int(cfg.image_height),
        )

        raycast_targets = []
        for expr in tuple(cfg.raycast_static_prim_exprs):
            print(f"[raycaster] STATIC target: {expr!r}", flush=True)
            raycast_targets.append(
                MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                    prim_expr=str(expr), track_mesh_transforms=False
                )
            )
        for expr in tuple(cfg.raycast_dynamic_prim_exprs):
            expanded = _expand_link_wildcard(str(expr), env.num_envs)
            print(
                f"[raycaster] DYNAMIC target: {expr!r} -> "
                f"{len(expanded)} concrete: {expanded}",
                flush=True,
            )
            for concrete_expr in expanded:
                # Verify mesh/view counts match before the raycaster's
                # _obtain_trackable_prim_view sees this expression, so we
                # know exactly which target trips the
                # "1 mesh vs N physics" error.
                _diagnose_target_expr(concrete_expr)
                raycast_targets.append(
                    MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                        prim_expr=concrete_expr, track_mesh_transforms=True
                    )
                )
        if not raycast_targets:
            raise ValueError(
                "camera_backend='raycaster' requires at least one entry in "
                "cfg.student_obs.raycast_static_prim_exprs or "
                "raycast_dynamic_prim_exprs."
            )

        # MultiMeshRayCasterCamera attaches to an existing prim (it doesn't
        # spawn one the way TiledCamera does). setup_student_camera now runs
        # AFTER clone_environments(), so every env's namespace exists — we
        # just create a per-env Xform parent explicitly.
        for env_id in range(int(env.num_envs)):
            sim_utils.create_prim(
                f"/World/envs/env_{env_id}/StudentCamera", "Xform"
            )

        raycaster_cfg = MultiMeshRayCasterCameraCfg(
            prim_path="/World/envs/env_.*/StudentCamera",
            update_period=float(getattr(cfg, "camera_update_period_s", 0.0)),
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=tuple(float(x) for x in cfg.camera_pos),
                rot=tuple(float(x) for x in cfg.camera_quat_wxyz),
                convention=str(cfg.camera_convention),
            ),
            mesh_prim_paths=raycast_targets,
            pattern_cfg=pattern_cfg,
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior=str(cfg.raycast_depth_clipping_behavior),
            max_distance=float(cfg.raycast_max_distance_m),
        )
        env.student_camera = MultiMeshRayCasterCamera(cfg=raycaster_cfg)
        env.scene.sensors["student_camera"] = env.student_camera
        size_str = (
            f"{int(cfg.image_width)}x{int(cfg.image_height)} "
            f"targets={len(raycast_targets)} "
            f"(static={len(tuple(cfg.raycast_static_prim_exprs))}, "
            f"dynamic={len(tuple(cfg.raycast_dynamic_prim_exprs))})"
        )

    elif backend == "foundation_stereo":
        # Stereo TiledCamera pair (RGB), rendered at fs_stereo_{width,height}.
        # Fast-FS runs inference at the same resolution to produce disparity ->
        # depth, which the obs pipeline downsamples to (image_width, image_height)
        # before the usual noise / crop / window-normalize chain.
        from isaaclab.sensors import TiledCamera, TiledCameraCfg

        # Stereo render must be at multiples of 32 (FS InputPadder requirement).
        stereo_w = int(cfg.fs_stereo_width)
        stereo_h = int(cfg.fs_stereo_height)
        if stereo_w % 32 != 0 or stereo_h % 32 != 0:
            raise ValueError(
                f"camera_backend='foundation_stereo' requires "
                f"fs_stereo_width / fs_stereo_height to be multiples of 32 "
                f"(got {stereo_w}x{stereo_h})."
            )

        left_pos = tuple(float(x) for x in cfg.camera_pos)
        left_quat = tuple(float(x) for x in cfg.camera_quat_wxyz)
        right_pos, right_quat = _stereo_right_pose_from_left(
            left_pos, left_quat, float(cfg.fs_stereo_baseline_m)
        )

        def _build_stereo_cam_cfg(prim_path: str, pos: tuple, quat: tuple) -> "TiledCameraCfg":
            return TiledCameraCfg(
                prim_path=prim_path,
                update_period=float(getattr(cfg, "camera_update_period_s", 0.0)),
                update_latest_camera_pose=True,
                height=stereo_h,
                width=stereo_w,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(cfg.focal_length),
                    focus_distance=float(cfg.focus_distance),
                    horizontal_aperture=float(cfg.horizontal_aperture),
                    horizontal_aperture_offset=float(
                        getattr(cfg, "horizontal_aperture_offset", 0.0)
                    ),
                    vertical_aperture_offset=float(
                        getattr(cfg, "vertical_aperture_offset", 0.0)
                    ),
                    clipping_range=tuple(float(x) for x in cfg.clipping_range),
                ),
                offset=TiledCameraCfg.OffsetCfg(
                    pos=pos, rot=quat, convention=str(cfg.camera_convention),
                ),
            )

        env.student_camera_left = TiledCamera(
            cfg=_build_stereo_cam_cfg(
                "/World/envs/env_.*/StudentCameraLeft", left_pos, left_quat,
            )
        )
        env.student_camera_right = TiledCamera(
            cfg=_build_stereo_cam_cfg(
                "/World/envs/env_.*/StudentCameraRight", right_pos, right_quat,
            )
        )
        env.scene.sensors["student_camera_left"] = env.student_camera_left
        env.scene.sensors["student_camera_right"] = env.student_camera_right
        # Alias so any read paths that hit `env.student_camera` still resolve.
        env.student_camera = env.student_camera_left
        # The FS inference module is loaded lazily on first call to avoid
        # paying the model-load cost when the env is constructed for tasks
        # that don't actually consume the student image.
        env._fs_module = None
        print(
            f"[foundation_stereo] stereo pair: left  pos={left_pos} quat={left_quat}\n"
            f"[foundation_stereo]              right pos={right_pos} (baseline {cfg.fs_stereo_baseline_m:.3f} m along left +X)\n"
            f"[foundation_stereo]              capture {stereo_w}x{stereo_h}, model_dir={cfg.fs_model_dir}, "
            f"iters={cfg.fs_valid_iters}, max_disp={cfg.fs_max_disp}, "
            f"engine_dir={cfg.fs_engine_dir or '(none, using PyTorch)'}",
            flush=True,
        )
        size_str = (
            f"stereo {stereo_w}x{stereo_h} -> "
            f"{int(cfg.image_width)}x{int(cfg.image_height)} via Fast-FS"
        )

    _log_scene_step(
        t0,
        f"registered student camera backend={backend} "
        f"modality={cfg.image_modality} "
        f"size={size_str}",
    )


def _apply_depth_noise(env, depth: torch.Tensor) -> torch.Tensor:
    """5-stage depth noise pipeline on raw-meters depth, shape (B, 1, H, W).

    Stages (all gated by their own σ/prob being > 0):
      1. additive Gaussian
      2. spatially-correlated Gaussian (k×k mean-blur of i.i.d. noise)
      3. per-pixel dropout to 0
      4. per-pixel random-uniform replacement in [randu_min, randu_max]
      5. stick artifacts (small random streaks)

    No-op when `cfg.use_depth_aug=False`. Defaults match the team's "medium"
    preset (see StudentObsCfg).
    """
    cfg = env.cfg.student_obs
    if not bool(getattr(cfg, "use_depth_aug", False)):
        return depth

    out = depth.float()
    device = out.device

    # 1. additive Gaussian
    gauss_std = float(cfg.depth_aug_gaussian_std_m)
    if gauss_std > 0.0:
        out = out + torch.randn_like(out) * gauss_std

    # 2. spatially-correlated Gaussian: i.i.d. noise blurred by mean k×k kernel.
    corr_std = float(cfg.depth_aug_correlated_std_m)
    k = int(cfg.depth_aug_correlated_kernel_size)
    if corr_std > 0.0 and k > 1:
        noise = torch.randn_like(out) * corr_std
        kernel = torch.ones(1, 1, k, k, device=device, dtype=out.dtype) / (k * k)
        out = out + F.conv2d(noise, kernel, padding=k // 2)

    # 3. per-pixel dropout to 0
    p_drop = float(cfg.depth_aug_dropout_prob)
    if p_drop > 0.0:
        keep = (torch.rand_like(out) >= p_drop).to(out.dtype)
        out = out * keep

    # 4. per-pixel random-uniform replacement
    p_randu = float(cfg.depth_aug_randu_prob)
    if p_randu > 0.0:
        lo = float(cfg.depth_aug_randu_min_m)
        hi = float(cfg.depth_aug_randu_max_m)
        mask = torch.rand_like(out) < p_randu
        randu = torch.rand_like(out) * (hi - lo) + lo
        out = torch.where(mask, randu, out)

    # 5. stick artifacts (Poisson-count per image, vectorized line rasterization).
    stick_prob = float(cfg.depth_aug_stick_prob)
    max_sticks = int(cfg.depth_aug_max_sticks_per_image)
    if stick_prob > 0.0 and max_sticks > 0:
        out = _draw_depth_sticks(out, cfg=cfg, stick_prob=stick_prob, max_sticks=max_sticks)

    return out


def _draw_depth_sticks(
    depth_b1hw: torch.Tensor,
    *,
    cfg,
    stick_prob: float,
    max_sticks: int,
) -> torch.Tensor:
    """Add random line streaks to a (B, 1, H, W) depth buffer.

    Fully vectorized: zero Python loops, zero per-step ``.item()``/``.tolist()``
    syncs. Per env we sample up to ``max_sticks`` candidate sticks, rasterize
    each as a length-``max_len_px`` line, expand by the stick's width, mask
    in-bounds + active steps, and scatter fill values in a single index
    write. Profile shows ~250 ms/step at B=512 with the previous Python loop;
    this implementation runs in <1 ms.

    Per-stick parameter distributions match the previous Python loop:
    length ~ U[1, max_len], width ~ U[1, max_w], angle ~ U[0, 2π),
    fill ~ U[randu_min, randu_max], origin ~ U(pixel grid). The per-image
    stick count is sampled per-candidate as Bernoulli(p) where
    ``p = expected_per_image / max_sticks``, giving the same expected count
    as the original ``Poisson(expected).clamp(max=max_sticks)`` (variance
    differs by O(1), which is irrelevant for this DR noise channel).
    """
    B, _, H, W = depth_b1hw.shape
    device = depth_b1hw.device
    dtype = depth_b1hw.dtype

    if max_sticks <= 0 or stick_prob <= 0.0:
        return depth_b1hw

    max_len = max(1, int(cfg.depth_aug_stick_max_len_px))
    max_w = max(1, int(cfg.depth_aug_stick_max_width_px))
    w_half_max = max_w // 2
    K = 2 * w_half_max + 1  # kernel side, matches the old `out[..., y±w_half, x±w_half]` slice
    lo = float(cfg.depth_aug_randu_min_m)
    hi = float(cfg.depth_aug_randu_max_m)

    # Per-candidate gate: expected_per_image = max_sticks * p_candidate.
    expected_per_image = stick_prob * float(H * W)
    p_candidate = min(1.0, expected_per_image / float(max_sticks))
    active = torch.rand(B, max_sticks, device=device) < p_candidate          # (B, S)

    # Per-stick parameters. All shapes (B, S).
    x0 = torch.randint(0, W, (B, max_sticks), device=device)
    y0 = torch.randint(0, H, (B, max_sticks), device=device)
    theta = torch.rand(B, max_sticks, device=device) * (2.0 * torch.pi)
    lengths = torch.randint(1, max_len + 1, (B, max_sticks), device=device)
    widths = torch.randint(1, max_w + 1, (B, max_sticks), device=device)
    fills = torch.rand(B, max_sticks, device=device) * (hi - lo) + lo
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Rasterize the line center across max_len pixels.
    s = torch.arange(max_len, device=device, dtype=torch.float32)            # (L,)
    xs = x0.float()[:, :, None] + cos_t[:, :, None] * s[None, None, :]       # (B, S, L)
    ys = y0.float()[:, :, None] + sin_t[:, :, None] * s[None, None, :]       # (B, S, L)
    s_active = s[None, None, :] < lengths.float()[:, :, None]                # (B, S, L)
    line_active = active[:, :, None] & s_active                              # (B, S, L)

    # Expand by stick width.
    offs = torch.arange(-w_half_max, w_half_max + 1, device=device)          # (K,)
    dy_g, dx_g = torch.meshgrid(offs, offs, indexing="ij")                   # (K, K) each
    w_half_b = (widths // 2)[:, :, None, None, None]                         # (B, S, 1, 1, 1)
    within_w = (dx_g[None, None, None] .abs() <= w_half_b) & \
               (dy_g[None, None, None].abs() <= w_half_b)                    # (B, S, 1, K, K)

    xi = xs[:, :, :, None, None].long() + dx_g[None, None, None]             # (B, S, L, K, K)
    yi = ys[:, :, :, None, None].long() + dy_g[None, None, None]
    in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    final_mask = line_active[:, :, :, None, None] & within_w & in_bounds     # (B, S, L, K, K)

    if not bool(final_mask.any()):
        return depth_b1hw

    # Flatten and scatter. One kernel launch.
    b_idx = torch.arange(B, device=device)[:, None, None, None, None].expand_as(xi)
    fill_full = fills[:, :, None, None, None].expand_as(xi).to(dtype)
    flat_mask = final_mask.reshape(-1)
    flat_b = b_idx.reshape(-1)[flat_mask]
    flat_y = yi.reshape(-1)[flat_mask]
    flat_x = xi.reshape(-1)[flat_mask]
    flat_fill = fill_full.reshape(-1)[flat_mask]

    out = depth_b1hw.clone()
    out[flat_b, 0, flat_y, flat_x] = flat_fill
    return out


def _apply_camera_pose_rand_at_reset(env, env_ids: torch.Tensor) -> None:
    """Sample per-env camera-pose noise and apply via student_camera.set_world_poses.

    Per-env at reset cadence: a fresh random offset is drawn each time
    `env_ids` reset and the camera is moved on the spot. No persistent
    per-env state; noise is regenerated every reset. No-op when
    `cfg.use_camera_pose_rand=False`.
    """
    cfg = getattr(env.cfg, "student_obs", None)
    camera = getattr(env, "student_camera", None)
    if cfg is None or camera is None:
        return
    if not bool(getattr(cfg, "use_camera_pose_rand", False)):
        return

    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = int(env_ids.numel())
    if n == 0:
        return

    device = env.device
    pos_range = torch.as_tensor(cfg.camera_pos_noise_m, device=device, dtype=torch.float32)
    rot_range_deg = torch.as_tensor(cfg.camera_rot_noise_deg, device=device, dtype=torch.float32)
    base_pos = torch.as_tensor(cfg.camera_pos, device=device, dtype=torch.float32)
    base_quat = torch.as_tensor(cfg.camera_quat_wxyz, device=device, dtype=torch.float32)

    pos_noise = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * pos_range
    rot_noise_rad = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * rot_range_deg * (torch.pi / 180.0)

    # RPY → wxyz quat: q = q_yaw * q_pitch * q_roll
    axes = torch.eye(3, device=device, dtype=torch.float32)
    q_roll = quat_from_angle_axis(rot_noise_rad[:, 0], axes[0].expand(n, -1))
    q_pitch = quat_from_angle_axis(rot_noise_rad[:, 1], axes[1].expand(n, -1))
    q_yaw = quat_from_angle_axis(rot_noise_rad[:, 2], axes[2].expand(n, -1))
    rot_noise_quat = quat_mul(q_yaw, quat_mul(q_pitch, q_roll))

    pos_w = env.scene.env_origins[env_ids] + base_pos + pos_noise
    quat = quat_mul(rot_noise_quat, base_quat.expand(n, -1))

    # Isaac Lab 5.1+: when Fabric is enabled, write through to USD so the
    # renderer actually picks up the new camera pose.
    view = getattr(camera, "_view", None)
    if view is not None and hasattr(view, "_sync_usd_on_fabric_write"):
        view._sync_usd_on_fabric_write = True

    camera.set_world_poses(
        positions=pos_w,
        orientations=quat,
        env_ids=env_ids,
        convention=str(cfg.camera_convention),
    )


def _preprocess_student_depth(env, depth: torch.Tensor) -> torch.Tensor:
    cfg = env.cfg.student_obs
    depth = depth.float()
    near = float(cfg.depth_min_m)
    far = float(cfg.depth_max_m)
    if far <= near:
        raise ValueError(
            "cfg.student_obs.depth_max_m must be greater than "
            "cfg.student_obs.depth_min_m."
        )

    mode = str(cfg.depth_preprocess_mode).lower()
    valid = torch.isfinite(depth) & (depth >= near) & (depth <= far)
    if mode == "clip_divide":
        clipped = torch.clamp(
            torch.nan_to_num(depth, nan=far, posinf=far, neginf=near),
            near,
            far,
        )
        return clipped / far
    if mode == "metric":
        return torch.where(valid, depth, torch.zeros_like(depth))
    if mode == "window_normalize":
        safe_depth = torch.nan_to_num(depth, nan=far, posinf=far, neginf=near)
        normalized = (safe_depth - near) / (far - near)
        return torch.clamp(normalized, 0.0, 1.0)
    raise ValueError(
        "cfg.student_obs.depth_preprocess_mode must be one of "
        f"('clip_divide', 'window_normalize', 'metric'), got {mode!r}."
    )


def _validate_student_image_shape(env, image: torch.Tensor) -> torch.Tensor:
    cfg = env.cfg.student_obs
    size = (int(cfg.image_input_height), int(cfg.image_input_width))
    if image.shape[-2:] == size:
        return image
    raise RuntimeError(
        "Student image shape does not match configured input shape. "
        "Adjust crop/image_input settings; this path does not resize images. "
        f"got HxW={tuple(image.shape[-2:])}, expected HxW={size}."
    )


def _crop_student_image(env, image: torch.Tensor) -> torch.Tensor:
    cfg = env.cfg.student_obs
    if not cfg.crop_enabled:
        return image

    height, width = image.shape[-2:]
    x0, y0 = (int(v) for v in cfg.crop_top_left)
    x1, y1 = (int(v) for v in cfg.crop_bottom_right)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            "Invalid student image crop coordinates: "
            f"top_left=({x0}, {y0}), bottom_right=({x1}, {y1}), "
            f"image={width}x{height}. Coordinates use x/y pixels with "
            "bottom_right exclusive."
        )
    return image[..., y0:y1, x0:x1]


def _run_foundation_stereo(env, cfg) -> torch.Tensor:
    """Render stereo RGB and run Fast-FS to produce depth in meters.

    Lazy-loads the FS module on first call. Reads from
    ``env.student_camera_{left,right}.data.output["rgb"]`` (shape
    ``(B, H, W, 4)`` uint8 from Replicator), converts to ``(B, 3, H, W)``
    float in [0, 255] for FS, runs inference, converts disparity to depth
    via ``depth = fx_px * baseline / disparity``, and downsamples to
    ``(image_height, image_width)`` if ``fs_downsample_to_policy_res`` is
    set (the policy's depth-window crop pipeline runs on the result).
    """
    if getattr(env, "_fs_module", None) is None:
        raise NotImplementedError(
            "The Foundation-Stereo depth backend is not included in this "
            "state-based release of Play2Perfect."
        )
        engine_dir = str(cfg.fs_engine_dir).strip() or None
        env._fs_module = FastFoundationStereoModule(
            model_dir=str(cfg.fs_model_dir),
            engine_dir=engine_dir,
            valid_iters=int(cfg.fs_valid_iters),
            max_disp=int(cfg.fs_max_disp),
            device=str(env.device),
        )
        print(
            f"[foundation_stereo] FS module loaded ({env._fs_module.backend})",
            flush=True,
        )

    left_rgba = env.student_camera_left.data.output["rgb"]
    right_rgba = env.student_camera_right.data.output["rgb"]
    if left_rgba is None or right_rgba is None:
        raise RuntimeError(
            "foundation_stereo backend: one of student_camera_{left,right} "
            "produced no RGB output."
        )
    # Replicator returns (B, H, W, 4) uint8 -> (B, 3, H, W) float in [0, 255].
    left  = left_rgba[..., :3].permute(0, 3, 1, 2).float().contiguous()
    right = right_rgba[..., :3].permute(0, 3, 1, 2).float().contiguous()

    # fx in pixels at the FS-render resolution.
    fx_px = float(cfg.fs_stereo_width) * float(cfg.focal_length) \
            / float(cfg.horizontal_aperture)
    depth = env._fs_module(
        left, right,
        fx_px=fx_px,
        baseline_m=float(cfg.fs_stereo_baseline_m),
    )                                              # (B, 1, H_stereo, W_stereo) in m

    if bool(getattr(cfg, "fs_downsample_to_policy_res", True)):
        depth = F.interpolate(
            depth,
            size=(int(cfg.image_height), int(cfg.image_width)),
            mode="bilinear",
            antialias=True,
        )
    return depth


def read_student_camera_image(env) -> torch.Tensor:
    """Return the configured student image as ``(num_envs, channels, H, W)``."""
    cfg = getattr(env.cfg, "student_obs", None)
    if cfg is None or not cfg.enabled:
        raise RuntimeError(
            "cfg.student_obs.enabled is false; no student image is available."
        )
    if not cfg.image_enabled:
        raise RuntimeError(
            "cfg.student_obs.image_enabled is false; no student image is available."
        )

    # ---- env-level frame-skip gate (true 30Hz / 15Hz / ... behavior) ----
    # Isaac Lab's sensor `update_period` is ineffective at our 60Hz policy
    # cadence because `_timestamp` accumulates across the decimation loop's
    # multiple `scene.update(dt=physics_dt)` calls plus our own explicit
    # `camera.update`. So we gate at the env level instead: only fall through
    # to a fresh render every `skip_every`-th call, otherwise return the
    # cached previous frame to mimic a slower physical camera.
    period_s = float(getattr(cfg, "camera_update_period_s", 0.0))
    if period_s > 0.0:
        step_dt = float(getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation))
        skip_every = max(1, int(round(period_s / step_dt)))
    else:
        skip_every = 1
    counter = int(getattr(env, "_student_camera_skip_counter", -1)) + 1
    env._student_camera_skip_counter = counter
    if skip_every > 1 and (counter % skip_every) != 0:
        cached = getattr(env, "_last_student_image_noisy", None)
        if cached is not None:
            return _validate_student_image_shape(env, cached)

    # --- Fast-FoundationStereo path -------------------------------------------
    # Stereo backend bypasses the regular single-camera retrieve: it renders
    # both stereo views, runs FS inference, downsamples the depth, then
    # re-uses the depth modality's noise / preprocess / crop chain.
    if str(cfg.camera_backend).lower() == "foundation_stereo":
        if str(cfg.image_modality).lower() != "depth":
            raise ValueError(
                f"camera_backend='foundation_stereo' requires "
                f"image_modality='depth' (got {cfg.image_modality!r})."
            )
        env.sim.render()
        dt = float(getattr(env, "physics_dt", env.cfg.sim.dt))
        # force_recompute=False so cfg.update_period is honored. With
        # update_period=0 the sensor refreshes every call (60Hz default);
        # with update_period=1/30 it caches alternate calls (30Hz).
        env.student_camera_left.update(dt, force_recompute=False)
        env.student_camera_right.update(dt, force_recompute=False)
        depth = _run_foundation_stereo(env, cfg)
        depth_raw = depth
        depth = _apply_depth_noise(env, depth)
        depth_policy = _crop_student_image(
            env, _preprocess_student_depth(env, depth)
        )
        env._last_student_image_noisy = depth_policy.detach()
        if bool(getattr(env.cfg.student_obs, "use_depth_aug", False)):
            depth_clean = _crop_student_image(
                env, _preprocess_student_depth(env, depth_raw)
            )
            env._last_student_image_clean = depth_clean.detach()
        else:
            env._last_student_image_clean = depth_policy.detach()
        return _validate_student_image_shape(env, depth_policy)

    camera = getattr(env, "student_camera", None)
    if camera is None:
        raise RuntimeError(
            "Student camera was not created. Check cfg.student_obs.enabled and "
            "launch with cameras enabled."
        )

    env.sim.render()
    dt = float(getattr(env, "physics_dt", env.cfg.sim.dt))
    # force_recompute=False so cfg.update_period is honored (see stereo branch
    # above). With update_period=0 the sensor refreshes every call; with
    # update_period=1/30 it caches alternate calls for a true 30Hz camera.
    camera.update(dt, force_recompute=False)

    outputs = camera.data.output
    available = {key: value for key, value in outputs.items() if value is not None}
    if not available:
        raise RuntimeError("Student camera produced no outputs.")

    modality = str(cfg.image_modality).lower()
    image_parts: list[torch.Tensor] = []
    if modality in ("rgb", "rgbd"):
        rgb = available.get("rgb")
        if rgb is None:
            raise RuntimeError(
                f"RGB output missing. Available student camera outputs: {list(available.keys())}"
            )
        rgb = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
        image_parts.append(_crop_student_image(env, rgb))

    if modality in ("depth", "rgbd"):
        depth = available.get("distance_to_image_plane")
        if depth is None:
            raise RuntimeError(
                f"Depth output missing. Available student camera outputs: {list(available.keys())}"
            )
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)
        else:
            raise RuntimeError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")
        depth_raw = depth
        depth = _apply_depth_noise(env, depth)
        depth_noisy_m = depth
        # The policy view: preprocess + crop on the (maybe-noisy) depth.
        depth_policy = _crop_student_image(env, _preprocess_student_depth(env, depth_noisy_m))
        image_parts.append(depth_policy)
        # Stash the cropped, normalized policy view + the matching clean view
        # for the interactive viewer's clean/noisy A/B. When noise is off the
        # two are identical (cheap, and avoids branching in the viewer).
        env._last_student_image_noisy = depth_policy.detach()
        if bool(getattr(env.cfg.student_obs, "use_depth_aug", False)):
            depth_clean = _crop_student_image(env, _preprocess_student_depth(env, depth_raw))
            env._last_student_image_clean = depth_clean.detach()
        else:
            env._last_student_image_clean = depth_policy.detach()

    if not image_parts:
        raise ValueError(f"Unsupported student image modality: {modality!r}")
    return _validate_student_image_shape(env, torch.cat(image_parts, dim=1))


def _set_usd_attr(prim, name: str, value, value_type) -> None:
    # The URDF converter occasionally emits attributes with malformed type
    # names; in that case remove and recreate so the typed Set lands.
    attr = prim.GetAttribute(name)
    if attr and (not attr.GetTypeName() or not str(attr.GetTypeName())):
        prim.RemoveProperty(name)
        attr = None
    (attr or prim.CreateAttribute(name, value_type, False)).Set(value)


@dataclass(frozen=True)
class _UrdfSdfCollisionMarker:
    mesh_stem: str
    mesh_filename: str
    resolution: int | None = None
    margin: float | None = None
    narrow_band_thickness: float | None = None
    subgrid_resolution: int | None = None


def _usd_safe_identifier(name: str) -> str:
    """Mirror the conservative subset of USD identifier rules we need here."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"mesh_{safe}"
    return safe


def _resolve_urdf_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()


def _prepare_urdf_for_isaacsim(asset_path: str, usd_work_dir: Path) -> str:
    """Return a URDF path whose mesh stems are valid USD prim identifiers.

    Isaac's URDF importer names USD prims from mesh stems.  Meshes such as
    ``6_hole_patch.obj`` therefore fail conversion because USD identifiers
    cannot start with a digit.  When needed, write a temporary URDF with safe
    mesh aliases while leaving the source asset untouched.
    """
    urdf_path = Path(asset_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    changed = False
    alias_dir = usd_work_dir / "_mesh_aliases" / urdf_path.stem
    alias_by_source: dict[Path, Path] = {}
    used_alias_names: set[str] = set()

    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if not filename:
            continue
        source_mesh = _resolve_urdf_mesh_path(urdf_path, filename)
        original_stem = source_mesh.stem
        safe_stem = _usd_safe_identifier(original_stem)
        if safe_stem == original_stem:
            continue

        changed = True
        alias = alias_by_source.get(source_mesh)
        if alias is None:
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias_name = f"{safe_stem}{source_mesh.suffix}"
            if alias_name in used_alias_names:
                index = 1
                while f"{safe_stem}_{index}{source_mesh.suffix}" in used_alias_names:
                    index += 1
                alias_name = f"{safe_stem}_{index}{source_mesh.suffix}"
            used_alias_names.add(alias_name)
            alias = alias_dir / alias_name
            shutil.copy2(source_mesh, alias)
            alias_by_source[source_mesh] = alias

        mesh_tag.set("filename", str(alias))

    if not changed:
        return asset_path

    # The copied URDF lives in the converter work dir, so make every remaining
    # mesh path absolute to preserve source-relative references.
    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if filename:
            mesh_tag.set("filename", str(_resolve_urdf_mesh_path(urdf_path, filename)))

    out_dir = usd_work_dir / "_urdf_preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / urdf_path.name
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return str(out_path)


def _parse_optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _parse_urdf_sdf_collision_markers(asset_path: str) -> list[_UrdfSdfCollisionMarker]:
    urdf_path = Path(asset_path)
    root = ET.parse(urdf_path).getroot()
    markers: list[_UrdfSdfCollisionMarker] = []
    for collision in root.findall(".//collision"):
        sdf_tag = collision.find("sdf")
        if sdf_tag is None:
            continue
        mesh_tag = collision.find("geometry/mesh")
        if mesh_tag is None or not mesh_tag.get("filename"):
            continue
        mesh_filename = str(mesh_tag.get("filename"))
        mesh_path = _resolve_urdf_mesh_path(urdf_path, mesh_filename)
        markers.append(
            _UrdfSdfCollisionMarker(
                mesh_stem=_usd_safe_identifier(mesh_path.stem),
                mesh_filename=mesh_filename,
                resolution=_parse_optional_int(sdf_tag.get("resolution")),
                margin=_parse_optional_float(sdf_tag.get("margin")),
                narrow_band_thickness=_parse_optional_float(
                    sdf_tag.get("narrow_band_thickness")
                    or sdf_tag.get("narrowBandThickness")
                ),
                subgrid_resolution=_parse_optional_int(
                    sdf_tag.get("subgrid_resolution")
                    or sdf_tag.get("subgridResolution")
                ),
            )
        )
    return markers


def _apply_urdf_sdf_collision_markers(
    usd_path: str,
    source_asset_path: str,
    markers: list[_UrdfSdfCollisionMarker],
) -> None:
    if not markers:
        return

    from pxr import Usd, UsdPhysics

    from isaaclab.sim.schemas import SDFMeshPropertiesCfg, define_mesh_collision_properties

    raw_usd_path = Path(usd_path)
    physics_usd_path = raw_usd_path.parent / "configuration" / f"{raw_usd_path.stem}_physics.usd"
    edit_usd_path = physics_usd_path if physics_usd_path.exists() else raw_usd_path

    stage = Usd.Stage.Open(str(edit_usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying URDF SDF markers: {edit_usd_path}")
    stage.Load()

    marker_by_stem = {marker.mesh_stem: marker for marker in markers}
    matched: dict[str, int] = {marker.mesh_stem: 0 for marker in markers}

    fallback_matches = []
    collider_matches = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        path = prim.GetPath().pathString
        path_parts = [part for part in path.split("/") if part]
        marker = next((marker_by_stem[part] for part in path_parts if part in marker_by_stem), None)
        if marker is None:
            continue
        if path.startswith("/colliders/"):
            collider_matches.append((prim, marker))
        else:
            fallback_matches.append((prim, marker))

    for prim, marker in collider_matches or fallback_matches:
        define_mesh_collision_properties(
            str(prim.GetPath()),
            SDFMeshPropertiesCfg(
                sdf_margin=marker.margin,
                sdf_narrow_band_thickness=marker.narrow_band_thickness,
                sdf_resolution=marker.resolution,
                sdf_subgrid_resolution=marker.subgrid_resolution,
            ),
            stage=stage,
        )
        matched[marker.mesh_stem] += 1

    stage.GetRootLayer().Save()

    matched_count = sum(matched.values())
    missing = [stem for stem, count in matched.items() if count == 0]
    if missing:
        print(
            f"[scene_utils] warning: URDF SDF markers in {source_asset_path!r} did not match "
            f"USD collision prims for mesh stems {missing}",
            flush=True,
        )
    if matched_count:
        details = ", ".join(f"{stem}:{count}" for stem, count in matched.items() if count)
        print(
            f"[scene_utils] applied URDF SDF collision markers to {matched_count} prims "
            f"in {edit_usd_path.name} ({details})",
            flush=True,
        )


def _urdf_box_size(urdf_path) -> tuple[float, float, float]:
    """The table's collision ``<box size>`` (the single-box table URDFs)."""
    boxes = ET.parse(urdf_path).getroot().findall("link/collision/geometry/box")
    if len(boxes) != 1:
        raise ValueError(f"{urdf_path}: expected one collision box, got {len(boxes)}")
    return tuple(float(v) for v in boxes[0].get("size").split())


def _generate_scaled_table_urdfs(
    base_urdf_path: str,
    num_variants: int,
    scale_range_x: tuple[float, float],
    scale_range_y: tuple[float, float],
    out_dir: Path,
    seed: int = 0,
) -> tuple[list[str], list[tuple[float, float]]]:
    """Write `num_variants` scaled copies of a single-box table URDF.

    Each variant samples (sx, sy) independently from the configured ranges
    (Z scale held at 1.0 so the table surface height matches what the policy
    was trained on). The base URDF must have a single `<box size="X Y Z"/>`
    in both the `<visual>` and `<collision>` blocks (matches the bundled
    `assets/urdf/table_narrow.urdf`).

    Returns the list of written URDF paths, in deterministic order.
    """
    import re
    import numpy as np

    base_text = Path(base_urdf_path).read_text()
    match = re.search(r'<box\s+size="([\d.\-+eE\s]+)"\s*/>', base_text)
    if match is None:
        raise ValueError(
            f"table URDF {base_urdf_path!r} has no <box size=\"...\"/> element; "
            "scaling helper only supports the simple single-box table."
        )
    base_dims = tuple(float(v) for v in match.group(1).split())
    if len(base_dims) != 3:
        raise ValueError(
            f"expected 3-element <box size>, got {base_dims!r} from {base_urdf_path}"
        )

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    scales: list[tuple[float, float]] = []
    for i in range(int(num_variants)):
        sx = float(rng.uniform(*scale_range_x))
        sy = float(rng.uniform(*scale_range_y))
        new_size = f"{base_dims[0] * sx:.6f} {base_dims[1] * sy:.6f} {base_dims[2]:.6f}"
        new_text = re.sub(
            r'<box\s+size="[\d.\-+eE\s]+"\s*/>',
            f'<box size="{new_size}"/>',
            base_text,
        )
        path = out_dir / f"table_variant_{i:03d}.urdf"
        path.write_text(new_text)
        paths.append(str(path))
        scales.append((sx, sy))
    return paths, scales


def _convert_urdf_to_usd(
    asset_path: str,
    usd_work_dir: Path,
    *,
    fix_base: bool,
    self_collision: bool | None = None,
    replace_cylinders_with_capsules: bool = False,
    joint_drive=None,
) -> str:
    converter_asset_path = _prepare_urdf_for_isaacsim(asset_path, usd_work_dir)
    cfg_kwargs = dict(
        asset_path=converter_asset_path,
        usd_dir=str(usd_work_dir / Path(asset_path).stem),
        force_usd_conversion=True,
        fix_base=fix_base,
        merge_fixed_joints=True,
        make_instanceable=False,
        replace_cylinders_with_capsules=replace_cylinders_with_capsules,
        joint_drive=joint_drive,
    )
    if self_collision is not None:
        cfg_kwargs["self_collision"] = self_collision
    usd_path = UrdfConverter(UrdfConverterCfg(**cfg_kwargs)).usd_path
    _apply_urdf_sdf_collision_markers(
        usd_path,
        converter_asset_path,
        _parse_urdf_sdf_collision_markers(converter_asset_path),
    )
    return usd_path



def _apply_self_collision_filters(usd_path: str, pairs: dict[str, tuple[str, ...]]) -> int:
    """Author ``FilteredPairsAPI`` on the robot's bodies so ``pairs`` (``RobotSpec.
    self_collision_filter``) do not collide with each other; returns how many were authored.

    With the articulation's self-collisions on, this is Isaac Gym's behaviour in SimToolReal:
    everything collides except the listed pairs. Ported from simtoolreal 84058661, with one
    change: every name must be a body of the imported robot. Upstream skipped unknown names,
    which is how a filter list for the wrong robot (or pre-merge_fixed_joints names) would leave
    self-collision on with nothing filtered, and the hand exploding at reset.
    """
    from pxr import Usd, UsdPhysics

    raw_usd_path = Path(usd_path)
    physics_usd_path = raw_usd_path.parent / "configuration" / f"{raw_usd_path.stem}_physics.usd"
    edit_usd_path = physics_usd_path if physics_usd_path.exists() else raw_usd_path
    stage = Usd.Stage.Open(str(edit_usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying self-collision filters: {edit_usd_path}")

    body_by_name = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    unknown = sorted({n for a, bs in pairs.items() for n in (a, *bs)} - set(body_by_name))
    if unknown:
        raise RuntimeError(
            f"self-collision filter names {unknown} are not bodies of {edit_usd_path.name} "
            f"(bodies: {sorted(body_by_name)}); names must be post-merge_fixed_joints"
        )

    authored = 0
    for link, partners in pairs.items():
        rel = UsdPhysics.FilteredPairsAPI.Apply(body_by_name[link]).CreateFilteredPairsRel()
        existing = set(rel.GetTargets())
        for partner in partners:
            path = body_by_name[partner].GetPath()
            if path not in existing:
                rel.AddTarget(path)
                existing.add(path)
                authored += 1
    stage.GetRootLayer().Save()
    print(
        f"[scene_utils] self-collision: filtered {authored} link pairs "
        f"across {len(body_by_name)} robot bodies in {edit_usd_path.name}",
        flush=True,
    )
    return authored


def _robot_joint_drive_cfg():
    # DriveAPI prims must exist for ImplicitActuator runtime gains to land.
    return UrdfConverterCfg.JointDriveCfg(
        drive_type="force", target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    )


def _bake_usd(
    raw_usd_path: str,
    bake_root: Path,
    baked_subdir: str,
    *,
    props: dict | None = None,
    apply_physx_articulation: bool = False,
    collision_enabled: bool | None = None,
) -> str:
    """Copy raw USD into bake_root/baked_subdir and pre-author physics props.

    ``props`` keys come from ``_PHYSICS_SPECS``; ``None`` values are skipped,
    and keys whose group doesn't match a prim's APIs are skipped per-prim.
    """
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    vtypes = {
        "Bool": Sdf.ValueTypeNames.Bool,
        "Float": Sdf.ValueTypeNames.Float,
        "Int": Sdf.ValueTypeNames.Int,
    }
    props = props or {}

    raw = Path(raw_usd_path)
    baked_usd_path = bake_root / baked_subdir / raw.parent.name / raw.name
    baked_usd_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, baked_usd_path)
    for child in raw.parent.iterdir():
        if child.name.startswith(".") or child.name in (raw.name, "config.yaml"):
            continue
        dst = baked_usd_path.parent / child.name
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child, dst)
        elif child.is_file():
            shutil.copy2(child, dst)

    stage = Usd.Stage.Open(str(baked_usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open baked USD: {baked_usd_path}")
    root = stage.GetDefaultPrim()
    if not (root and root.IsValid()):
        root = next((p for p in stage.GetPseudoRoot().GetChildren() if p.IsValid()), None)
    if root is None:
        raise RuntimeError(f"No root prim in USD: {baked_usd_path}")

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    for prim in Usd.PrimRange(root):
        is_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if is_rb:
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if is_art and apply_physx_articulation:
            PhysxSchema.PhysxArticulationAPI.Apply(prim)
        for key, val in props.items():
            if val is None:
                continue
            group, attr_name, vtype_str = _PHYSICS_SPECS[key]
            if group == "rb" and not is_rb:
                continue
            if group == "art" and not is_art:
                continue
            _set_usd_attr(prim, attr_name, val, vtypes[vtype_str])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            px = PhysxSchema.PhysxCollisionAPI(prim) or PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(_CONTACT_OFFSET)
            px.CreateRestOffsetAttr().Set(_REST_OFFSET)
            if collision_enabled is not None:
                ce = UsdPhysics.CollisionAPI(prim)
                (ce.GetCollisionEnabledAttr() or ce.CreateCollisionEnabledAttr()).Set(
                    collision_enabled
                )

    stage.GetRootLayer().Save()
    return str(baked_usd_path)


# ----------------------------------------------------------------------------
# Runtime material setup (post-launch, via PhysX views)
# ----------------------------------------------------------------------------

def apply_physx_material_properties(env) -> None:
    """Set contact materials through PhysX tensor views.

    Follows Isaac Lab's large-scale randomization path: avoid post-spawn USD
    relationship authoring and per-clone material prims. Must run after
    ``DirectRLEnv`` starts the simulator and ``root_physx_view`` exists.
    """
    assets_cfg = env.cfg.assets
    if not assets_cfg.modify_asset_frictions:
        return

    t0 = time.perf_counter()
    default = torch.tensor(
        [float(assets_cfg.robot_friction), float(assets_cfg.robot_friction), 0.0],
        dtype=torch.float32, device="cpu",
    )
    fingertip = torch.tensor(
        [float(assets_cfg.finger_tip_friction), float(assets_cfg.finger_tip_friction), 0.0],
        dtype=torch.float32, device="cpu",
    )
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")

    dr = env.cfg.domain_randomization
    n_buckets = int(dr.friction_n_buckets)
    ft_lo, ft_hi = float(dr.fingertip_friction_scale_range[0]), float(dr.fingertip_friction_scale_range[1])
    obj_lo, obj_hi = float(dr.object_friction_scale_range[0]), float(dr.object_friction_scale_range[1])
    ft_active = (ft_lo, ft_hi) != (1.0, 1.0)
    obj_active = (obj_lo, obj_hi) != (1.0, 1.0)

    robot_view = env.robot.root_physx_view
    robot_materials = robot_view.get_material_properties()
    robot_materials[:] = default

    fingertip_mask = torch.zeros(robot_view.max_shapes, dtype=torch.bool, device="cpu")
    shape_start = 0
    for link_name, link_path in zip(robot_view.shared_metatype.link_names, robot_view.link_paths[0]):
        link_view = env.robot._physics_sim_view.create_rigid_body_view(link_path)
        shape_end = shape_start + link_view.max_shapes
        if link_name in env.robot_spec.fingertip_link_names:
            robot_materials[:, shape_start:shape_end] = fingertip
            fingertip_mask[shape_start:shape_end] = True
        shape_start = shape_end
    if shape_start != robot_view.max_shapes:
        raise RuntimeError(
            f"Robot shape count mismatch while assigning materials: "
            f"computed {shape_start}, view reports {robot_view.max_shapes}."
        )

    # Per-env bucketed fingertip friction (init-only). Quantizing to
    # `n_buckets` distinct values caps the PhysX material count regardless
    # of n_envs.
    if ft_active:
        ft_base = float(assets_cfg.finger_tip_friction)
        bucket_vals = torch.linspace(ft_lo, ft_hi, n_buckets) * ft_base  # (B,)
        bucket_idx = torch.randint(0, n_buckets, (env.num_envs,))
        per_env_ft = bucket_vals[bucket_idx]  # (N_envs,)
        ft_indices = fingertip_mask.nonzero(as_tuple=True)[0]
        if ft_indices.numel() > 0:
            robot_materials[:, ft_indices, 0] = per_env_ft.unsqueeze(-1)
            robot_materials[:, ft_indices, 1] = per_env_ft.unsqueeze(-1)

    robot_view.set_material_properties(robot_materials, env_ids)

    for name in ("table", "object", "goal_viz", "hole"):
        if not hasattr(env, name):
            continue
        view = getattr(env, name).root_physx_view
        materials = view.get_material_properties()
        materials[:] = default
        if name == "object" and obj_active:
            # Was `robot_friction`. Dormant on both counts today -- the branch needs
            # `object_friction_scale_range != (1.0, 1.0)` to run at all, and the two frictions
            # are both 0.5 -- so it changes no current result. It would have scaled object
            # friction off the robot's base the moment either of those stopped being true.
            obj_base = float(assets_cfg.object_friction)
            bucket_vals = torch.linspace(obj_lo, obj_hi, n_buckets) * obj_base
            bucket_idx = torch.randint(0, n_buckets, (env.num_envs,))
            per_env_obj = bucket_vals[bucket_idx]  # (N_envs,)
            materials[:, :, 0] = per_env_obj.unsqueeze(-1)
            materials[:, :, 1] = per_env_obj.unsqueeze(-1)
        view.set_material_properties(materials, env_ids)

    _log_scene_step(t0, "applied PhysX material properties")


# ----------------------------------------------------------------------------
# Scene assembly
# ----------------------------------------------------------------------------

def _materialize_env_prims(env) -> None:
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")


def _build_object_scale_tensor(env, object_scales_normalized, num_object_usds: int) -> None:
    num_envs = env.num_envs
    object_prim_paths = find_matching_prim_paths("/World/envs/env_.*/Object")
    if len(object_prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Object prims after MultiUsdFileCfg spawn, "
            f"got {len(object_prim_paths)}. Cloner-drop bug may have returned."
        )

    env._object_scale_per_env = torch.zeros(num_envs, 3, device=env.device, dtype=torch.float32)
    env._object_asset_index_per_env = torch.zeros(num_envs, device=env.device, dtype=torch.long)
    for source_idx, obj_path in enumerate(object_prim_paths):
        env_id = int(obj_path.rsplit("/", 2)[-2].removeprefix("env_"))
        asset_index = source_idx % num_object_usds
        env._object_scale_per_env[env_id] = torch.tensor(
            object_scales_normalized[asset_index], device=env.device, dtype=torch.float32,
        )
        env._object_asset_index_per_env[env_id] = asset_index


def setup_scene(env) -> None:
    """Build and register robot, table, object, goal, ground, and light."""
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(
        setup_t0,
        f"setup start num_envs={env.num_envs} "
        f"num_assets_per_type={assets_cfg.num_assets_per_type}",
    )

    # 1. Generate procedural URDFs in a per-launch temp dir.
    env._tmp_asset_dir = tempfile.mkdtemp(prefix="play_assets_")
    urdf_paths, object_scales_normalized = generate_handle_head_urdfs(
        handle_head_types=tuple(assets_cfg.handle_head_types),
        num_per_type=assets_cfg.num_assets_per_type,
        out_dir=env._tmp_asset_dir,
        shuffle=assets_cfg.shuffle_assets,
    )
    if not urdf_paths:
        raise ValueError(
            "No procedural object URDFs were generated. "
            "Check cfg.assets.handle_head_types and num_assets_per_type."
        )
    env._object_urdf_paths = [str(path) for path in urdf_paths]
    _log_scene_step(setup_t0, f"generated {len(urdf_paths)} object URDFs")

    # 2. Convert URDFs -> raw USDs -> role-specific baked USDs.
    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    object_raw_usds = [
        _convert_urdf_to_usd(
            str(urdf), usd_work_dir, fix_base=False, replace_cylinders_with_capsules=True,
        )
        for urdf in urdf_paths
    ]
    object_usd_paths = [
        _bake_usd(usd, bake_root, "object", props=dict(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=1000.0, articulation_enabled=False,
        ))
        for usd in object_raw_usds
    ]
    goalviz_usd_paths = [
        _bake_usd(usd, bake_root, "goalviz", props=dict(
            kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
        ), collision_enabled=False)
        for usd in object_raw_usds
    ]

    # Self-collision is per registered task (cfg.robot_self_collision): on, it is SimToolReal's
    # Isaac Gym setup -- everything collides except the spec's filtered pairs.
    self_collision = bool(getattr(env.cfg, "robot_self_collision", False))
    robot_converted_usd = _convert_urdf_to_usd(
        assets_cfg.robot_urdf, usd_work_dir,
        fix_base=True, self_collision=self_collision,
        joint_drive=_robot_joint_drive_cfg(),
    )
    if self_collision:
        _apply_self_collision_filters(robot_converted_usd, env.robot_spec.self_collision_filter)
    robot_usd_path = _bake_usd(
        robot_converted_usd,
        bake_root, "robot",
        props=dict(
            disable_gravity=True, max_depenetration_velocity=1000.0,
            enabled_self_collisions=self_collision,
            solver_position_iterations=8, solver_velocity_iterations=0,
        ),
        apply_physx_articulation=True,
    )
    # Table USD(s). When table_scale_range_x/y are non-trivial and
    # table_scale_num_variants > 1, pre-bake N scaled URDF variants and pass
    # them as a list to RigidObject — Isaac Lab's MultiUsdFileCfg cycles
    # through the list, giving each env one of the variants. Z scale is held
    # at 1.0 so the table surface height matches the policy's expectation.
    scale_range_x = tuple(float(v) for v in getattr(assets_cfg, "table_scale_range_x", (1.0, 1.0)))
    scale_range_y = tuple(float(v) for v in getattr(assets_cfg, "table_scale_range_y", (1.0, 1.0)))
    n_table_variants = int(getattr(assets_cfg, "table_scale_num_variants", 1))
    table_scale_is_trivial = (
        scale_range_x == (1.0, 1.0) and scale_range_y == (1.0, 1.0)
    ) or n_table_variants <= 1
    if table_scale_is_trivial:
        table_usd_paths = [_bake_usd(
            _convert_urdf_to_usd(assets_cfg.table_urdf, usd_work_dir, fix_base=False),
            bake_root, "table",
            props=dict(
                kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
            ),
        )]
        # Single (sx, sy) = (1.0, 1.0) for downstream consumers (eval viz).
        env._table_variant_scales = [(1.0, 1.0)]
    else:
        variant_urdf_dir = Path(env._tmp_asset_dir) / "table_variants"
        variant_urdf_paths, variant_scales = _generate_scaled_table_urdfs(
            base_urdf_path=assets_cfg.table_urdf,
            num_variants=n_table_variants,
            scale_range_x=scale_range_x,
            scale_range_y=scale_range_y,
            out_dir=variant_urdf_dir,
            # Deterministic across runs so the on-disk variants are stable.
            # The env-level seed governs which variant lands in which env via
            # Isaac Lab's round-robin spawn ordering.
            seed=0,
        )
        env._table_variant_scales = list(variant_scales)
        table_usd_paths = [
            _bake_usd(
                _convert_urdf_to_usd(p, usd_work_dir, fix_base=False),
                bake_root, f"table_variant_{idx:03d}",
                props=dict(
                    kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
                ),
            )
            for idx, p in enumerate(variant_urdf_paths)
        ]
        # variant_scales already stashed above for downstream consumers.
        _log_scene_step(
            setup_t0,
            f"baked {len(table_usd_paths)} scaled table USD variants "
            f"x_range={scale_range_x} y_range={scale_range_y}",
        )
    _log_scene_step(setup_t0, "resolved baked USDs")

    # 3. Pre-create env roots so regex spawns resolve to every env.
    _materialize_env_prims(env)

    # 4. Spawn assets.
    env.robot = Articulation(build_robot_articulation_usd_cfg(robot_usd_path, spec=env.robot_spec))
    env.table = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Table", table_usd_paths))
    env.object = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_paths))
    env.goal_viz = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_paths))
    _log_scene_step(setup_t0, "spawned robot/table/object/goalviz")

    # 5. Ground plane + dome light (global, outside env_*).
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 6. Per-env scale tensor for spawned Objects.
    _build_object_scale_tensor(env, object_scales_normalized, len(object_usd_paths))
    # ... and each env's object collision-box corners and the table's half height, which keep
    # goals clear of the table (reset.goal_table_clearance_m).
    pool_corners = torch.stack([
        torch.as_tensor(collision_corners(p), device=env.device) for p in env._object_urdf_paths
    ])
    env._object_collision_corners = pool_corners[env._object_asset_index_per_env]
    env._table_half_height = _urdf_box_size(assets_cfg.table_urdf)[2] / 2

    # 7. Register with scene so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    hide_goal_viz_for_student_camera(env)
    setup_student_camera(env)
    _log_scene_step(setup_t0, "registered assets with scene")
