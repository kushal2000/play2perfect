"""Render a multi-step assembly back to back in Isaac Sim, one policy per step.

Each step is its own precise-assembly problem with its own policy (e.g.
beam_assembly_step1 inserts part 2 into the base, beam_assembly_step2 inserts
part 0 into base + part 2). The steps are chained so the video reads as one
continuous take:

  * every step uses the same hole pose, so the assembly stays put;
  * step k+1 starts from the robot's joint state (and PD targets) at the end
    of step k, captured just before the env auto-resets on retract success;
  * parts of later steps are drawn at their start poses from the first frame,
    and parts placed by earlier steps are drawn assembled, so nothing pops in
    or out at the cut.

Each step runs in its own Isaac Sim process (a scene cannot be rebuilt in
place); this script is both the orchestrator and, with ``--stage``, the worker.
The fixture's collision-proxy visuals and the goal marker are hidden, and the
assembly parts are drawn from their real meshes under a studio floor and
lighting. Frames go straight to H.264; the steps are concatenated at the end.

    python evaluation/render_assembly_sequence.py \\
        --robot flexiv_wuji_right_human_like \\
        --problems beam_assembly_step1 beam_assembly_step2 \\
        --out renders/beam_assembly
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_isaacsim import (  # noqa: E402
    DEFAULT_AGENT,
    DEFAULT_ROBOT_POLICY_DIRS,
    ROBOT_TASKS,
)

FPS = 60  # one frame per policy step (60 Hz control)

# Per-part colours (linear-ish sRGB), applied to the moving part and to its
# static copies alike so a part looks the same before, during and after its step.
PART_COLORS = [
    (0.20, 0.42, 0.72),  # blue
    (0.86, 0.42, 0.18),  # orange
    (0.30, 0.62, 0.42),  # green
    (0.62, 0.36, 0.70),  # purple
]
BASE_COLOR = (0.80, 0.64, 0.46)  # light wood
FLOOR_COLOR = (0.40, 0.41, 0.43)


# ---------------------------------------------------------------------------
# Pose helpers (numpy, quaternions are wxyz)
# ---------------------------------------------------------------------------


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(m):
    from scipy.spatial.transform import Rotation as R

    x, y, z, w = R.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


def pose_to_mat(pos, quat_wxyz):
    t = np.eye(4)
    t[:3, :3] = quat_to_mat(quat_wxyz)
    t[:3, 3] = pos
    return t


def rpy_to_mat(rpy):
    from scipy.spatial.transform import Rotation as R

    return R.from_euler("xyz", rpy).as_matrix()


# ---------------------------------------------------------------------------
# URDF visuals -> list of (mesh path or box size, 4x4 in object frame)
# ---------------------------------------------------------------------------


def urdf_visuals(urdf_path: Path) -> list[dict]:
    """Visual geometry of a fixed-joint URDF, in the root link's frame."""
    root = ET.parse(urdf_path).getroot()
    link_tf = {}
    children = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        t = np.eye(4)
        if origin is not None:
            t[:3, :3] = rpy_to_mat([float(v) for v in origin.get("rpy", "0 0 0").split()])
            t[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        children.setdefault(parent, []).append((child, t))
    child_links = {c for kids in children.values() for c, _ in kids}
    roots = [link.get("name") for link in root.findall("link") if link.get("name") not in child_links]
    stack = [(roots[0], np.eye(4))]
    while stack:
        name, tf = stack.pop()
        link_tf[name] = tf
        for child, t in children.get(name, []):
            stack.append((child, tf @ t))

    out = []
    for link in root.findall("link"):
        for vis in link.findall("visual"):
            origin = vis.find("origin")
            t = np.eye(4)
            if origin is not None:
                t[:3, :3] = rpy_to_mat([float(v) for v in origin.get("rpy", "0 0 0").split()])
                t[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            geom = vis.find("geometry")
            mesh = geom.find("mesh")
            box = geom.find("box")
            entry = {"tf": (link_tf[link.get("name")] @ t).tolist()}
            if mesh is not None:
                entry["mesh"] = str((urdf_path.parent / mesh.get("filename")).resolve())
                entry["scale"] = [float(v) for v in mesh.get("scale", "1 1 1").split()]
            elif box is not None:
                entry["box"] = [float(v) for v in box.get("size").split()]
            else:
                continue
            out.append(entry)
    return out


def load_part_mesh(visuals: list[dict]):
    """All mesh visuals merged into one trimesh in the object frame."""
    import trimesh

    parts = []
    for v in visuals:
        if "mesh" in v:
            m = trimesh.load(v["mesh"], force="mesh")
            m.apply_scale(v["scale"])
        else:  # primitive box visual (e.g. the L-peg)
            m = trimesh.creation.box(extents=v["box"])
        m.apply_transform(np.array(v["tf"]))
        parts.append(m)
    return trimesh.util.concatenate(parts)


def resting_pose(mesh, xy, yaw_deg, table_top_z):
    """Most stable resting pose of ``mesh`` on the table at ``xy``, turned by ``yaw``."""
    tfs, probs = mesh.compute_stable_poses(n_samples=1)
    rest = tfs[int(np.argmax(probs))]
    yaw = math.radians(yaw_deg)
    rz = np.eye(4)
    rz[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    t = rz @ rest
    # compute_stable_poses puts the resting face on z=0 and the centroid at xy=0.
    verts = mesh.vertices @ t[:3, :3].T + t[:3, 3]
    t[0, 3] += xy[0] - verts[:, 0].mean()
    t[1, 3] += xy[1] - verts[:, 1].mean()
    t[2, 3] += table_top_z - verts[:, 2].min() + 0.001
    return t[:3, 3].tolist(), mat_to_quat(t[:3, :3]).tolist()


# ---------------------------------------------------------------------------
# Plan (orchestrator side, no Isaac imports)
# ---------------------------------------------------------------------------


def base_meshes(problem) -> list[dict]:
    """Display meshes for the fixture the first step inserts into.

    Fabrica: the real A-frame part meshes. Otherwise the non-SDF sibling of the
    receptive URDF (e.g. one_leg.urdf next to one_leg_sdf_hybrid.urdf), whose
    visuals are the real meshes; falls back to the receptive URDF itself.
    """
    if "fabrica" in problem.receptive_urdf:
        return fabrica_base_meshes(problem)
    urdf = (REPO_ROOT / "assets" / problem.receptive_urdf).resolve()
    plain = urdf.with_name(urdf.name.replace("_sdf_hybrid", ""))
    return urdf_visuals(plain if plain.is_file() else urdf)


def fabrica_base_meshes(problem) -> list[dict]:
    """Real A-frame meshes of the fixture parts nobody inserts in this sequence.

    The sdf_hybrid fixture URDFs draw the base as collision boxes; the
    assembled-frame ``<pid>.obj`` files sit at identity in the receptive frame.
    """
    urdf = (REPO_ROOT / "assets" / problem.receptive_urdf).resolve()
    assembly_dir = urdf.parent.parent
    order = json.loads((assembly_dir / "assembly_order.json").read_text())
    # fixture of the first step = parts that come before the inserted part
    inserted = Path(urdf).stem.split("_")[1]  # part_<pid>[_sdf_hybrid]
    steps = order["steps"]
    base_ids = steps[: steps.index(inserted)]
    return [{"mesh": str(assembly_dir / pid / f"{pid}.obj"), "scale": [1, 1, 1],
             "tf": np.eye(4).tolist(), "part_id": pid} for pid in base_ids]


def build_plan(args) -> dict:
    from evaluation.objects import NAME_TO_OBJECT
    from evaluation.problems import PROBLEM_REGISTRY

    task = ROBOT_TASKS[args.robot]
    policy_dir = REPO_ROOT / (args.policies_dir or DEFAULT_ROBOT_POLICY_DIRS[args.robot])
    # Table geometry of the robot's precise-assembly task (yaml overlay values).
    import yaml

    cfg_yaml = {
        "kuka_sharpa": "PreciseAssembly.yaml",
        "flexiv_wuji_right_human_like": "PreciseAssemblyFlexivWujiRight.yaml",
    }[args.robot]
    y = yaml.safe_load((REPO_ROOT / "isaacsimenvs" / "cfg" / "task" / cfg_yaml).read_text())
    table_xy = [float(v) for v in y["reset"].get("table_reset_xy", (0.0, 0.0))]
    table_box = [float(v) for v in ET.parse(REPO_ROOT / y["assets"]["table_urdf"]).getroot()
                 .find("link/visual/geometry/box").get("size").split()]
    table_top = float(y["reset"]["table_reset_z"]) + table_box[2] / 2

    hole_rel = [float(v) for v in args.hole_xy]  # relative to the table centre
    first_problem = PROBLEM_REGISTRY[args.problems[0]]
    hole_pos = [table_xy[0] + hole_rel[0], table_xy[1] + hole_rel[1],
                table_top + float(first_problem.hole_z_offset)]

    stages = []
    for i, name in enumerate(args.problems):
        problem = PROBLEM_REGISTRY[name]
        obj = NAME_TO_OBJECT[problem.insertion_object_name]
        visuals = urdf_visuals(Path(obj.urdf_path).resolve())
        mesh = load_part_mesh(visuals)
        sx, sy, yaw = args.starts[i] if i < len(args.starts) else (0.0, -0.06, 0.0)
        start_pos, start_quat = resting_pose(
            mesh, (table_xy[0] + sx, table_xy[1] + sy), yaw, table_top
        )
        final = problem.final_insert_pose_rel_receptive  # x y z qx qy qz qw, receptive frame
        final_tf = pose_to_mat(final[:3], [final[6], final[3], final[4], final[5]])
        assembled_tf = pose_to_mat(hole_pos, [1, 0, 0, 0]) @ final_tf
        ckpt = Path(args.checkpoints[i]) if args.checkpoints else policy_dir / name / "model.pth"
        stages.append({
            "problem": name,
            "checkpoint": str(ckpt.resolve()),
            "visuals": visuals,
            "color": PART_COLORS[i % len(PART_COLORS)],
            "start_pose": start_pos + start_quat,  # env-local xyz + wxyz
            "assembled_tf": assembled_tf.tolist(),
        })

    first = PROBLEM_REGISTRY[args.problems[0]]
    return {
        "task": task,
        "robot": args.robot,
        "table_xy": table_xy,
        "table_top": table_top,
        "hole_rel": hole_rel,
        "hole_pos": hole_pos,
        "base_meshes": base_meshes(first),
        "table_size": table_box[:2],
        "stages": stages,
        "camera": {
            "eye": [float(v) for v in args.camera_eye],
            "lookat": [float(v) for v in args.camera_lookat],
            "orbit_deg": float(args.orbit_deg),
        },
        "resolution": [int(args.width), int(args.height)],
        "max_steps": int(args.max_steps),
        "hold_end_s": float(args.hold_end_s),
        "path_tracing": bool(args.path_tracing),
        "record_only": bool(args.record_only),
        "insertion_tolerance": args.insertion_tolerance,
        "fps": FPS,
        "robot_urdf": str((REPO_ROOT / y["assets"]["robot_urdf"]).resolve()),
        "spp": int(args.spp),
    }


# ---------------------------------------------------------------------------
# Worker (one Isaac Sim process per stage)
# ---------------------------------------------------------------------------


def _scene_dressing(env, plan: dict, stage_idx: int) -> None:
    """Hide proxies, add floor/lights, draw base + other steps' parts."""
    import trimesh
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    origin = env.scene.env_origins[0].tolist()

    def hide(path):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()

    hide("/World/ground")
    hide("/World/envs/env_0/GoalViz")
    hide("/World/envs/env_0/Hole")
    hide("/World/envs/env_0/Table")
    hide("/World/Light")

    looks = "/World/Render/Looks"

    def material(name, rgb, roughness=0.55, metallic=0.0):
        path = f"{looks}/{name}"
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def add_mesh(path, mesh, tf, mat):
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in mesh.vertices]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(mesh.faces.reshape(-1).tolist()))
        m.CreateSubdivisionSchemeAttr("none")
        normals = np.repeat(mesh.face_normals, 3, axis=0)  # faceVarying: crisp edges
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in normals]))
        m.SetNormalsInterpolation("faceVarying")
        xf = UsdGeom.Xformable(m)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(np.array(tf).T.tolist()))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

    def add_visuals(path, visuals, tf_world, mat):
        for j, v in enumerate(visuals):
            local = np.array(v["tf"])
            if "mesh" in v:
                mesh = trimesh.load(v["mesh"], force="mesh")
                mesh.apply_scale(v["scale"])
            else:
                mesh = trimesh.creation.box(extents=v["box"])
            add_mesh(f"{path}_{j}", mesh, np.array(tf_world) @ local, mat)

    def world(tf_local):
        t = np.array(tf_local, dtype=float).copy()
        t[:3, 3] += origin
        return t

    # Floor flush with the table top: the table becomes the studio surface.
    floor = trimesh.creation.box(extents=[60.0, 60.0, 0.02])
    floor_tf = np.eye(4)
    floor_tf[:3, 3] = [origin[0], origin[1], origin[2] + plan["table_top"] - 0.0105]
    add_mesh("/World/Render/Floor", floor, floor_tf, material("floor", FLOOR_COLOR, 0.7))

    hole_tf = pose_to_mat(plan["hole_pos"], [1, 0, 0, 0])
    base_mat = material("base", BASE_COLOR, 0.6)
    for j, v in enumerate(plan["base_meshes"]):
        add_visuals(f"/World/Render/Base{j}", [v], world(hole_tf), base_mat)

    for k, st in enumerate(plan["stages"]):
        mat = material(f"part{k}", st["color"], 0.45)
        if k < stage_idx:  # placed by an earlier step
            add_visuals(f"/World/Render/Done{k}", st["visuals"], world(st["assembled_tf"]), mat)
        elif k > stage_idx:  # waiting on the table
            sp = st["start_pose"]
            add_visuals(f"/World/Render/Todo{k}", st["visuals"], world(pose_to_mat(sp[:3], sp[3:])), mat)
        else:  # the moving part: the env's own (instanced, uneditable) visual is
            # hidden and a recoloured copy follows the rigid body every frame
            hide("/World/envs/env_0/Object")
            UsdGeom.Xform.Define(stage, "/World/Render/Moving")
            add_visuals("/World/Render/Moving/part", st["visuals"], np.eye(4), mat)

    # Lights: soft key from the front-left, dim fill dome.
    dome = UsdLux.DomeLight.Define(stage, "/World/Render/Dome")
    dome.CreateIntensityAttr(480.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Render/Key")
    key.CreateIntensityAttr(1900.0)
    key.CreateAngleAttr(6.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, -35.0))


def _follow_object(env) -> None:
    """Move the recoloured copy of the moving part onto the env's rigid body."""
    from pxr import Gf, UsdGeom
    from isaaclab.sim.utils import get_current_stage

    pos = env.object.data.root_pos_w[0].tolist()
    quat = env.object.data.root_quat_w[0].tolist()  # wxyz
    xf = UsdGeom.Xformable(get_current_stage().GetPrimAtPath("/World/Render/Moving"))
    ops = xf.GetOrderedXformOps()
    if not ops:
        ops = [xf.AddTranslateOp(), xf.AddOrientOp()]
    ops[0].Set(Gf.Vec3d(*pos))
    ops[1].Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))


def _camera_at(plan: dict, frame: int, total_frames: int):
    cam = plan["camera"]
    hole = np.array(plan["hole_pos"])
    eye = hole + np.array(cam["eye"])
    look = hole + np.array(cam["lookat"])
    if cam["orbit_deg"]:
        a = math.radians(cam["orbit_deg"]) * (frame / max(1, total_frames) - 0.5)
        rel = eye - look
        rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        eye = look + rot @ rel
    return eye.tolist(), look.tolist()


def run_stage(args) -> int:
    plan = json.loads(Path(args.plan).read_text())
    k = int(args.stage)
    st = plan["stages"][k]
    out = Path(args.out)
    prev_state = None
    if k > 0:
        prev_state = json.loads((out / f"state_{k - 1}.json").read_text())
    frame_offset = prev_state["frame_end"] if prev_state else 0

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(p)
    launcher_args, _ = p.parse_known_args([])
    launcher_args.headless = True
    record_only = bool(plan.get("record_only"))
    launcher_args.enable_cameras = not record_only
    app = AppLauncher(launcher_args).app

    import importlib

    import gymnasium as gym
    import imageio_ffmpeg
    import torch

    import isaacsimenvs  # noqa: F401
    from evaluation.eval_isaacsim import _apply_env_overrides, _configure_agent, _load_env_cfg
    from isaacsimenvs.tasks.play.utils import reset_utils
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner, _load_checkpoint_weights

    task = plan["task"]
    cfg = _load_env_cfg(task)
    _apply_env_overrides(
        cfg, problem=st["problem"], goal_mode="preInsertAndFinal", random_goal_fraction=0.0,
        insertion_success_tolerance=plan.get("insertion_tolerance"),
        retract_success_tolerance=0.005, num_envs=1,
        sim_device="cuda:0", sdf=True, keep_dr=False, extra_overrides={},
    )
    hx, hy = plan["hole_rel"]
    cfg.precise_assembly.hole_x_range = (hx, hx)
    cfg.precise_assembly.hole_y_range = (hy, hy)
    cfg.precise_assembly.hole_yaw_range_deg = 0.0
    cfg.precise_assembly.goal_xy_obs_noise = 0.0
    cfg.reset.fixed_start_pose = list(st["start_pose"])
    cfg.viewer.resolution = tuple(plan["resolution"])
    r = cfg.sim.render
    r.rendering_mode = "quality"
    r.antialiasing_mode = "DLAA"
    r.enable_shadows = True
    r.enable_ambient_occlusion = True
    r.enable_reflections = True
    if plan["path_tracing"]:
        r.carb_settings = {
            "rtx.rendermode": "PathTracing",
            "rtx.pathtracing.spp": plan["spp"],
            "rtx.pathtracing.totalSpp": plan["spp"],
            "rtx.pathtracing.optixDenoiser.enabled": True,
        }

    # Start from the previous step's robot state instead of the default pose.
    if prev_state is not None:
        def _restore_robot(env, env_ids):
            dev = env.device
            pos = torch.tensor([prev_state["joint_pos"]], device=dev).expand(len(env_ids), -1)
            vel = torch.tensor([prev_state["joint_vel"]], device=dev).expand(len(env_ids), -1)
            env.robot.write_joint_state_to_sim(pos.clone(), vel.clone(), env_ids=env_ids)
            env._prev_targets[env_ids] = torch.tensor([prev_state["prev_targets"]], device=dev)
            env._cur_targets[env_ids] = torch.tensor([prev_state["cur_targets"]], device=dev)

        reset_utils._randomize_robot_dof_state = _restore_robot

    spec = gym.spec(task)
    mod, cls = spec.entry_point.split(":")
    env = getattr(importlib.import_module(mod), cls)(
        cfg=cfg, render_mode=None if record_only else "rgb_array"
    )
    if not record_only:
        _scene_dressing(env, plan, k)

    agent_cfg = _configure_agent(task, DEFAULT_AGENT, rl_device="cuda:0", num_envs=1,
                                 deterministic=True, games=1, extra_overrides={})
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device="cuda:0", clip_obs=clip_obs, clip_actions=clip_actions)
    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.set_weights(_load_checkpoint_weights(player, st["checkpoint"]))
    player.has_batch_dimension = True

    # Snapshot the robot right before the env auto-resets (retract success).
    snap = {}
    orig_reset_idx = env._reset_idx

    def _reset_idx(env_ids):
        if snap.get("armed"):
            snap["state"] = {
                "joint_pos": env.robot.data.joint_pos[0].tolist(),
                "joint_vel": env.robot.data.joint_vel[0].tolist(),
                "cur_targets": env._cur_targets[0].tolist(),
                "prev_targets": env._prev_targets[0].tolist(),
                "successes": int(env._successes[0]),
                "max_goals": int(env.env_max_goals[0]),
                "retract_succeeded": bool(env.retract_succeeded[0]),
            }
            snap["armed"] = False
            return None  # we stop here; keep the finished scene for the end hold
        return orig_reset_idx(env_ids)

    env._reset_idx = _reset_idx

    player.reset()
    obs = player.env_reset(wrapped)
    snap["armed"] = True

    # Rough total for the camera orbit: assume every step uses about max_steps/2.
    total = int(plan.get("orbit_total_frames") or len(plan["stages"]) * plan["max_steps"] // 2)

    # Rollout recording: world poses of every robot body and the moving part,
    # one row per policy step, so any renderer can replay it without the sim.
    rec = {"body_pos": [], "body_quat": [], "joint_pos": [], "object_pos": [], "object_quat": []}

    def record():
        rec["body_pos"].append(env.robot.data.body_pos_w[0].cpu().numpy().copy())
        rec["body_quat"].append(env.robot.data.body_quat_w[0].cpu().numpy().copy())
        rec["joint_pos"].append(env.robot.data.joint_pos[0].cpu().numpy().copy())
        rec["object_pos"].append(env.object.data.root_pos_w[0].cpu().numpy().copy())
        rec["object_quat"].append(env.object.data.root_quat_w[0].cpu().numpy().copy())

    def grab(frame_idx):
        if record_only:
            return None
        eye, look = _camera_at(plan, frame_idx, total)
        env.sim.set_camera_view(eye=eye, target=look)
        _follow_object(env)
        env.sim.render()
        return env.render()

    class _NoWriter:
        def send(self, _frame):
            pass

        def close(self):
            pass

    video = out / f"stage_{k}.mp4"
    if record_only:
        writer = _NoWriter()
    else:
        # Warm up the renderer (first frames come back black / unconverged).
        for _ in range(20 if not plan["path_tracing"] else 4):
            grab(frame_offset)
        w, h = plan["resolution"]
        writer = imageio_ffmpeg.write_frames(
            str(video), (w, h), fps=FPS, codec="libx264", quality=None,
            output_params=["-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p"],
            macro_block_size=8,
        )
        writer.send(None)

    def emit(frame_idx):
        img = grab(frame_idx)
        if img is not None:
            writer.send(np.ascontiguousarray(img[..., :3]))
        return img

    frame = frame_offset
    record()
    last = emit(frame)
    frame += 1
    steps = 0
    while steps < plan["max_steps"]:
        action = player.get_action(obs, is_deterministic=True)
        obs, _rew, _dones, _infos = player.env_step(wrapped, action)
        steps += 1
        record()
        last = emit(frame)
        frame += 1
        if "state" in snap:
            break
    if k == len(plan["stages"]) - 1:  # hold the finished assembly
        for _ in range(int(plan["hold_end_s"] * FPS)):
            last = emit(frame)
            frame += 1
    writer.close()
    np.savez_compressed(
        out / f"rollout_{k}.npz",
        body_names=np.array(env.robot.body_names),
        joint_names=np.array(env.robot.joint_names),
        **{key: np.stack(v) for key, v in rec.items()},
    )

    state = snap.get("state") or {
        "joint_pos": env.robot.data.joint_pos[0].tolist(),
        "joint_vel": env.robot.data.joint_vel[0].tolist(),
        "cur_targets": env._cur_targets[0].tolist(),
        "prev_targets": env._prev_targets[0].tolist(),
        "successes": int(env._successes[0]),
        "max_goals": int(env.env_max_goals[0]),
        "retract_succeeded": bool(env.retract_succeeded[0]),
    }
    state["frame_end"] = frame
    state["steps"] = steps
    (out / f"state_{k}.json").write_text(json.dumps(state))
    print(f"[render] stage {k} {st['problem']}: steps={steps} goals={state['successes']}/"
          f"{state['max_goals']} retract={state['retract_succeeded']} -> {video}", flush=True)
    if last is not None:
        import imageio.v3 as iio

        iio.imwrite(out / f"stage_{k}_last.png", last[..., :3])
    env.close()
    app.close()
    return 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def concat(videos: list[Path], dest: Path) -> None:
    import imageio_ffmpeg

    lst = dest.with_suffix(".txt")
    lst.write_text("".join(f"file '{v.resolve()}'\n" for v in videos))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(lst), "-c", "copy", str(dest)], check=True)
    lst.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="flexiv_wuji_right_human_like", choices=sorted(ROBOT_TASKS))
    ap.add_argument("--problems", nargs="+", default=["beam_assembly_step1", "beam_assembly_step2"])
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="one per problem (default: <policies-dir>/<problem>/model.pth)")
    ap.add_argument("--policies-dir", default=None)
    ap.add_argument("--out", default="renders/assembly")
    ap.add_argument("--hole-xy", nargs=2, type=float, default=[0.10, 0.02],
                    help="hole position relative to the table centre (m)")
    ap.add_argument("--starts", type=json.loads,
                    default=[[-0.05, -0.06, 20.0], [-0.06, 0.08, -15.0]],
                    help="JSON list of [x, y, yaw_deg] per step, relative to the table centre")
    ap.add_argument("--camera-eye", nargs=3, type=float, default=[0.85, -1.15, 0.70],
                    help="relative to the hole")
    ap.add_argument("--camera-lookat", nargs=3, type=float, default=[-0.08, 0.18, 0.22],
                    help="relative to the hole")
    ap.add_argument("--orbit-deg", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--hold-end-s", type=float, default=1.5)
    ap.add_argument("--path-tracing", action="store_true")
    ap.add_argument("--insertion-tolerance", type=float, default=None,
                    help="override the problem's insertion tolerance (match the checkpoint's training)")
    ap.add_argument("--record-only", action="store_true",
                    help="no camera: just roll out and save rollout_<k>.npz + plan.json "
                         "(replay with evaluation/blender_render_rollout.py)")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--python", default=sys.executable, help="Isaac Sim python for the stage workers")
    ap.add_argument("--stage", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--plan", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.stage is not None:
        return run_stage(args)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=1))
    videos = []
    for k, st in enumerate(plan["stages"]):
        print(f"[render] stage {k}: {st['problem']} ({st['checkpoint']})", flush=True)
        cmd = [args.python, str(Path(__file__).resolve()), "--stage", str(k),
               "--plan", str(plan_path), "--out", str(out)]
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        state = json.loads((out / f"state_{k}.json").read_text())
        if not state["retract_succeeded"]:
            print(f"[render] WARNING: stage {k} did not finish (goals {state['successes']}/"
                  f"{state['max_goals']}, retract={state['retract_succeeded']})", flush=True)
        videos.append(out / f"stage_{k}.mp4")
    print(f"[render] rollout: {plan_path} + rollout_<k>.npz", flush=True)
    if not plan["record_only"]:
        final = out / "assembly.mp4"
        concat(videos, final)
        print(f"[render] wrote {final}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
