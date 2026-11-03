"""Render a recorded assembly rollout in Blender (Cycles).

Replays what ``evaluation/render_assembly_sequence.py`` recorded (plan.json +
rollout_<k>.npz: per-step world poses of every robot body and of the moving
part) with no simulator in the loop, so camera, lighting, materials, resolution
and timing can be changed and re-rendered freely.

    blender -b -P evaluation/blender_render_rollout.py -- \\
        --rollout renders/beam_assembly --out renders/beam_assembly/blender.mp4

    # one still to check the look (frame index on the full timeline)
    blender -b -P evaluation/blender_render_rollout.py -- \\
        --rollout renders/beam_assembly --still 150 --out still.png

Timeline: one frame per policy step (60 fps), step after step, then a hold on
the finished assembly. Each part sits at its start pose until its step,
follows the recorded pose during it, and stays where it ended afterwards.
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

PART_ROUGHNESS = 0.38
# sRGB. Base (fixture parts), then one colour per inserted part in step order.
BASE_COLOR = (0.88, 0.84, 0.76)  # cream
PART_COLORS = [(0.08, 0.58, 0.56), (0.97, 0.66, 0.14), (0.80, 0.22, 0.30), (0.45, 0.35, 0.70)]
TABLE_COLOR = (0.17, 0.17, 0.18)  # charcoal
ROBOT_ROUGHNESS = 0.32


# ---------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def origin_tf(elem):
    t = np.eye(4)
    if elem is not None:
        t[:3, :3] = rpy_matrix(*[float(v) for v in elem.get("rpy", "0 0 0").split()])
        t[:3, 3] = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    return t


def pose_matrix(pos, quat_wxyz):
    return Matrix.LocRotScale(Vector(pos), Quaternion(quat_wxyz), Vector((1, 1, 1)))


def np_to_matrix(t):
    return Matrix([list(row) for row in np.asarray(t, dtype=float)])


# ---------------------------------------------------------------------------
# scene building
# ---------------------------------------------------------------------------


def material(name, rgb, roughness, metallic=0.0, coat=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    return mat


def srgb_to_linear(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


def import_mesh(path, scale=(1, 1, 1)):
    """Import an OBJ/STL in its raw (Z-up, metres) coordinates; returns the new objects."""
    before = set(bpy.data.objects)
    ext = Path(path).suffix.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    else:
        raise ValueError(f"unsupported mesh {path}")
    objs = [o for o in bpy.data.objects if o not in before]
    for o in objs:
        o.scale = scale
    return objs


def add_box(size):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    o = bpy.context.active_object
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    return [o]


def attach(objs, parent, local_tf, mat=None):
    """Parent ``objs`` to ``parent`` at ``local_tf`` (4x4, parent frame)."""
    for o in objs:
        base = o.matrix_basis.copy()
        o.parent = parent
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = np_to_matrix(local_tf) @ base
        if mat is not None:
            o.data.materials.clear()
            o.data.materials.append(mat)
        o.data.polygons.foreach_set("use_smooth", [False] * len(o.data.polygons))


def build_part(name, visuals, mat):
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    for v in visuals:
        objs = import_mesh(v["mesh"], v.get("scale", (1, 1, 1))) if "mesh" in v else add_box(v["box"])
        attach(objs, root, v["tf"], mat)
    return root


# MuJoCo Menagerie kuka_iiwa_14 (BSD-3-Clause, derived from Drake): multi-part
# visuals in the same link frames as our iiwa14 URDF (checked: identical bounds).
MENAGERIE_IIWA = {
    0: [("link_0", "grey")], 1: [("link_1", "grey")],
    2: [("link_2_orange", "orange"), ("link_2_grey", "grey")],
    3: [("link_3", "grey"), ("band", "band"), ("kuka", "black")],
    4: [("link_4_orange", "orange"), ("link_4_grey", "grey")],
    5: [("link_5", "grey"), ("band", "band"), ("kuka", "black")],
    6: [("link_6_orange", "orange"), ("link_6_grey", "grey")],
    7: [("link_7", "grey")],
}
MENAGERIE_COLORS = {"grey": (0.40, 0.40, 0.41), "orange": (1.0, 0.42, 0.04),
                    "band": (0.62, 0.62, 0.62), "black": (0.02, 0.02, 0.02)}


def menagerie_iiwa_overrides(assets_dir):
    """URDF link name -> [(mesh path, sRGB)] for the iiwa14 links."""
    d = Path(assets_dir)
    return {f"iiwa14_link_{k}": [(d / f"{name}.obj", MENAGERIE_COLORS[c]) for name, c in parts]
            for k, parts in MENAGERIE_IIWA.items()}


def build_robot(urdf_path, body_names, overrides=None):
    """One empty per simulated body; each URDF link's visuals hang off the body
    it was merged into (fixed joints), at the composed fixed-joint offset."""
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    colors = {m.get("name"): m.find("color") for m in root.findall("material")}
    parent_of = {}
    for j in root.findall("joint"):
        parent_of[j.find("child").get("link")] = (
            j.find("parent").get("link"), j.get("type"), origin_tf(j.find("origin")))
    bodies = set(body_names)

    def body_and_offset(link):
        t = np.eye(4)
        while link not in bodies:
            parent, jtype, jt = parent_of[link]
            if jtype != "fixed":
                raise RuntimeError(f"link {link} is not a simulated body and moves")
            t = jt @ t
            link = parent
        return link, t

    empties = {}
    for b in body_names:
        e = bpy.data.objects.new(f"body_{b}", None)
        bpy.context.scene.collection.objects.link(e)
        empties[b] = e
    mats = {}
    for link in root.findall("link"):
        visuals = link.findall("visual")
        if not visuals:
            continue
        body, offset = body_and_offset(link.get("name"))
        if overrides and link.get("name") in overrides:
            for mesh_path, rgb in overrides[link.get("name")]:
                key = ("override",) + tuple(rgb)
                if key not in mats:
                    mats[key] = material(f"robot_{len(mats)}", srgb_to_linear(rgb), ROBOT_ROUGHNESS)
                objs = import_mesh(mesh_path)
                attach(objs, empties[body], offset, mats[key])
            continue
        for vis in visuals:
            geom = vis.find("geometry")
            mesh = geom.find("mesh")
            if mesh is None:
                continue
            path = (urdf_path.parent / mesh.get("filename")).resolve()
            scale = tuple(float(v) for v in mesh.get("scale", "1 1 1").split())
            objs = import_mesh(path, scale)
            mat = None
            mel = vis.find("material")
            has_mtl = path.suffix.lower() == ".obj" and any(
                m is not None for o in objs for m in o.data.materials)
            if has_mtl:
                mel = None  # the mesh's own MTL colours win over the URDF's flat colour
            if mel is not None:
                col = mel.find("color")
                if col is None:
                    col = colors.get(mel.get("name"))
                if col is not None:
                    rgba = tuple(float(v) for v in col.get("rgba").split())
                    key = rgba[:3]
                    if key == (0.2, 1.0, 0.2):  # Sharpa fingertip markers: debug green -> rubber
                        key = (0.16, 0.16, 0.17)
                    if key not in mats:
                        # brighten the near-white plastic a touch, keep accents
                        mats[key] = material(f"robot_{len(mats)}", srgb_to_linear(key), ROBOT_ROUGHNESS)
                    mat = mats[key]
            if mat is None and path.suffix.lower() == ".obj":
                # OBJ visuals bring their own MTL colours; just soften the finish
                for o in objs:
                    for m in o.data.materials:
                        if m and m.use_nodes and "Principled BSDF" in m.node_tree.nodes:
                            bsdf = m.node_tree.nodes["Principled BSDF"]
                            bsdf.inputs["Roughness"].default_value = ROBOT_ROUGHNESS
                            col = bsdf.inputs["Base Color"].default_value
                            if min(col[:3]) > 0.9:  # white covers: off-white so they don't blow out
                                bsdf.inputs["Base Color"].default_value = (0.74, 0.74, 0.75, 1.0)
                            else:  # arm body: the MTL blue-grey (Kd 0.49 0.49 0.53), pushed a bit bluer
                                bsdf.inputs["Base Color"].default_value = (0.24, 0.29, 0.40, 1.0)
            elif mat is None:
                mat = mats.setdefault("default", material("robot_default", (0.85, 0.85, 0.85), ROBOT_ROUGHNESS))
            attach(objs, empties[body], offset @ origin_tf(vis.find("origin")), mat)
    return empties


def keyframe_matrix(obj, frame, m):
    obj.matrix_world = m
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("rotation_quaternion", frame=frame)


def set_linear(obj):
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True, help="directory with plan.json + rollout_<k>.npz")
    ap.add_argument("--out", required=True, help=".mp4 for a video, .png with --still")
    ap.add_argument("--still", type=int, default=None)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--engine", choices=["cycles", "eevee"], default="cycles",
                    help="eevee: fast raster preview (~0.1 s/frame)")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--speed", type=float, default=1.0, help="<1 for slow motion")
    ap.add_argument("--hold-start-s", type=float, default=0.6)
    ap.add_argument("--hold-end-s", type=float, default=None)
    ap.add_argument("--camera-eye", nargs=3, type=float, default=None, help="relative to the hole")
    ap.add_argument("--camera-lookat", nargs=3, type=float, default=None, help="relative to the hole")
    ap.add_argument("--orbit-deg", type=float, default=None)
    ap.add_argument("--lens", type=float, default=40.0)
    ap.add_argument("--push-in", type=float, default=0.85,
                    help="camera distance at the end as a fraction of the start (slow dolly in)")
    ap.add_argument("--part-colors", type=json.loads, default=None,
                    help='JSON list of sRGB triples, one per step (default: teal, amber, ...)')
    ap.add_argument("--base-color", nargs=3, type=float, default=BASE_COLOR, help="sRGB")
    ap.add_argument("--table-size", nargs=2, type=float, default=None,
                    help="table footprint (m); the table runs from the floor up to its top")
    ap.add_argument("--motion-blur", action="store_true")
    ap.add_argument("--frames", nargs=2, type=int, default=None,
                    help="render only this inclusive frame range (to split a render across GPUs)")
    ap.add_argument("--menagerie-iiwa", default=None,
                    help="mujoco_menagerie/kuka_iiwa_14/assets: nicer iiwa14 visuals")
    ap.add_argument("--save-blend", default=None, help="also save the scene as a .blend")
    args = ap.parse_args(argv)

    rdir = Path(args.rollout)
    plan = json.loads((rdir / "plan.json").read_text())
    stages = plan["stages"]
    rolls = [np.load(rdir / f"rollout_{k}.npz") for k in range(len(stages))]
    body_names = [str(b) for b in rolls[0]["body_names"]]
    sim_fps = plan.get("fps", 60)

    # clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = args.fps

    # --- static dressing ----------------------------------------------------
    hole = np.array(plan["hole_pos"])
    hole_tf = np.eye(4)
    hole_tf[:3, 3] = hole
    # Floor at the robot's base; the table is a solid block up to the work surface.
    floor_z = float(rolls[0]["body_pos"][0, 0, 2])
    floor_mat = material("floor", (0.30, 0.31, 0.33), 0.8)
    bpy.ops.mesh.primitive_plane_add(size=80.0, location=(hole[0], hole[1], floor_z))
    bpy.context.active_object.data.materials.append(floor_mat)
    tx, ty = plan["table_xy"]
    height = plan["table_top"] - floor_z
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(tx, ty, floor_z + height / 2))
    table = bpy.context.active_object
    size = args.table_size or plan.get("table_size") or (0.475, 0.4)
    table.scale = (size[0], size[1], height)
    bpy.ops.object.transform_apply(scale=True)
    bev = table.modifiers.new("bevel", "BEVEL")
    bev.width = 0.004
    bev.segments = 3
    table.data.materials.append(material("table", srgb_to_linear(TABLE_COLOR), 0.55))

    part_colors = args.part_colors or PART_COLORS
    base_mat = material("base", srgb_to_linear(tuple(args.base_color)), PART_ROUGHNESS, coat=0.2)
    for j, v in enumerate(plan["base_meshes"]):
        b = build_part(f"base_{j}", [v], base_mat)
        b.matrix_world = np_to_matrix(hole_tf)

    parts = []
    for k, st in enumerate(stages):
        rgb = srgb_to_linear(tuple(part_colors[k % len(part_colors)]))
        mat = material(f"part_{k}", rgb, PART_ROUGHNESS, coat=0.2)
        p = build_part(f"part_{k}", st["visuals"], mat)
        p.rotation_mode = "QUATERNION"
        parts.append(p)

    overrides = menagerie_iiwa_overrides(args.menagerie_iiwa) if args.menagerie_iiwa else None
    empties = build_robot(plan["robot_urdf"], body_names, overrides)
    for e in empties.values():
        e.rotation_mode = "QUATERNION"

    # --- timeline -------------------------------------------------------------
    step_frames = args.fps / sim_fps / args.speed  # output frames per sim step
    hold_start = int(round(args.hold_start_s * args.fps))
    hold_end_s = plan.get("hold_end_s", 1.5) if args.hold_end_s is None else args.hold_end_s
    frame = 1 + hold_start
    stage_frames = []
    for k, r in enumerate(rolls):
        n = len(r["body_pos"])
        frames = [frame + i * step_frames for i in range(n)]
        stage_frames.append(frames)
        frame = frames[-1] + step_frames
    last_frame = int(math.ceil(frame + hold_end_s * args.fps))

    def part_pose(k, r, i):
        return pose_matrix(r["object_pos"][i], r["object_quat"][i])

    for k, (r, frames) in enumerate(zip(rolls, stage_frames)):
        for i, f in enumerate(frames):
            for b, name in enumerate(body_names):
                keyframe_matrix(empties[name], f, pose_matrix(r["body_pos"][i, b], r["body_quat"][i, b]))
            keyframe_matrix(parts[k], f, part_pose(k, r, i))
        # before its step a part rests at its first recorded pose, after it stays put
        keyframe_matrix(parts[k], 1, part_pose(k, r, 0))
        keyframe_matrix(parts[k], last_frame, part_pose(k, r, len(r["body_pos"]) - 1))
    first, last = rolls[0], rolls[-1]
    for b, name in enumerate(body_names):
        keyframe_matrix(empties[name], 1, pose_matrix(first["body_pos"][0, b], first["body_quat"][0, b]))
        keyframe_matrix(empties[name], last_frame, pose_matrix(last["body_pos"][-1, b], last["body_quat"][-1, b]))
    for o in list(empties.values()) + parts:
        set_linear(o)
    # Static between keys: a part's start/end keys use CONSTANT so it does not drift
    for p in parts:
        for fc in p.animation_data.action.fcurves:
            pts = fc.keyframe_points
            if len(pts) > 1:
                pts[0].interpolation = "CONSTANT"
                pts[-2].interpolation = "CONSTANT"
    scene.frame_start = 1
    scene.frame_end = last_frame

    # --- camera -------------------------------------------------------------
    cam_cfg = plan["camera"]
    eye_rel = np.array(args.camera_eye if args.camera_eye else cam_cfg["eye"])
    look_rel = np.array(args.camera_lookat if args.camera_lookat else cam_cfg["lookat"])
    orbit = math.radians(cam_cfg["orbit_deg"] if args.orbit_deg is None else args.orbit_deg)
    target = bpy.data.objects.new("cam_target", None)
    scene.collection.objects.link(target)
    target.location = Vector(hole + look_rel)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = args.lens
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = 20.25  # = a 36 mm-wide 16:9 sensor: same vertical framing at any aspect
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = target
    cam_data.dof.aperture_fstop = 5.6
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    rel = eye_rel - look_rel
    for f in range(1, last_frame + 1, 10):
        u = (f - 1) / max(1, last_frame - 1)
        u_s = u * u * (3 - 2 * u)  # ease in/out
        a = orbit * (u_s - 0.5)
        rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        dist = 1.0 + (args.push_in - 1.0) * u_s
        cam.location = Vector(hole + look_rel + dist * (rot @ rel))
        cam.keyframe_insert("location", frame=f)
    if cam.animation_data:
        for fc in cam.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"

    # --- lights / world -------------------------------------------------------
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.80, 0.83, 0.88, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.22
    key = bpy.data.lights.new("key", "AREA")
    key.energy = 220.0
    key.size = 1.6
    key.color = (1.0, 0.96, 0.91)
    key_obj = bpy.data.objects.new("key", key)
    scene.collection.objects.link(key_obj)
    key_obj.location = Vector(hole + np.array([1.2, -1.0, 1.6]))
    kcon = key_obj.constraints.new("TRACK_TO")
    kcon.target = target
    kcon.track_axis = "TRACK_NEGATIVE_Z"
    kcon.up_axis = "UP_Y"
    rim = bpy.data.lights.new("rim", "AREA")
    rim.energy = 90.0
    rim.size = 1.0
    rim.color = (0.85, 0.9, 1.0)
    rim_obj = bpy.data.objects.new("rim", rim)
    scene.collection.objects.link(rim_obj)
    rim_obj.location = Vector(hole + np.array([-0.9, 1.3, 1.2]))
    rcon = rim_obj.constraints.new("TRACK_TO")
    rcon.target = target
    rcon.track_axis = "TRACK_NEGATIVE_Z"
    rcon.up_axis = "UP_Y"

    # --- render settings ----------------------------------------------------
    scene.render.use_persistent_data = True  # keep the scene/BVH between frames
    if args.engine == "eevee":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        ee = scene.eevee
        ee.taa_render_samples = max(1, min(args.samples, 64))
        if hasattr(ee, "use_raytracing"):
            ee.use_raytracing = True
        if hasattr(ee, "use_shadows"):
            ee.use_shadows = True
    scene.render.engine = "CYCLES" if args.engine == "cycles" else scene.render.engine
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == dev_type]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == dev_type
                scene.cycles.device = "GPU"
                print(f"[blender] rendering on {dev_type}: {[d.name for d in gpus]}", flush=True)
                break
        except TypeError:
            continue
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = args.motion_blur
    scene.render.film_transparent = False

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save_blend).resolve()))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.still is not None:
        scene.frame_set(args.still)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
    elif out.suffix.lower() != ".mp4":  # PNG sequence into a directory
        if args.frames:
            scene.frame_start, scene.frame_end = args.frames
        out.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(out) + "/"
        bpy.ops.render.render(animation=True)
    else:
        if args.frames:
            scene.frame_start, scene.frame_end = args.frames
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.filepath = str(out)
        bpy.ops.render.render(animation=True)
    print(f"[blender] wrote {out} (frames 1-{last_frame})", flush=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
