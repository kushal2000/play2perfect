"""Render saved MuJoCo states at presentation resolution without changing motion."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from OpenGL import GL
from PIL import Image

FIELDS = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "eq_active")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure_visuals(model, style):
    changes = []
    for i in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[i])).name or ""
        name = model.geom(i).name or ""
        if body.split("/")[-1] == "goal_object":
            model.geom_rgba[i, 3] = 0
            changes.append(dict(geom=i, change="Hide goal ghost"))
        elif style == "native":
            continue
        elif "floor" in name and int(model.geom_type[i]) == 0:
            model.geom_matid[i] = -1
            model.geom_rgba[i] = [0.67, 0.69, 0.72, 1]
            changes.append(dict(geom=i, change="Matte gray floor; no checker texture"))
        elif name == "table/table_geom":
            top = model.geom_pos[i, 2] + model.geom_size[i, 2]
            model.geom_rgba[i] = [0.105, 0.125, 0.15, 1]
            # Extend only the rendered pedestal downward. Preserve its top surface.
            model.geom_size[i, 2] += 0.115
            model.geom_pos[i, 2] -= 0.115
            model.geom_sameframe[i] = 0
            assert abs(model.geom_pos[i, 2] + model.geom_size[i, 2] - top) < 1e-12
            changes.append(dict(geom=i, change="Graphite pedestal; same top surface"))
        elif body.startswith("assembly_fixture/"):
            model.geom_rgba[i] = [0.72, 0.76, 0.8, 1]
        elif body.startswith("object/"):
            model.geom_rgba[i] = [0.93, 0.49, 0.12, 1]
    model.vis.quality.offsamples = 4
    model.vis.quality.shadowsize = 4096
    model.vis.quality.numslices = 64
    model.vis.quality.numstacks = 32
    if style == "studio":
        model.vis.headlight.ambient[:] = 0.3
        model.vis.headlight.diffuse[:] = 0.2
        model.vis.headlight.specular[:] = 0.12
        model.vis.rgba.fog[:] = [0.67, 0.69, 0.72, 1]
    return changes


def lights(scene):
    scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    for position, direction, diffuse, shadow in [
        ([1.4, -1.4, 2.8], [-0.4, 0.4, -1], [0.4, 0.38, 0.35], True),
        ([-1.5, 0.2, 2.0], [0.5, 0.1, -1], [0.14, 0.16, 0.19], False),
        ([0, 2, 2.5], [0, -0.7, -1], [0.15, 0.17, 0.2], False),
    ]:
        light = scene.lights[scene.nlight]
        light.pos[:] = position
        light.dir[:] = direction
        light.ambient[:] = 0
        light.diffuse[:] = diffuse
        light.specular[:] = [0.2, 0.2, 0.2]
        light.attenuation[:] = [1, 0, 0]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.castshadow = shadow
        light.headlight = False
        light.cutoff = 45
        light.exponent = 1
        scene.nlight += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--azimuth", type=float, default=35)
    parser.add_argument("--elevation", type=float, default=-22)
    parser.add_argument("--distance", type=float, default=1.8)
    parser.add_argument("--lookat", nargs=3, type=float, default=[0, 0.3, 0.62])
    parser.add_argument("--orthographic", action="store_true")
    parser.add_argument("--style", choices=["native", "studio"], default="native")
    parser.add_argument("--preview-controls", nargs="+", type=int)
    args = parser.parse_args()
    original_hashes = dict(model=digest(args.model), recording=digest(args.recording))
    with np.load(args.recording) as archive:
        states = {key: archive[key] for key in archive.files}
    valid = np.flatnonzero(states["active_worlds"][:, args.world])
    _, reverse = np.unique(states["control"][valid][::-1], return_index=True)
    valid = valid[np.sort(len(valid) - 1 - reverse)]
    controls = states["control"][valid]
    assert controls[0] == 0 and np.all(np.diff(controls) == 1)
    assert all(np.isfinite(value).all() for value in states.values())
    model = mujoco.MjModel.from_binary_path(str(args.model))
    arm = []
    for joint in range(7):
        assert model.joint(joint).name == f"robot/joint{joint + 1}"
        assert any(int(a[0]) == joint for a in model.actuator_trnid)
        span = float(np.ptp(states["qpos"][valid, args.world, joint]))
        arm.append(span)
    changes = configure_visuals(model, args.style)
    model.vis.global_.offwidth, model.vis.global_.offheight = args.width, args.height
    model.vis.global_.fovy = 38
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = args.lookat
    camera.distance, camera.azimuth, camera.elevation = (
        args.distance,
        args.azimuth,
        args.elevation,
    )
    camera.orthographic = args.orthographic
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[[0, 1, 2]] = 1
    args.output.mkdir(parents=True, exist_ok=False)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    graphics = {
        name: GL.glGetString(enum).decode()
        for name, enum in [
            ("vendor", GL.GL_VENDOR),
            ("renderer", GL.GL_RENDERER),
            ("version", GL.GL_VERSION),
        ]
    }

    def frame(index):
        for field in FIELDS:
            value = states[field][index, args.world]
            assert value.shape == getattr(data, field).shape
            getattr(data, field)[:] = value
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for field in FIELDS:
            assert np.array_equal(
                getattr(data, field), states[field][index, args.world]
            )
        renderer.update_scene(data, camera, scene_option=option)
        if args.style == "studio":
            lights(renderer.scene)
        return renderer.render()

    encoder = None
    selected = sorted(
        {
            int(valid[np.abs(controls - c).argmin()])
            for c in (args.preview_controls or np.linspace(0, controls[-1], 12))
        }
    )
    try:
        if not args.preview_controls:
            command = [
                "ffmpeg",
                "-y",
                "-v",
                "warning",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{args.width}x{args.height}",
                "-r",
                "60",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "slow",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output / "rollout.mp4"),
            ]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        for index in selected if args.preview_controls else valid:
            pixels = frame(index)
            if encoder:
                encoder.stdin.write(pixels.tobytes())
            if index in selected:
                Image.fromarray(pixels).save(
                    args.output / f"control_{states['control'][index]:04d}.png"
                )
        if encoder:
            for _ in range(60):
                encoder.stdin.write(pixels.tobytes())
    finally:
        renderer.close()
        if encoder:
            encoder.stdin.close()
            assert encoder.wait(timeout=60) == 0
    assert original_hashes == dict(
        model=digest(args.model), recording=digest(args.recording)
    )
    report = dict(
        inputs=original_hashes,
        script_sha256=digest(__file__),
        world=args.world,
        mujoco_version=mujoco.__version__,
        graphics=graphics,
        source_control_hz=60,
        video_fps=60,
        controls=int(controls[-1]),
        recorded_motion_seconds=float(controls[-1] / 60),
        frozen_endpoint_hold_seconds=1,
        integration_steps=0,
        interpolation=False,
        states_unchanged=True,
        appearance_only=changes,
        visual_style=args.style,
        full_seven_joint_arm_ranges_rad=arm,
        camera=dict(
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
            orthographic=args.orthographic,
            fovy=38,
        ),
        resolution=[args.width, args.height],
        preview_only=bool(args.preview_controls),
        rendered_video_sha256=None
        if args.preview_controls
        else digest(args.output / "rollout.mp4"),
        visual_review_pending=True,
    )
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
