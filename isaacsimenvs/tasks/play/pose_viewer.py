"""Pose-only interactive HTML capture for Play training.

This module deliberately does not use Isaac cameras, RTX sensors, Replicator, or
the Isaac viewport.  It samples one environment's state tensors and writes a
Three.js/URDF HTML viewer that can be opened locally or logged to WandB.
"""

from __future__ import annotations

import time
import urllib.request
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gymnasium as gym
import numpy as np

from isaacsimenvs.utils.interactive_viewer import create_html, make_embedded_robot, make_url_robot

from .utils.scene_utils import JOINT_NAMES_CANONICAL


REPO_ROOT = Path(__file__).resolve().parents[3]
FALLBACK_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/kushal2000/depthbasedRL/main/"
ROBOT_URDF_RELATIVE_PATH = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
TABLE_URDF_PATH = REPO_ROOT / "assets" / "urdf" / "table_narrow.urdf"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _quat_wxyz_to_xyzw(quat) -> np.ndarray:
    quat_np = _to_numpy(quat)
    return quat_np[[1, 2, 3, 0]]


def _pose_xyzw(pos, quat_wxyz) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = _to_numpy(pos)
    pose[3:] = _quat_wxyz_to_xyzw(quat_wxyz)
    return pose


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _github_slug_from_remote(remote_url: str | None) -> str | None:
    if not remote_url:
        return None

    remote = remote_url.strip()
    if remote.endswith(".git"):
        remote = remote[:-4]

    prefixes = (
        "git@github.com:",
        "ssh://git@github.com/",
        "https://github.com/",
        "http://github.com/",
    )
    for prefix in prefixes:
        if remote.startswith(prefix):
            slug = remote[len(prefix):]
            parts = slug.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
    return None


def _derive_github_raw_base() -> str:
    remote_url = _git_output(["remote", "get-url", "origin"])
    slug = _github_slug_from_remote(remote_url)
    if slug is None:
        print(
            f"[pose_viewer] Could not derive GitHub origin from {remote_url!r}; "
            f"falling back to {FALLBACK_GITHUB_RAW_BASE}",
            flush=True,
        )
        return FALLBACK_GITHUB_RAW_BASE

    ref = _git_output(["branch", "--show-current"])
    if not ref:
        ref = _git_output(["rev-parse", "HEAD"])
    if not ref:
        print(
            f"[pose_viewer] Could not derive git branch/commit; falling back to {FALLBACK_GITHUB_RAW_BASE}",
            flush=True,
        )
        return FALLBACK_GITHUB_RAW_BASE

    raw_base = f"https://raw.githubusercontent.com/{slug}/{quote(ref, safe='')}/"
    print(f"[pose_viewer] GitHub raw base: {raw_base}", flush=True)
    return raw_base


def _normalize_raw_base(github_raw_base: str | None) -> str:
    base = github_raw_base or _derive_github_raw_base()
    return base if base.endswith("/") else base + "/"


def _check_url(url: str, url_check: str) -> None:
    if url_check == "skip":
        return
    print(f"[pose_viewer] URL check ({url_check}) -> {url}", flush=True)
    start = time.monotonic()
    try:
        request = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(request, timeout=10)
        print(f"[pose_viewer]   PASSED ({time.monotonic() - start:.2f}s)", flush=True)
    except Exception as exc:
        message = f"[pose_viewer]   FAILED ({time.monotonic() - start:.2f}s): {exc}"
        if url_check == "error":
            raise RuntimeError(message) from exc
        print(message, flush=True)


def _raw_url_for_repo_path(path: Path, raw_base: str) -> str | None:
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    return raw_base + quote(rel.as_posix(), safe="/")


def _rewrite_embedded_urdf_mesh_urls(
    urdf_text: str,
    *,
    source_urdf_path: Path,
    raw_base: str,
) -> str:
    """Make relative mesh filenames loadable from a standalone HTML document."""

    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        return urdf_text

    changed = False
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if not filename:
            continue
        if filename.startswith(("http://", "https://", "data:")):
            continue

        if filename.startswith("package://"):
            rel_name = filename[len("package://") :]
            mesh_path = REPO_ROOT / rel_name
        else:
            candidate = Path(filename)
            if candidate.is_absolute():
                mesh_path = candidate
            elif filename.startswith("assets/"):
                mesh_path = REPO_ROOT / filename
            else:
                mesh_path = source_urdf_path.parent / filename

        mesh_url = _raw_url_for_repo_path(mesh_path, raw_base)
        if mesh_url is None:
            continue
        mesh_elem.set("filename", mesh_url)
        changed = True

    if not changed:
        return urdf_text
    return ET.tostring(root, encoding="unicode")


def object_urdf_for_env(env, env_id: int) -> tuple[str, Path]:
    """Return the procedural object URDF text assigned to one env."""

    urdf_paths = getattr(env, "_object_urdf_paths", None)
    asset_indices = getattr(env, "_object_asset_index_per_env", None)
    if not urdf_paths or asset_indices is None:
        raise RuntimeError(
            "Play env does not expose object URDF mapping. "
            "Expected _object_urdf_paths and _object_asset_index_per_env."
        )

    asset_index = int(asset_indices[env_id].detach().cpu().item())
    urdf_path = Path(urdf_paths[asset_index])
    return urdf_path.read_text(encoding="utf-8"), urdf_path


def object_urdf_text_for_env(env, env_id: int) -> str:
    return object_urdf_for_env(env, env_id)[0]


def table_urdf_for_env(env, env_id: int) -> tuple[str, Path]:
    """Return the table URDF text assigned to one env: its scaled variant if the task made them,
    else the task's own ``assets.table_urdf``."""

    table_paths = getattr(env, "_table_urdf_paths", None)
    if table_paths:
        urdf_path = Path(table_paths[env_id % len(table_paths)])
        return urdf_path.read_text(encoding="utf-8"), urdf_path
    cfg_path = getattr(getattr(getattr(env, "cfg", None), "assets", None), "table_urdf", None)
    urdf_path = TABLE_URDF_PATH
    if cfg_path:
        urdf_path = Path(cfg_path)
        if not urdf_path.is_absolute():
            urdf_path = REPO_ROOT / urdf_path
    return urdf_path.read_text(encoding="utf-8"), urdf_path


def table_urdf_text_for_env(env, env_id: int) -> str:
    return table_urdf_for_env(env, env_id)[0]


def hole_urdf_for_env(env, env_id: int) -> tuple[str, Path] | tuple[None, None]:
    """Return the peg-in-hole receptive URDF text when the env exposes one."""

    hole_paths = getattr(env, "_hole_urdf_paths", None)
    if not hole_paths:
        return None, None
    urdf_path = Path(hole_paths[env_id % len(hole_paths)])
    return urdf_path.read_text(encoding="utf-8"), urdf_path


def capture_pose_viewer_frame(env, env_id: int) -> dict[str, Any]:
    """Capture one env-local frame from a live PlayEnv."""

    if env_id < 0 or env_id >= env.num_envs:
        raise ValueError(f"capture_viewer_env_id={env_id} out of range for num_envs={env.num_envs}")

    origin = env.scene.env_origins[env_id]

    if hasattr(env, "_perm_lab_to_canon"):
        joint_pos = env.robot.data.joint_pos[env_id, env._perm_lab_to_canon]
        spec = getattr(env, "robot_spec", None)
        joint_names = list(spec.joint_names_canonical if spec else JOINT_NAMES_CANONICAL)
    else:
        joint_pos = env.robot.data.joint_pos[env_id]
        joint_names = list(env.robot.data.joint_names)

    robot_root_pos = env.robot.data.root_pos_w[env_id] - origin
    object_pos = env.object.data.root_pos_w[env_id] - origin
    goal_pos = env.goal_viz.data.root_pos_w[env_id] - origin
    table_pos = env.table.data.root_pos_w[env_id] - origin
    hole = getattr(env, "hole", None)

    frame = {
        "env_id": int(env_id),
        "robot_joint_names": joint_names,
        "robot_joint_pos": _to_numpy(joint_pos),
        "robot_base_pose": _pose_xyzw(robot_root_pos, env.robot.data.root_quat_w[env_id]),
        "object_pose": _pose_xyzw(object_pos, env.object.data.root_quat_w[env_id]),
        "goal_pose": _pose_xyzw(goal_pos, env.goal_viz.data.root_quat_w[env_id]),
        "table_pose": _pose_xyzw(table_pos, env.table.data.root_quat_w[env_id]),
    }
    if hole is not None:
        hole_pos = hole.data.root_pos_w[env_id] - origin
        frame["hole_pose"] = _pose_xyzw(hole_pos, hole.data.root_quat_w[env_id])
    return frame


def build_pose_viewer_html(
    *,
    frames: list[dict[str, Any]],
    object_urdf_text: str,
    table_urdf_text: str,
    hole_urdf_text: str | None = None,
    object_urdf_path: Path | None = None,
    table_urdf_path: Path | None = None,
    hole_urdf_path: Path | None = None,
    github_raw_base: str | None = None,
    url_check: str = "skip",
    robot_urdf_relative_path: str = ROBOT_URDF_RELATIVE_PATH,
) -> str:
    """Build a self-contained-ish viewer HTML string from captured frames.

    Object and table URDFs are embedded.  The robot URDF is URL-backed because
    its hand references mesh files that the browser must fetch, so
    ``robot_urdf_relative_path`` (repo-relative) and its meshes must be pushed.
    """

    if not frames:
        raise ValueError("Cannot build pose viewer from zero frames.")

    raw_base = _normalize_raw_base(github_raw_base)
    robot_urdf_url = raw_base + robot_urdf_relative_path
    _check_url(robot_urdf_url, url_check)
    if object_urdf_path is not None:
        object_urdf_text = _rewrite_embedded_urdf_mesh_urls(
            object_urdf_text,
            source_urdf_path=object_urdf_path,
            raw_base=raw_base,
        )
    if table_urdf_path is not None:
        table_urdf_text = _rewrite_embedded_urdf_mesh_urls(
            table_urdf_text,
            source_urdf_path=table_urdf_path,
            raw_base=raw_base,
        )
    if hole_urdf_text is not None and hole_urdf_path is not None:
        hole_urdf_text = _rewrite_embedded_urdf_mesh_urls(
            hole_urdf_text,
            source_urdf_path=hole_urdf_path,
            raw_base=raw_base,
        )

    timestamps = np.arange(len(frames), dtype=np.float32) / 60.0
    robots = [
        make_url_robot(name="robot", urdf_url=robot_urdf_url, animated=True),
        make_embedded_robot(name="table", urdf_text=table_urdf_text),
        make_embedded_robot(name="object", urdf_text=object_urdf_text),
        make_embedded_robot(
            name="goal",
            urdf_text=object_urdf_text,
            color_override=(0.20, 0.72, 0.31),
        ),
    ]
    object_poses = {
        "table": np.stack([frame["table_pose"] for frame in frames]),
        "object": np.stack([frame["object_pose"] for frame in frames]),
        "goal": np.stack([frame["goal_pose"] for frame in frames]),
    }
    if hole_urdf_text is not None and all("hole_pose" in frame for frame in frames):
        robots.insert(2, make_embedded_robot(name="hole", urdf_text=hole_urdf_text))
        object_poses["hole"] = np.stack([frame["hole_pose"] for frame in frames])

    return create_html(
        joint_names=frames[0]["robot_joint_names"],
        robot_joint_positions=np.stack([frame["robot_joint_pos"] for frame in frames]),
        robots=robots,
        object_poses=object_poses,
        robot_base_poses=np.stack([frame["robot_base_pose"] for frame in frames]),
        timestamps=timestamps,
    )


class PlayPoseViewerWrapper(gym.Wrapper):
    """Gym wrapper that periodically writes pose-only interactive HTML rollouts."""

    def __init__(
        self,
        env: gym.Env,
        *,
        output_dir: str | Path,
        capture_len: int,
        capture_interval: int,
        env_id: int = 0,
        wandb_key: str = "interactive_viewer",
        github_raw_base: str | None = None,
        url_check: str = "skip",
    ) -> None:
        super().__init__(env)
        if capture_len <= 0:
            raise ValueError(f"capture_viewer_len must be > 0, got {capture_len}")
        if url_check not in {"skip", "warn", "error"}:
            raise ValueError(f"capture_viewer_url_check must be skip/warn/error, got {url_check}")

        inner = self.env.unwrapped
        if env_id < 0 or env_id >= inner.num_envs:
            raise ValueError(f"capture_viewer_env_id={env_id} out of range for num_envs={inner.num_envs}")

        self.output_dir = Path(output_dir)
        self.capture_len = int(capture_len)
        self.capture_interval = int(capture_interval)
        self.env_id = int(env_id)
        self.wandb_key = wandb_key
        self.github_raw_base = github_raw_base
        self.url_check = url_check
        self._object_urdf_text, self._object_urdf_path = object_urdf_for_env(
            inner, self.env_id
        )
        self._table_urdf_text, self._table_urdf_path = table_urdf_for_env(
            inner, self.env_id
        )
        self._hole_urdf_text, self._hole_urdf_path = hole_urdf_for_env(inner, self.env_id)

        self._step = 0
        self._capture_index = 0
        self._frames: list[dict[str, Any]] | None = []
        # Per-capture buffers of (C, H, W) student-camera frames (numpy, [0, 1] floats).
        # _depth_frames is the policy's actual input (noise-on path).
        # _depth_frames_clean is the same view without depth augmentation,
        # useful for A/B-comparing the noise pipeline on the same camera pose.
        # Both stay empty when the env doesn't expose `get_student_obs`.
        self._depth_frames: list = []
        self._depth_frames_clean: list = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[pose_viewer] enabled: "
            f"env_id={self.env_id} len={self.capture_len} interval={self.capture_interval} "
            f"output_dir={self.output_dir}",
            flush=True,
        )

    def step(self, action):
        result = self.env.step(action)
        self._step += 1

        if self._frames is None and self.capture_interval > 0 and self._step % self.capture_interval == 0:
            self._frames = []
            self._depth_frames = []
            self._depth_frames_clean = []

        if self._frames is not None:
            self._frames.append(capture_pose_viewer_frame(self.env.unwrapped, self.env_id))
            depth_frame = self._capture_student_image()
            if depth_frame is not None:
                self._depth_frames.append(depth_frame)
            clean_frame = self._capture_student_image_clean()
            if clean_frame is not None:
                self._depth_frames_clean.append(clean_frame)
            if len(self._frames) >= self.capture_len:
                self._finalize_capture()

        return result

    def _capture_student_image(self):
        """Pull the env_id slice of the student's input image, if the env exposes one.

        Returns a (C, H, W) numpy array in [0, 1] (depth after env-side
        ``window_normalize`` preproc), or ``None`` when student_obs isn't
        configured. Called once per pose-viewer-captured step, so cost is
        capped at one extra `get_student_obs()` per (capture_len * frequency).
        """
        inner = self.env.unwrapped
        if not hasattr(inner, "get_student_obs"):
            return None
        try:
            student_obs = inner.get_student_obs()
        except Exception as exc:
            print(f"[pose_viewer] get_student_obs failed: {exc}", flush=True)
            return None
        image = student_obs.get("image") if isinstance(student_obs, dict) else None
        if image is None:
            return None
        return image[self.env_id].detach().cpu().numpy()

    def _capture_student_image_clean(self):
        """Same env_id slice as the noisy capture, but with depth noise bypassed.

        Reads the post-preprocess, post-crop tensor stashed by
        `scene_utils.read_student_camera_image` so the two videos differ
        ONLY in whether `_apply_depth_noise` was applied. When
        `use_depth_aug=False` this equals the noisy capture.

        Returns a (1, H, W) numpy array in [0, 1], or `None` when the env
        hasn't run a depth read yet.
        """
        inner = self.env.unwrapped
        clean = getattr(inner, "_last_student_image_clean", None)
        if clean is None:
            return None
        return clean[self.env_id].detach().cpu().numpy()

    def close(self) -> None:
        if self._frames:
            self._finalize_capture(partial=True)
        return self.env.close()

    def _finalize_capture(self, *, partial: bool = False) -> None:
        assert self._frames is not None
        frames = self._frames
        if not frames:
            self._frames = None
            return

        suffix = "partial" if partial else f"step_{self._step:09d}"
        html_path = self.output_dir / f"pose_viewer_{suffix}_{self._capture_index:04d}.html"
        html_text = build_pose_viewer_html(
            frames=frames,
            object_urdf_text=self._object_urdf_text,
            table_urdf_text=self._table_urdf_text,
            hole_urdf_text=self._hole_urdf_text,
            object_urdf_path=self._object_urdf_path,
            table_urdf_path=self._table_urdf_path,
            hole_urdf_path=self._hole_urdf_path,
            github_raw_base=self.github_raw_base,
            url_check=self.url_check,
            robot_urdf_relative_path=self.env.unwrapped.cfg.assets.robot_urdf,
        )
        html_path.write_text(html_text, encoding="utf-8")
        print(f"[pose_viewer] wrote {len(frames)} frames to {html_path}", flush=True)
        self._log_wandb(html_text)

        self._capture_index += 1
        self._frames = None
        self._depth_frames = []
        self._depth_frames_clean = []

    def _log_wandb(self, html_text: str) -> None:
        try:
            import wandb
        except Exception:
            return

        if wandb.run is None:
            return

        try:
            wandb.log({self.wandb_key: wandb.Html(html_text)})
            print(f"[pose_viewer] logged WandB Html key={self.wandb_key} step={self._step}", flush=True)
        except Exception as exc:
            print(f"[pose_viewer] WandB log failed: {exc}", flush=True)

        if not self._depth_frames and not self._depth_frames_clean:
            return

        import numpy as np

        def _to_uint8_grayscale_rgb(frames: list) -> np.ndarray:
            video = np.stack(frames, axis=0)                        # (T, C, H, W) float in [0, 1]
            video = (np.clip(video, 0.0, 1.0) * 255.0).astype(np.uint8)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, axis=1)                 # (T, 3, H, W) for wandb.Video single-video tiling
            return video

        for label, frames in (("depth", self._depth_frames), ("depth_clean", self._depth_frames_clean)):
            if not frames:
                continue
            try:
                video = _to_uint8_grayscale_rgb(frames)
                key = f"{self.wandb_key}_{label}"
                wandb.log({key: wandb.Video(video, fps=30, format="mp4")})
                print(f"[pose_viewer] logged WandB Video key={key} "
                      f"shape={video.shape} dtype={video.dtype}", flush=True)
            except Exception as exc:
                print(f"[pose_viewer] WandB {label} video log failed: {exc}", flush=True)
