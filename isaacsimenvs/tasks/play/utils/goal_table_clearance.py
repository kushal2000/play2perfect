"""Keep sampled goal poses clear of the table (numpy + torch, no Isaac Lab).

A goal is an object pose, so whether it intersects the table depends on the object's shape and
the goal orientation, not on the goal position alone. Each object's collision geometry is bounded
by the corners of one box per shape (handle, head) in the object's frame; for a goal orientation,
the lowest rotated corner says how far the object reaches below its origin, and the goal z is
raised until that point is ``clearance`` above the table top.

Boxes are bounded exactly. Cylinders are bounded as the capsules the URDF importer turns them into
(``replace_cylinders_with_capsules=True``: length + 2 r along the axis), by the capsule's bounding
box -- conservative by at most a few mm at oblique angles.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

#: Corners per object: two shapes (handle, head) x 8. A handle-only object repeats its handle's.
NUM_CORNERS = 16


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF rpy: fixed-axis x, then y, then z."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def collision_corners(urdf_path: str | Path) -> np.ndarray:
    """(NUM_CORNERS, 3) bounding-box corners of every collision shape, in the root link's frame.

    Handles the procedural tools (generate_objects.py): one link, box and/or cylinder collisions.
    """
    root = ET.parse(urdf_path).getroot()
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError(f"{urdf_path}: expected a single-link object, got {len(links)} links")
    corners = []
    for col in links[0].findall("collision"):
        origin = col.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]) if origin is not None else np.zeros(3)
        rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
        geom = col.find("geometry")
        box, cyl = geom.find("box"), geom.find("cylinder")
        if box is not None:
            half = np.array([float(v) for v in box.get("size").split()]) / 2
        elif cyl is not None:
            r, length = float(cyl.get("radius")), float(cyl.get("length"))
            half = np.array([r, r, length / 2 + r])  # the capsule the importer makes, axis z
        else:
            raise ValueError(f"{urdf_path}: unsupported collision geometry {[c.tag for c in geom]}")
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
        corners.append(xyz + (signs * half) @ _rpy_matrix(rpy).T)
    if not 1 <= len(corners) <= NUM_CORNERS // 8:
        raise ValueError(f"{urdf_path}: expected 1-{NUM_CORNERS // 8} collision shapes, got {len(corners)}")
    while len(corners) < NUM_CORNERS // 8:
        corners.append(corners[0])
    return np.concatenate(corners).astype(np.float32)


def _rotate(quat_wxyz: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate (N, K, 3) points by (N, 4) wxyz quaternions."""
    w, xyz = quat_wxyz[:, None, :1], quat_wxyz[:, None, 1:]
    t = 2.0 * torch.cross(xyz.expand_as(points), points, dim=-1)
    return points + w * t + torch.cross(xyz.expand_as(points), t, dim=-1)


def min_goal_z(
    quat_wxyz: torch.Tensor,     # (N, 4) goal orientation
    corners: torch.Tensor,       # (N, K, 3) collision_corners of each env's object
    table_top_z: torch.Tensor,   # (N,) table top, env coordinates
    clearance: float,
) -> torch.Tensor:
    """(N,) lowest goal z at which the object, at that orientation, is ``clearance`` above the table."""
    lowest = _rotate(quat_wxyz, corners)[..., 2].min(dim=-1).values  # relative to the origin
    return table_top_z + clearance - lowest


__all__ = ["NUM_CORNERS", "collision_corners", "min_goal_z"]
