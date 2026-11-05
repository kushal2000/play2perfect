"""Finite point-to-solid geometry checks; triangle order and tessellation may differ."""

import numpy as np


def distance_to_solid(points, vertices, faces):
    """Distance to a closed triangle solid; zero for interior points.

    Project onto each triangle plane and its three edges. A fixed oblique ray
    supplies the inside parity. This is an offline diagnostic, not a simulator.
    """
    triangles = np.asarray(vertices, dtype=float)[faces]
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm2 = np.einsum("ij,ij->i", normal, normal)
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    denominator = d00 * d11 - d01**2
    safe = np.where(denominator > 1e-30, denominator, 1)
    ray = np.sqrt([1.0, 2.0, 3.0])
    ray /= np.linalg.norm(ray)
    h = np.cross(ray, ac)
    determinant = np.einsum("ij,ij->i", ab, h)
    inverse = 1 / np.where(np.abs(determinant) > 1e-15, determinant, 1)
    output = []
    for start in range(0, len(points), 64):
        selected = np.asarray(points[start : start + 64], dtype=float)
        delta = selected[:, None, :] - a
        d20 = np.einsum("bti,ti->bt", delta, ab)
        d21 = np.einsum("bti,ti->bt", delta, ac)
        v = (d11 * d20 - d01 * d21) / safe
        w = (d00 * d21 - d01 * d20) / safe
        inside_face = (
            (v >= -1e-12) & (w >= -1e-12) & (v + w <= 1 + 1e-12) & (denominator > 1e-30)
        )
        plane = np.einsum("bti,ti->bt", delta, normal) ** 2 / np.maximum(norm2, 1e-30)
        squared = np.where(inside_face, plane, np.inf)
        for first, second in [(a, b), (b, c), (c, a)]:
            edge = second - first
            local = selected[:, None, :] - first
            parameter = np.einsum("bti,ti->bt", local, edge) / np.maximum(
                np.einsum("ij,ij->i", edge, edge), 1e-30
            )
            nearest = local - np.clip(parameter, 0, 1)[..., None] * edge
            squared = np.minimum(squared, np.einsum("bti,bti->bt", nearest, nearest))
        closest = np.sqrt(squared.min(axis=1))
        u = np.einsum("bti,ti->bt", delta, h) * inverse
        q = np.cross(delta, ab)
        v = np.einsum("bti,i->bt", q, ray) * inverse
        t = np.einsum("bti,ti->bt", q, ac) * inverse
        hit = (
            (np.abs(determinant) > 1e-15)
            & (u >= 0)
            & (v >= 0)
            & (u + v <= 1)
            & (t > 1e-10)
        )
        interior = hit.sum(axis=1) % 2 == 1
        output.extend(np.where(interior, 0, closest))
    return np.asarray(output)


def distance_to_union(points, meshes):
    return np.min([distance_to_solid(points, *mesh) for mesh in meshes], axis=0)
