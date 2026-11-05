"""Check the offline surface metric against analytic cube distances."""

import itertools
from pathlib import Path
from runpy import run_path

import numpy as np

SURFACE = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/collision_surface.py")
)
distance_to_solid = SURFACE["distance_to_solid"]
distance_to_union = SURFACE["distance_to_union"]

VERTICES = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
FACES = np.array(
    [
        [0, 1, 3],
        [0, 3, 2],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [2, 3, 7],
        [2, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [1, 5, 7],
        [1, 7, 3],
    ]
)


def test_closed_cube_matches_analytic_distance():
    points = np.array([[0, 0, 0], [0, 0.2, 0.4], [1, 0, 0], [2, 0, 0], [2, 2, 2]])
    np.testing.assert_allclose(
        distance_to_solid(points, VERTICES, FACES), [0, 0, 0, 1, np.sqrt(3)], atol=1e-12
    )


def test_tessellation_and_union_do_not_create_false_gaps():
    centers = VERTICES[FACES].mean(axis=1)
    split_vertices = np.concatenate([VERTICES, centers])
    split_faces = np.array(
        [
            [a, b, 8 + i]
            for i, face in enumerate(FACES)
            for a, b in zip(face, np.roll(face, -1), strict=True)
        ]
    )
    np.testing.assert_allclose(
        distance_to_solid(centers, VERTICES, FACES), 0, atol=1e-12
    )
    np.testing.assert_allclose(
        distance_to_solid(VERTICES, split_vertices, split_faces), 0, atol=1e-12
    )
    points = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]])
    np.testing.assert_allclose(
        distance_to_union(points, [(VERTICES, FACES), (VERTICES + [3, 0, 0], FACES)]),
        [0, 0, 2],
        atol=1e-12,
    )
