"""Process-local EPA scratch capacity for the pinned MuJoCo Warp runtime."""


def configure_epa_horizon(edges=24):
    """Increase scratch storage before creating any simulation or CUDA graph.

    The pinned runtime exposes this allocation size as a module constant,
    rather than a model option. GJK reads the actual array length. No installed
    source file, collision tolerance, or iteration limit is changed.
    """
    from mujoco_warp._src import collision_convex

    if not isinstance(edges, int) or not 24 <= edges <= 256:
        raise ValueError("EPA horizon must contain between24 and256 edges")
    collision_convex.MJ_MAX_EPAHORIZON = edges
    return {
        "epa_horizon_edges": edges,
        "upstream_default_edges": 24,
        "implementation": "process-local collision_convex scratch allocation",
        "scope": "buffer capacity; physical parameters and solver limits unchanged",
    }
