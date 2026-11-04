"""Installed simulator provenance, including git dependency revisions."""

import importlib.metadata
import json


def simulation_options(env):
    """Read compiled native/Warp options without changing simulation state."""
    native, warp = env.sim.mj_model.opt, env.sim.wp_model.opt
    fields = ("solver", "integrator", "cone", "iterations", "ls_iterations")
    model = env.sim.mj_model
    parts = {}
    for body_id in range(model.nbody):
        name = model.body(body_id).name
        if name in ("object/object", "second/second"):
            parts[name] = {
                "mass_kg": float(model.body_mass[body_id]),
                "inertia_kg_m2": model.body_inertia[body_id].tolist(),
                "center_of_mass_m": model.body_ipos[body_id].tolist(),
            }
    return {
        "native": {name: int(getattr(native, name)) for name in fields},
        "warp": {name: int(getattr(warp, name)) for name in fields},
        "sdf_initpoints": int(native.sdf_initpoints),
        "sdf_iterations": int(native.sdf_iterations),
        "physics_dt": float(native.timestep),
        "impratio": float(native.impratio),
        "decimation": env.cfg.decimation,
        "num_envs": env.num_envs,
        "device": str(env.device),
        "nconmax": env.cfg.sim.nconmax,
        "njmax": env.cfg.sim.njmax,
        "manipulated_parts": parts,
        "writes_simulation_state": False,
    }


def runtime_info():
    result = {}
    for name in (
        "mjlab",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "torch",
        "onnxruntime",
        "rl-games",
    ):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        entry = {"version": distribution.version}
        direct = distribution.read_text("direct_url.json")
        if direct:
            source = json.loads(direct)
            if "vcs_info" in source:
                entry.update(url=source["url"], vcs=source["vcs_info"])
        result[name] = entry
    return result
