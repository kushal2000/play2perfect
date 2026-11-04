"""Activate the recorded Warp backport before any simulator import."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "patches/mujoco_warp_230_free_body_stabilization.json"


def activate_source_overlay(source_root: Path) -> dict:
    if "mujoco_warp" in sys.modules:
        raise RuntimeError("Activate the source overlay before importing the simulator")
    source = source_root.resolve()
    record = json.loads(MANIFEST.read_text())
    for name, expected in record["files_sha256"].items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f"Overlay source escapes its checkout: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Overlay source hash differs: {name}")
    import mujoco
    import warp

    if (mujoco.__version__, warp.__version__) != ("3.12.0", "1.18.0"):
        raise ValueError("This backport requires MuJoCo3.12.0 and Warp-lang1.18.0")
    sys.path.insert(0, str(source))
    import mujoco_warp

    if not Path(mujoco_warp.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Warp did not import from the verified source checkout")
    from play2perfect_mjlab import runtime_info

    original = runtime_info.runtime_info
    proof = dict(
        name=record["name"],
        source_root=str(source),
        module_path=mujoco_warp.__file__,
        base_commit=record["base_commit"],
        upstream_commits=record["upstream_commits"],
        patch_sha256=record["patch_sha256"],
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        verified_python_files=len(record["files_sha256"]),
        mujoco_version=mujoco.__version__,
        warp_lang_version=warp.__version__,
        installed_packages_modified=False,
    )

    def with_effective_source():
        result = original()
        installed = result["mujoco-warp"]
        result["mujoco-warp"] = dict(
            version=installed["version"],
            installed_distribution=installed,
            effective_source=proof,
        )
        return result

    runtime_info.runtime_info = with_effective_source
    print(json.dumps(dict(effective_warp_source=proof)), flush=True)
    return proof


def training_contract(proof: dict) -> dict:
    """Exclude checkout paths so an identical patch can resume on another host."""
    return {
        key: proof[key]
        for key in ["manifest_sha256", "mujoco_version", "warp_lang_version"]
    }
