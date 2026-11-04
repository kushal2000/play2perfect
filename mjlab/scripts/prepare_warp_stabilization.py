"""Build the verified SDF-plus-stabilization source checkout on local disk."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "mujoco_warp_230_free_body_stabilization"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--destination", type=Path, default=ROOT / ".cache/mujoco_warp_sdf_gyro"
    )
    args = p.parse_args()
    target = args.destination.resolve()
    if not target.is_relative_to(ROOT / ".cache"):
        p.error("Use the local repository .cache directory")
    patch = ROOT / "patches" / (STEM + ".patch")
    manifest = ROOT / "patches" / (STEM + ".json")
    record = json.loads(manifest.read_text())
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == record["patch_sha256"]

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(target), *arguments], text=True, timeout=60
        )

    if not (target / ".git").exists():
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(target)], check=True, timeout=10)
        git("remote", "add", "origin", "https://github.com/thowell/mujoco_warp.git")
        git(
            "fetch",
            "--filter=blob:none",
            "--depth",
            "1",
            "origin",
            record["base_commit"],
        )
        git("checkout", "--detach", record["base_commit"])
    assert git("rev-parse", "HEAD").strip() == record["base_commit"]
    if not git("diff", "HEAD", "--name-only").strip():
        git("apply", "--check", str(patch))
        git("apply", str(patch))
    for name, expected in record["files_sha256"].items():
        path = target / name
        assert not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    print(
        json.dumps(
            dict(
                source_root=str(target),
                verified_python_files=len(record["files_sha256"]),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                installed_packages_modified=False,
                passed=True,
            )
        )
    )


if __name__ == "__main__":
    main()
