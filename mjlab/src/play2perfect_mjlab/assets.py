"""Stage immutable browser bundles without changing their physics assets."""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

WEBSITE_COMMIT = "a61913d2720a16e9062114555823fc4af7c4ed6c"
WEBSITE_REPO = "play2perfect/play2perfect.github.io"
TASKS = ("tight_insertion", "fabrica", "screwing")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_bundle(task: str, destination: Path, website: Path | None = None) -> Path:
    """Flatten shared assets as the browser's virtual filesystem does.

    A local source must be the pinned clean website checkout. Remote downloads
    always use its immutable commit. Policies and meshes are checksum-verified.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if website:
        commit = subprocess.check_output(
            ["git", "-C", str(website), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(website), "status", "--porcelain", "--", "interactive"],
            text=True,
        )
        if commit != WEBSITE_COMMIT or dirty:
            raise ValueError("Local website must match the clean pinned commit")
    root = destination / task
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"interactive/assets/{task}/"
    base = f"https://raw.githubusercontent.com/{WEBSITE_REPO}/{WEBSITE_COMMIT}/"

    def fetch(relative: str, target: Path) -> None:
        if website:
            shutil.copyfile(website / relative, target)
        else:
            url = urllib.parse.urljoin(base, relative)
            with urllib.request.urlopen(url, timeout=60) as response:
                with target.open("wb") as out:
                    shutil.copyfileobj(response, out)

    fetch(prefix + "manifest.json", root / "manifest.json")
    metadata = json.loads((root / "manifest.json").read_text())
    fetch(prefix + "scene.xml", root / "scene.xml")
    for name, expected in metadata["files"].items():
        source = metadata.get("asset_sources", {}).get(name, name)
        # Resolve ../shared references relative to the task directory.
        relative = str(Path(prefix, source))
        fetch(relative, root / name)
        if sha256(root / name) != expected["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {name}")
    for policy in metadata["policy_provenance"]:
        path = root / policy["policy"]
        fetch(prefix + policy["policy"], path)
        if sha256(path) != policy["onnx_sha256"]:
            raise ValueError(f"Policy checksum mismatch: {path}")
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "website_repo": WEBSITE_REPO,
                "website_commit": WEBSITE_COMMIT,
                "manifest_sha256": sha256(root / "manifest.json"),
                "scene_sha256": sha256(root / "scene.xml"),
                "source": str(website) if website else base,
            },
            indent=2,
        )
        + "\n"
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--destination", type=Path, default=Path("data/browser"))
    parser.add_argument("--website", type=Path)
    args = parser.parse_args()
    for task in TASKS if args.task == "all" else (args.task,):
        print(stage_bundle(task, args.destination, args.website))


if __name__ == "__main__":
    main()
