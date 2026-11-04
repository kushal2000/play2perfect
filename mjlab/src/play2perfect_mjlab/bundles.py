"""Prepare the exact tested assembly scenes from versioned public inputs."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path

from play2perfect_mjlab.assets import sha256


def resources():
    return files("play2perfect_mjlab").joinpath("resources", "bundles")


def catalog():
    return json.loads(resources().joinpath("catalog.json").read_text())


def checked_file(path, expected, size=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular asset file: {path}")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Asset size differs: {path}")
    if sha256(path) != expected:
        raise ValueError(f"Asset checksum differs: {path}")


def cached_asset(record, cache):
    target = cache / record["sha256"]
    if target.exists():
        checked_file(target, record["sha256"], record["bytes"])
        return target
    cache.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="asset-", dir=cache)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            if "bundled" in record:
                source = resources().joinpath(record["bundled"])
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
            else:
                with urllib.request.urlopen(record["url"], timeout=60) as stream:
                    count = 0
                    while chunk := stream.read(1024 * 1024):
                        count += len(chunk)
                        if count > record["bytes"]:
                            raise ValueError("Downloaded asset exceeds declared size")
                        output.write(chunk)
        checked_file(temporary, record["sha256"], record["bytes"])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def verify_bundle(name, root):
    definition = catalog()["bundles"][name]
    for filename, expected in definition["metadata"].items():
        checked_file(root / filename, expected)
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, expected in definition["asset_hashes"].items():
        checked_file(root / filename, expected, manifest["files"][filename]["bytes"])
    return root


def prepare_bundle(name, destination=Path("data"), cache=None):
    """Download once by hash; leave existing scenes unchanged and verified."""
    index = catalog()
    if name not in index["bundles"]:
        raise ValueError(f"Unknown assembly bundle: {name}")
    definition = index["bundles"][name]
    destination = Path(destination)
    cache = Path(cache) if cache else destination / ".asset-cache"
    target = destination / definition["destination"]
    if target.exists():
        return verify_bundle(name, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=target.parent))
    try:
        for filename, expected in definition["metadata"].items():
            content = resources().joinpath(name, filename).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Packaged metadata differs: {name}/{filename}")
            (temporary / filename).write_bytes(content)
        for filename, expected in definition["asset_hashes"].items():
            if Path(filename).name != filename:
                raise ValueError("Asset names must be flat filenames")
            source = cached_asset(index["assets"][expected], cache)
            try:
                os.link(source, temporary / filename)
            except OSError:
                shutil.copyfile(source, temporary / filename)
        verify_bundle(name, temporary)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *catalog()["bundles"]], default="all")
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    names = catalog()["bundles"] if args.task == "all" else [args.task]
    for name in names:
        path = prepare_bundle(name, args.destination, args.cache)
        print(json.dumps(dict(task=name, bundle=str(path), hashes_verified=True)))


if __name__ == "__main__":
    main()
