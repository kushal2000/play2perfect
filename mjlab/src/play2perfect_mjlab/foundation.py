"""Fetch the exact public SimToolReal foundation used by the assembly recipes."""

import argparse
import json
import os
import shutil
from pathlib import Path

from play2perfect_mjlab.bundles import cached_asset, checked_file

REPOSITORY = "Kushal20/play2perfect-checkpoints"
REVISION = "91537ffa9584dbe74548623f2cc88e0922e9cbe3"
FOUNDATION = {
    "url": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/play/model.pth",
    "bytes": 413895044,
    "sha256": "4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9",
}


def prepare_foundation(destination=Path("data/pretrained_policy"), cache=None):
    """Preserve existing weights; verify every downloaded or reused checkpoint."""
    destination = Path(destination)
    target = destination / "model.pth"
    if target.exists() or target.is_symlink():
        checked_file(target, FOUNDATION["sha256"], FOUNDATION["bytes"])
        return target
    cache = Path(cache) if cache else destination.parent / ".asset-cache"
    source = cached_asset(FOUNDATION, cache)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        checked_file(target, FOUNDATION["sha256"], FOUNDATION["bytes"])
    except OSError:
        # Exclusive creation prevents replacing a user's checkpoint.
        try:
            with target.open("xb") as output, source.open("rb") as stream:
                shutil.copyfileobj(stream, output)
        except FileExistsError:
            checked_file(target, FOUNDATION["sha256"], FOUNDATION["bytes"])
    checked_file(target, FOUNDATION["sha256"], FOUNDATION["bytes"])
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path("data/pretrained_policy")
    )
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    path = prepare_foundation(args.destination, args.cache)
    print(json.dumps(dict(checkpoint=str(path), revision=REVISION, **FOUNDATION)))


if __name__ == "__main__":
    main()
