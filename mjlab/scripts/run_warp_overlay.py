"""Run an evaluation with a verified, isolated Warp source overlay.

Installed packages remain unchanged. The result's runtime record distinguishes
the installed distribution from the effective imported source and patch.
"""

import argparse
import runpy
import sys
from pathlib import Path

from warp_source import activate_source_overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.script.name.startswith("train"):
        p.error("This experimental overlay currently supports evaluation and checks")
    activate_source_overlay(args.source_root)
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    sys.argv = [str(args.script), *arguments]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
