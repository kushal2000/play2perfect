"""Run a historical recipe; remaining arguments go to train_sapg.py.

Run from the mjlab directory. For example:
python scripts/train_recipe.py --recipe long_leg_fine_v1 --run-dir runs/long
Use train_sapg.py --help for resource, checkpoint, and resume options.
Explicit extra arguments override the recipe defaults and enter the run record.
"""

import argparse
import os
import sys
from pathlib import Path

from play2perfect_mjlab.recipes import RECIPES, training_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    trainer = Path(__file__).with_name("train_sapg.py")
    command = [
        sys.executable,
        "-u",
        str(trainer),
        *training_arguments(args.recipe),
        *extra,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
