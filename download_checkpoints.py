"""Download Play2Perfect checkpoints from the Hugging Face Hub.

Downloads the Stage-1 "play" policy and/or the finetuned precise-assembly
policies:

  pretrained_policy/{model.pth,config.yaml}                      # Stage-1 play
  pretrained_assembly/<problem_key>/{model.pth,config.yaml}      # assembly tasks

for the Kuka + Sharpa (``--robot kuka_sharpa``, the default), or
``pretrained_policy_flexiv_wuji/`` and ``pretrained_assembly_flexiv_wuji/<problem_key>/``
for the Flexiv Rizon 4s + right Wuji Hand 2 (``--robot flexiv_wuji_right_human_like``).
These are the layouts ``evaluation/eval_isaacsim.py`` discovers.

Usage:
    python download_checkpoints.py                       # everything (play + all tasks)
    python download_checkpoints.py --what play           # just the Stage-1 policy
    python download_checkpoints.py --what assembly       # just the task policies
    python download_checkpoints.py --problems tight_insertion
    python download_checkpoints.py --robot flexiv_wuji_right_human_like
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tyro

DEFAULT_REPO_ID = "Kushal20/play2perfect-checkpoints"
REPO_ROOT = Path(__file__).parent

# Subfolder in the HF repo that holds the Stage-1 play policy.
PLAY_SUBDIR = "play"

# The released assembly tasks (keys into evaluation.problems.PROBLEM_REGISTRY).
ALL_PROBLEMS = [
    "tight_insertion",
    "beam_assembly_step1",
    "beam_assembly_step2",
    "screwing",
    "fork_insertion",
    "plug_insertion",
]


@dataclass(frozen=True)
class RobotLayout:
    hub_prefix: str  # folder in the HF repo ("" = repo root)
    play_dir: str  # local destination of the play policy
    assembly_dir: str  # local parent of the per-problem policies
    problems: tuple  # problems released for this robot


ROBOTS = {
    "kuka_sharpa": RobotLayout("", "pretrained_policy", "pretrained_assembly", tuple(ALL_PROBLEMS)),
    "flexiv_wuji_right_human_like": RobotLayout(
        "flexiv_wuji_right_human_like/",
        "pretrained_policy_flexiv_wuji",
        "pretrained_assembly_flexiv_wuji",
        tuple(p for p in ALL_PROBLEMS if p != "screwing"),  # no Flexiv screwing policy yet
    ),
}


@dataclass
class DownloadArgs:
    repo_id: str = DEFAULT_REPO_ID
    """Hugging Face model repo holding the checkpoints."""
    robot: str = "kuka_sharpa"
    """Robot whose checkpoints to fetch: kuka_sharpa or flexiv_wuji_right_human_like."""
    what: str = "all"
    """Which checkpoints to fetch: 'all', 'play', or 'assembly'."""
    problems: Optional[List[str]] = None
    """Subset of assembly problem keys to download (defaults to all released for the robot)."""


def _fetch(repo_id: str, subdir: str, dest: Path) -> None:
    """Download ``<subdir>/*`` from the hub into ``dest`` (which then holds model.pth)."""
    from huggingface_hub import snapshot_download

    staging = dest.parent / ".hf_download"
    snapshot_download(repo_id=repo_id, allow_patterns=[f"{subdir}/*"], local_dir=str(staging))
    got = staging / subdir
    if not (got / "model.pth").is_file():
        raise FileNotFoundError(
            f"Expected '{subdir}/model.pth' in {repo_id}; nothing was downloaded to {got}."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(got), str(dest))
    shutil.rmtree(staging)


def run_download(args: DownloadArgs) -> None:
    if args.what not in ("all", "play", "assembly"):
        raise ValueError(f"--what must be all|play|assembly, got {args.what!r}")
    if args.robot not in ROBOTS:
        raise ValueError(f"--robot must be one of {sorted(ROBOTS)}, got {args.robot!r}")
    layout = ROBOTS[args.robot]
    problems = layout.problems if args.problems is None else args.problems
    unreleased = sorted(set(problems) - set(layout.problems))
    if unreleased:
        raise ValueError(f"no {args.robot} checkpoints released for {unreleased}")

    if args.what in ("all", "play"):
        dest = REPO_ROOT / layout.play_dir
        if (dest / "model.pth").is_file():
            print(f"[skip] play policy already at {dest}")
        else:
            subdir = layout.hub_prefix + PLAY_SUBDIR
            print(f"[download] Stage-1 play policy <- {args.repo_id}/{subdir}")
            _fetch(args.repo_id, subdir, dest)
            print(f"  -> {dest}")

    if args.what in ("all", "assembly"):
        out = REPO_ROOT / layout.assembly_dir
        for problem in problems:
            if (out / problem / "model.pth").is_file():
                print(f"[skip] {problem}: already at {out / problem}")
                continue
            subdir = layout.hub_prefix + problem
            print(f"[download] {problem} <- {args.repo_id}/{subdir}")
            _fetch(args.repo_id, subdir, out / problem)
            print(f"  -> {out / problem}")

    print("DONE!")


def main() -> None:
    run_download(tyro.cli(DownloadArgs))


if __name__ == "__main__":
    main()
