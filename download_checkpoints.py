"""Download Play2Perfect checkpoints from the Hugging Face Hub.

Downloads the Stage-1 "play" policy and/or the finetuned precise-assembly
policies:

  pretrained_policy/{model.pth,config.yaml}                      # Stage-1 play
  pretrained_assembly/<problem_key>/{model.pth,config.yaml}      # assembly tasks

The ``pretrained_assembly/`` layout is what ``evaluation/eval_isaacsim.py``
discovers via ``--policies-dir pretrained_assembly``.

Usage:
    python download_checkpoints.py                       # everything (play + all tasks)
    python download_checkpoints.py --what play           # just the Stage-1 policy
    python download_checkpoints.py --what assembly       # just the task policies
    python download_checkpoints.py --problems tight_insertion
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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


@dataclass
class DownloadArgs:
    repo_id: str = DEFAULT_REPO_ID
    """Hugging Face model repo holding the checkpoints."""
    what: str = "all"
    """Which checkpoints to fetch: 'all', 'play', or 'assembly'."""
    problems: List[str] = field(default_factory=lambda: list(ALL_PROBLEMS))
    """Subset of assembly problem keys to download (defaults to all)."""


def _snapshot(repo_id: str, subdir: str, into: Path) -> Path:
    from huggingface_hub import snapshot_download

    into.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subdir}/*"],
        local_dir=str(into),
    )
    dest = into / subdir
    if not (dest / "model.pth").is_file():
        raise FileNotFoundError(
            f"Expected {dest / 'model.pth'} after download; check that "
            f"'{subdir}/model.pth' exists in {repo_id}."
        )
    return dest


def run_download(args: DownloadArgs) -> None:
    if args.what not in ("all", "play", "assembly"):
        raise ValueError(f"--what must be all|play|assembly, got {args.what!r}")

    if args.what in ("all", "play"):
        dest = REPO_ROOT / "pretrained_policy"
        if (dest / "model.pth").is_file():
            print(f"[skip] play policy already at {dest}")
        else:
            print(f"[download] Stage-1 play policy <- {args.repo_id}/{PLAY_SUBDIR}")
            got = _snapshot(args.repo_id, PLAY_SUBDIR, REPO_ROOT)
            if got != dest:
                got.rename(dest)
            print(f"  -> {dest}")

    if args.what in ("all", "assembly"):
        out = REPO_ROOT / "pretrained_assembly"
        for problem in args.problems:
            if (out / problem / "model.pth").is_file():
                print(f"[skip] {problem}: already at {out / problem}")
                continue
            print(f"[download] {problem} <- {args.repo_id}")
            _snapshot(args.repo_id, problem, out)
            print(f"  -> {out / problem}")

    print("DONE!")


def main() -> None:
    run_download(tyro.cli(DownloadArgs))


if __name__ == "__main__":
    main()
