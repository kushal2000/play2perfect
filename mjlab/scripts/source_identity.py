"""Record checkout identity without importing an alternative learner."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    source_file = ROOT / "SOURCE_COMMIT"
    if source_file.exists():
        return {"commit": source_file.read_text().strip(), "snapshot": True}
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        ),
    }
