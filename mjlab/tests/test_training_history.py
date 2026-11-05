"""Keep rollback work charged and reject ambiguous training histories."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plot_training.py"
spec = importlib.util.spec_from_file_location("plot_training", SCRIPT)
plot_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_training)


def session(root, name, steps, times, setup=4, resume=None, worlds=6):
    directory = root / name
    directory.mkdir()
    config = {
        "resume": resume,
        "resume_contract": {
            "num_envs": worlds,
            "horizon": 4,
            "base_checkpoint_sha256": "a" * 64,
        },
        "setup_timing": {"main_to_environment_ready_seconds": setup},
    }
    (directory / "config.json").write_text(json.dumps(config))
    rows = [
        {"transitions": step, "wall_seconds": time}
        for step, time in zip(steps, times, strict=True)
    ]
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return directory


def test_resume_discards_rollback_values_but_charges_all_elapsed_time(tmp_path):
    first = session(tmp_path, "first", [24, 48, 72], [1, 2, 3])
    second = session(
        tmp_path, "second", [72, 96], [1, 2], setup=5, resume="checkpoint48.pth"
    )
    rows, _ = plot_training.read_history([first, second])
    assert [row["transitions"] for row in rows] == [24, 48, 72, 96]
    assert [row["recorded_process_seconds"] for row in rows] == [5, 6, 13, 14]


def test_active_writer_may_leave_only_the_last_row_incomplete(tmp_path):
    directory = session(tmp_path, "first", [24, 48], [1, 2])
    path = directory / "metrics.jsonl"
    path.write_text(path.read_text() + '{"transitions":')
    rows, _ = plot_training.read_history([directory])
    assert len(rows) == 2
    path.write_text(path.read_text() + '\n{"transitions":72,"wall_seconds":3}\n')
    with pytest.raises(ValueError, match="Corrupt metrics row 3"):
        plot_training.read_history([directory])


@pytest.mark.parametrize(
    "steps,times,message",
    [([24, 24], [1, 2], "Transitions"), ([24, 48], [2, 1], "restarted timers")],
)
def test_reject_ambiguous_single_session(tmp_path, steps, times, message):
    directory = session(tmp_path, "first", steps, times)
    with pytest.raises(ValueError, match=message):
        plot_training.read_history([directory])


@pytest.mark.parametrize(
    "steps,worlds,message",
    [([96, 120], 6, "boundary is absent"), ([48, 96], 12, "contracts differ")],
)
def test_reject_incompatible_resume(tmp_path, steps, worlds, message):
    first = session(tmp_path, "first", [24, 48], [1, 2])
    second = session(
        tmp_path, "second", steps, [1, 2], resume="checkpoint.pth", worlds=worlds
    )
    with pytest.raises(ValueError, match=message):
        plot_training.read_history([first, second])
