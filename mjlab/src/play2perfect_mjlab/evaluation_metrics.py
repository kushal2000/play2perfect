"""Contact diagnostics from actual terminal Warp states, before auto-reset."""

from pathlib import Path

import numpy as np

from play2perfect_mjlab.assets import sha256


def declared_start_metrics(goals, withdrawn, valid, total_goals, contact_free=None):
    """Keep early failures and unfinished starts in the reporting denominator."""
    goals, withdrawn, valid = (
        np.asarray(goals),
        np.asarray(withdrawn, bool),
        np.asarray(valid, bool),
    )
    count = len(goals)
    if count < 1 or withdrawn.shape != goals.shape or valid.shape != goals.shape:
        raise ValueError("Expected one outcome per declared start")
    all_goals = int(((goals >= total_goals) & valid).sum())
    releases = int((withdrawn & valid).sum())
    result = {
        "num_starts": count,
        "all_goals_hit_count": all_goals,
        "all_goals_hit_rate": all_goals / count,
        "withdrawal_count": releases,
        "withdrawal_rate": releases / count,
        "scope": "All declared starts in denominator; early/unfinished episodes "
        "receive no completion credit. Legacy valid-episode rates remain separate.",
    }
    if contact_free is not None:
        clean = np.asarray(contact_free, bool)
        if clean.shape != goals.shape:
            raise ValueError("Expected one contact result per declared start")
        clean_count = int((clean & withdrawn & valid).sum())
        result.update(
            contact_free_withdrawal_count=clean_count,
            contact_free_withdrawal_rate=clean_count / count,
        )
    return result


def terminal_release_metrics(recording, withdrawn, controls, valid):
    """Require source withdrawal and zero penetrating robot/object contacts.

    This supplements the source's fingertip-distance proxy. It does not change
    episode termination, physics, or the metrics used for Isaac curve matching.
    Missing/unfinished terminal states are never credited as releases.
    """
    withdrawn, valid = np.asarray(withdrawn, bool), np.asarray(valid, bool)
    controls = np.asarray(controls)
    counts = np.full(len(withdrawn), -1, dtype=int)
    with np.load(recording) as states:
        for world in range(len(withdrawn)):
            terminal = np.flatnonzero(states["terminal_worlds"][:, world])
            if not len(terminal):
                continue
            frame = terminal[0]
            if int(states["control"][frame]) != controls[world]:
                raise ValueError(
                    "Recording terminal control differs from episode result"
                )
            counts[world] = states["robot_object_penetrating_contacts"][frame, world]
    released = withdrawn & valid & (counts == 0)
    return {
        "terminal_penetrating_robot_object_contacts": [
            int(count) if count >= 0 else None for count in counts
        ],
        "contact_free_withdrawn": released.tolist(),
        "contact_free_withdrawal_rate": float(released[valid].mean())
        if valid.any()
        else None,
        "release_metric_scope": (
            "Source withdrawal proxy plus zero penetrating robot/object contacts "
            "in recorded terminal Warp state before auto-reset; valid first episodes"
        ),
        "release_recording_sha256": sha256(Path(recording)),
        "release_postprocessor_sha256": sha256(Path(__file__)),
    }
