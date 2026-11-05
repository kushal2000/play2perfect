"""Evaluation timing must be deliberate and remain separate from training."""

import pytest

from play2perfect_mjlab.evaluation_history import QUEUE_KEYS, evaluation_queues


def test_source_history_preserves_training_lengths_without_mutating_input():
    source = dict(zip(QUEUE_KEYS, [10, 3, 3])) | {"other": "unchanged"}
    assert list(evaluation_queues("source", source).values()) == [10, 3, 3]
    assert source == dict(zip(QUEUE_KEYS, [10, 3, 3])) | {"other": "unchanged"}


def test_immediate_history_retains_old_evaluation_contract():
    assert evaluation_queues("immediate", {}) == dict.fromkeys(QUEUE_KEYS, 1)


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_reject_invalid_source_lengths(length):
    with pytest.raises(ValueError, match="positive integers"):
        evaluation_queues("source", dict.fromkeys(QUEUE_KEYS, length))


def test_reject_unknown_profile():
    with pytest.raises(ValueError, match="Unknown"):
        evaluation_queues("typo", {})


def test_config_changes_only_queues_and_keeps_training_unchanged():
    from pathlib import Path

    from play2perfect_mjlab.assembly_env_cfg import assembly_cfg

    bundle = Path(__file__).resolve().parents[1] / "data/long_leg/long_leg"
    if not bundle.exists():
        pytest.skip("Run p2p-mjlab-assets before compiled-scene checks")
    immediate = assembly_cfg(bundle, evaluation_preset="public_offline")
    source = assembly_cfg(
        bundle, evaluation_preset="public_offline", evaluation_history="source"
    )
    expected = dict(immediate.source_randomization)
    expected.update(zip(QUEUE_KEYS, [10, 3, 3]))
    assert source.source_randomization == expected
    assert source.p2p_metadata == immediate.p2p_metadata
    training = assembly_cfg(bundle, training_preset="isaac_lab_long_leg")
    assert [training.source_randomization[key] for key in QUEUE_KEYS] == [10, 3, 3]
    with pytest.raises(ValueError, match="requires an evaluation"):
        assembly_cfg(bundle, evaluation_history="source")
