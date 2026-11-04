"""Explicit observation/action timing for frozen-policy evaluation."""

QUEUE_KEYS = (
    "object_state_queue_length",
    "observation_queue_length",
    "action_queue_length",
)


def evaluation_queues(profile, source_randomization):
    """Return immediate feedback or the task's training-time queue lengths.

    Queues sample an entry uniformly each control; lengths are not fixed delays.
    Noise and reset randomization remain controlled by the evaluation preset.
    """
    if profile == "immediate":
        return dict.fromkeys(QUEUE_KEYS, 1)
    if profile != "source":
        raise ValueError(f"Unknown evaluation history profile: {profile}")
    lengths = {key: source_randomization[key] for key in QUEUE_KEYS}
    if any(type(value) is not int or value < 1 for value in lengths.values()):
        raise ValueError("Source queue lengths must be positive integers")
    return lengths
