"""Recurrent actor inference and the browser's float32 target processing."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


def action_targets(previous, action, ranges):
    previous = np.asarray(previous, dtype=np.float32)
    action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
    lower, upper = np.asarray(ranges, dtype=np.float32).T
    raw = lower + np.float32(0.5) * (action + 1) * (upper - lower)
    raw[:7] = np.clip(
        previous[:7] + np.float32(0.025) * action[:7], lower[:7], upper[:7]
    )
    return np.clip(np.float32(0.1) * raw + np.float32(0.9) * previous, lower, upper)


class RecurrentActor:
    def __init__(self, path: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 1024), dtype=np.float32)
        self.c = np.zeros_like(self.h)

    def __call__(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, 141)
        mu, self.h, self.c = self.session.run(
            ["mu", "h_out", "c_out"],
            {"obs": obs, "h_in": self.h, "c_in": self.c},
        )
        if not all(np.isfinite(x).all() for x in (mu, self.h, self.c)):
            raise FloatingPointError("Nonfinite recurrent actor output")
        return mu.reshape(29)
