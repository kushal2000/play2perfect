"""Stateless batched wrapper for the browser's single-world ONNX actor."""

import numpy as np
import torch

from play2perfect_mjlab.policy import RecurrentActor


class OnnxActor:
    def __init__(self, path, device="cpu"):
        self.session = RecurrentActor(path).session
        self.device = device

    def initial_state(self, count):
        h = torch.zeros(1, count, 1024, device=self.device)
        return h, h.clone()

    def __call__(self, observation, hidden, cell):
        obs, h, c = [v.detach().cpu().numpy() for v in (observation, hidden, cell)]
        results = [
            self.session.run(
                ["mu", "h_out", "c_out"],
                {
                    "obs": obs[i : i + 1],
                    "h_in": h[:, i : i + 1],
                    "c_in": c[:, i : i + 1],
                },
            )
            for i in range(len(obs))
        ]
        values = [
            np.concatenate([r[j] for r in results], axis=0 if j == 0 else 1)
            for j in range(3)
        ]
        if not all(np.isfinite(v).all() for v in values):
            raise FloatingPointError("Nonfinite ONNX policy output")
        return tuple(torch.from_numpy(v).to(observation.device) for v in values)
