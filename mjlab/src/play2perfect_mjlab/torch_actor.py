"""Trainable SAPG actor with the original normalization and recurrent layout.

Adapted from Play2Perfect's MIT native exporter. Keep normalization statistics
fixed for the initial port; changing them changes a pretrained policy's inputs.
"""

from pathlib import Path

import torch
from torch import nn


class PretrainedActor(nn.Module):
    def __init__(self, weights, block_id=50.0):
        super().__init__()
        block_ids = torch.linspace(50, 0, 6)
        if block_id not in block_ids.tolist():
            raise ValueError(f"Unknown SAPG block: {block_id}")
        self.block_id = float(block_id)
        self.block_index = block_ids.tolist().index(block_id)
        self.register_buffer("mean", weights["running_mean_std.running_mean"].float())
        self.register_buffer(
            "variance", weights["running_mean_std.running_var"].float()
        )
        self.observation_dim = self.mean.numel()
        self.embedding = nn.Parameter(
            weights["a2c_network.extra_params"].float().clone()
        )
        self.rnn = nn.LSTM(self.observation_dim + self.embedding.shape[-1], 1024)
        self.layer_norm = nn.LayerNorm(1024)
        sizes = [1024, 1024, 1024, 512, 512]
        self.mlp = nn.Sequential(
            *[
                layer
                for index in range(4)
                for layer in (nn.Linear(sizes[index], sizes[index + 1]), nn.ELU())
            ]
        )
        self.mu = nn.Linear(512, 29)
        self.log_std = nn.Parameter(
            weights["a2c_network.sigma"][self.block_index].float().clone()
        )
        for module, prefix in (
            (self.rnn, "a2c_network.rnn.rnn."),
            (self.layer_norm, "a2c_network.layer_norm."),
            (self.mlp, "a2c_network.actor_mlp."),
            (self.mu, "a2c_network.mu."),
        ):
            module.load_state_dict(
                {
                    key.removeprefix(prefix): value
                    for key, value in weights.items()
                    if key.startswith(prefix)
                },
                strict=True,
            )

    @classmethod
    def from_checkpoint(cls, path: Path, block_id=50.0):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if 0 in checkpoint:
            checkpoint = checkpoint[0]
        return cls(checkpoint["model"], block_id=block_id)

    def initial_state(self, num_envs):
        h = self.mean.new_zeros((1, num_envs, 1024))
        return h, torch.zeros_like(h)

    def forward(self, observation, h, c):
        if observation.shape[-1] not in (
            self.observation_dim,
            self.observation_dim + 1,
        ):
            raise ValueError(
                f"Expected {self.observation_dim} observations, with optional SAPG id"
            )
        normalized = torch.clamp(
            (observation[..., : self.observation_dim] - self.mean)
            / torch.sqrt(self.variance + 1e-5),
            -5,
            5,
        )
        embedding = self.embedding[self.block_index].expand(normalized.shape[0], -1)
        features = torch.cat((normalized, embedding), dim=-1)
        recurrent, (next_h, next_c) = self.rnn(features.unsqueeze(0), (h, c))
        mean = self.mu(self.mlp(self.layer_norm(recurrent.squeeze(0))))
        return mean, next_h, next_c

    def distribution(self, observation, h, c):
        mean, h, c = self(observation, h, c)
        return torch.distributions.Normal(mean, self.log_std.exp()), h, c


class PretrainedCritic(nn.Module):
    """The checkpoint's asymmetric feed-forward value network (162 inputs)."""

    def __init__(self, weights, block_id=50.0):
        super().__init__()
        index = torch.linspace(50, 0, 6).tolist().index(block_id)
        for name, key in (
            ("mean", "model.running_mean_std.running_mean"),
            ("variance", "model.running_mean_std.running_var"),
            ("value_mean", "model.value_mean_std.running_mean"),
            ("value_variance", "model.value_mean_std.running_var"),
        ):
            self.register_buffer(name, weights[key].float())
        self.embedding = nn.Parameter(
            weights["model.a2c_network.extra_params"][index].float().clone()
        )
        self.observation_dim = self.mean.numel()
        sizes = [self.observation_dim + self.embedding.numel(), 1024, 1024, 512, 512]
        self.mlp = nn.Sequential(
            *[
                layer
                for i in range(4)
                for layer in (nn.Linear(sizes[i], sizes[i + 1]), nn.ELU())
            ]
        )
        self.value = nn.Linear(512, 1)
        for module, prefix in (
            (self.mlp, "model.a2c_network.actor_mlp."),
            (self.value, "model.a2c_network.value."),
        ):
            module.load_state_dict(
                {
                    key.removeprefix(prefix): value
                    for key, value in weights.items()
                    if key.startswith(prefix)
                },
                strict=True,
            )

    @classmethod
    def from_checkpoint(cls, path: Path, block_id=50.0):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if 0 in checkpoint:
            checkpoint = checkpoint[0]
        return cls(checkpoint["assymetric_vf_nets"], block_id)

    def forward(self, observation, *, source_inference=False):
        """Apply the source's inference-only normalized-value clamp on request.

        The early PPO prototype uses an unclamped differentiable value in both
        collection and updates. rl_games clamps normalized inference values to
        [-5,5] before denormalizing; its training values stay unclamped.
        """
        normalized = ((observation - self.mean) / (self.variance + 1e-5).sqrt()).clamp(
            -5, 5
        )
        features = torch.cat(
            (normalized, self.embedding.expand(normalized.shape[0], -1)), -1
        )
        output = self.value(self.mlp(features)).squeeze(-1)
        if source_inference:
            output = output.clamp(-5, 5)
        return output * (self.value_variance + 1e-5).sqrt() + self.value_mean
