"""状态输入的最小 Behavior Cloning MLP policy。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class BCPolicy(nn.Module):
    """将标准化 observation 映射为标准化 action。"""

    def __init__(
        self,
        observation_dim: int = 3,
        action_dim: int = 1,
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim 和 action_dim 必须大于 0")
        if not hidden_dims or any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("hidden_dims 必须包含正整数")

        layers: list[nn.Module] = []
        input_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """保持 batch 维度，将 ``[..., 3]`` 映射为 ``[..., 1]``。"""
        return self.network(observation)
