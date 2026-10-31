# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small image-conditioned chunk critics and a bounded stochastic edit policy."""

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.utils.constants import OBS_STATE


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ObservationEncoder(nn.Module):
    def __init__(self, image_keys: list[str], state_dim: int, hidden_dim: int, image_size: int) -> None:
        super().__init__()
        self.image_keys = image_keys
        self.image_size = image_size
        self.images = nn.ModuleDict(
            {
                str(i): nn.Sequential(
                    nn.Conv2d(3, 32, 5, stride=2),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, 3, stride=2),
                    nn.SiLU(),
                    nn.Conv2d(64, 64, 3, stride=2),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d((2, 2)),
                    nn.Flatten(),
                )
                for i in range(len(image_keys))
            }
        )
        self.projection = mlp(256 * len(image_keys) + state_dim, hidden_dim, hidden_dim)

    def forward(self, observation: dict[str, Tensor]) -> Tensor:
        features = [observation[OBS_STATE]]
        for i, key in enumerate(self.image_keys):
            image = functional.interpolate(
                observation[key], (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            features.append(self.images[str(i)](image))
        return self.projection(torch.cat(features, dim=-1))


class ChunkCritics(nn.Module):
    def __init__(
        self,
        image_keys: list[str],
        state_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        image_size: int,
        num_critics: int,
    ) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(image_keys, state_dim, hidden_dim, image_size)
        self.heads = nn.ModuleList(
            [mlp(hidden_dim + horizon * action_dim, 1, hidden_dim) for _ in range(num_critics)]
        )

    def forward(self, observation: dict[str, Tensor], actions: Tensor) -> Tensor:
        return self.score(self.encoder(observation), actions)

    def score(self, features: Tensor, actions: Tensor) -> Tensor:
        inputs = torch.cat([features, actions.flatten(start_dim=1)], dim=-1)
        return torch.cat([head(inputs) for head in self.heads], dim=-1)


class EditPolicy(nn.Module):
    def __init__(
        self, feature_dim: int, action_dim: int, horizon: int, hidden_dim: int, scale: float
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.scale = scale
        self.net = mlp(feature_dim + horizon * action_dim, 2 * horizon * action_dim, hidden_dim)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor, base: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.net(torch.cat([features, base.flatten(start_dim=1)], -1)).chunk(2, -1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp())
        latent = distribution.rsample()
        edit = latent.tanh() * self.scale
        # Stable tanh Jacobian, including the scale change of variables.
        log_prob = distribution.log_prob(latent) - 2 * (
            math.log(2) - latent - functional.softplus(-2 * latent)
        )
        log_prob = (log_prob - math.log(self.scale)).sum(-1)
        return base + edit.view(-1, self.horizon, self.action_dim), log_prob
