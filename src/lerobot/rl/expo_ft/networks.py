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

"""Image-conditioned chunk critics and a bounded stochastic edit policy."""

# Reference encoder adapted from https://github.com/pd-perry/expo-ft (MIT License).
# Copyright (c) 2026 The EXPO-FT Authors
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        self.output_dim = hidden_dim
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


def _same_pad(x: Tensor, kernel_size: int, stride: int, value: float = 0.0) -> Tensor:
    """Flax/JAX SAME padding (asymmetric for even-sized, strided inputs)."""
    height, width = x.shape[-2:]
    pad_h = max((math.ceil(height / stride) - 1) * stride + kernel_size - height, 0)
    pad_w = max((math.ceil(width / stride) - 1) * stride + kernel_size - width, 0)
    return functional.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)


class ResNetV2Block(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.stride = stride
        self.norm1 = nn.GroupNorm(4, in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, bias=False)
        self.norm2 = nn.GroupNorm(4, channels, eps=1e-5)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False)
            if in_channels != channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(_same_pad(functional.relu(self.norm1(x)), 3, self.stride))
        y = self.conv2(functional.relu(self.norm2(y)))
        return self.shortcut(x) + y


class ReferenceObservationEncoder(nn.Module):
    """PyTorch port of the released EXPO-FT ResNetV2 + pixel/state projections.

    Reference: github.com/pd-perry/expo-ft, networks/encoders.py and pixel_multiplexer.py.
    The paper calls this ResNet-50; the code uses two-convolution blocks (3, 4, 6, 3),
    not torchvision's bottleneck ResNet-50. Cameras are channel-stacked in a stable order.
    No pretrained weights or downloads are required.
    """

    def __init__(self, image_keys: list[str], state_dim: int, image_size: int) -> None:
        super().__init__()
        if image_size != 224:
            raise ValueError("The reference critic encoder requires critic_image_size=224")
        self.image_keys = sorted(image_keys)
        self.image_size = image_size
        self.output_dim = 512 + 64
        self.stem = nn.Conv2d(3 * len(image_keys), 64, 7, stride=2, padding=3, bias=False)
        blocks = []
        in_channels = 64
        for stage, count in enumerate((3, 4, 6, 3)):
            channels = 64 * 2**stage
            for block in range(count):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(ResNetV2Block(in_channels, channels, stride))
                in_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(4, 512, eps=1e-5)
        self.image_projection = nn.Sequential(
            nn.Linear(7 * 7 * 512, 512), nn.LayerNorm(512, eps=1e-6), nn.Tanh()
        )
        self.state_projection = nn.Sequential(nn.Linear(state_dim, 64), nn.LayerNorm(64, eps=1e-6), nn.Tanh())
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observation: dict[str, Tensor]) -> Tensor:
        images = [
            functional.interpolate(
                observation[key], (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            for key in self.image_keys
        ]
        x = self.stem(torch.cat(images, dim=1))
        x = functional.max_pool2d(_same_pad(x, 3, 2, value=-math.inf), 3, stride=2)
        x = functional.relu(self.norm(self.blocks(x)))
        # Flax flattens NHWC; preserve that layout for the learned spatial projection.
        x = self.image_projection(x.permute(0, 2, 3, 1).reshape(x.shape[0], -1))
        return torch.cat([x, self.state_projection(observation[OBS_STATE])], dim=-1)


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
        encoder_type: str = "compact",
    ) -> None:
        super().__init__()
        if encoder_type == "compact":
            self.encoder = ObservationEncoder(image_keys, state_dim, hidden_dim, image_size)
        elif encoder_type == "resnet":
            self.encoder = ReferenceObservationEncoder(image_keys, state_dim, image_size)
        else:
            raise ValueError(f"Unknown critic_encoder: {encoder_type}")
        self.feature_dim = self.encoder.output_dim
        self.heads = nn.ModuleList(
            [mlp(self.feature_dim + horizon * action_dim, 1, hidden_dim) for _ in range(num_critics)]
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
