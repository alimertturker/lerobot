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

"""Joint-space control helpers, independent of VLA inference and inverse kinematics."""

import math

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


class JointSafetyLimitError(RuntimeError):
    """Raised when a measured joint crosses a configured hard rollout limit."""


def prepare_replay_image(image: Tensor, size: int = 0) -> Tensor:
    """Own a CPU uint8 frame; preserve its resolution unless a resize is explicitly requested.

    Clone even contiguous decoder frames: their storage can be reused or contain a whole video batch.
    The critic makes its smaller input on demand; replay keeps just this one image per camera.
    """
    height, width = image.shape[-2:]
    if size > 0 and max(height, width) > size:
        shape = (
            max(1, round(height * size / max(height, width))),
            max(1, round(width * size / max(height, width))),
        )
        image = (
            functional.interpolate(image[None].float(), shape, mode="bilinear", align_corners=False)[0]
            .round()
            .clamp(0, 255)
        )
    return image.to(device="cpu", dtype=torch.uint8, copy=True).contiguous()


def compact_observation(
    raw: dict, state_names: list[str], image_keys: list[str], size: int = 0
) -> dict[str, Tensor]:
    """Snapshot state and camera pixels for both VLA inference and replay."""
    result = {OBS_STATE: torch.tensor([raw[name] for name in state_names], dtype=torch.float32)}
    for key in image_keys:
        camera = key.removeprefix(f"{OBS_IMAGES}.")
        frame = torch.from_numpy(np.asarray(raw[camera])).permute(2, 0, 1)
        result[key] = prepare_replay_image(frame, size)
    return result


def limit_joint_target(
    desired: dict[str, float],
    measured: dict[str, float],
    max_step: float,
    bounds: dict[str, list[float]],
) -> dict[str, float]:
    """Cap EVERY policy or human target relative to measured joints; reject NaN before motor writes."""
    result = {}
    for key, value in desired.items():
        current = float(measured[key])
        if not math.isfinite(value) or not math.isfinite(current):
            raise ValueError(f"Nonfinite joint position: {key}")
        calibrated = (0.0, 100.0) if key == "gripper.pos" else (-math.inf, math.inf)
        lower, upper = bounds.get(key, calibrated)
        lower, upper = max(lower, calibrated[0]), min(upper, calibrated[1])
        if key in bounds and not lower <= current <= upper:
            raise JointSafetyLimitError(
                f"Measured joint position {current:.3f} is outside the configured safety bound "
                f"[{lower:.3f}, {upper:.3f}] for {key}; stop and inspect the arm before continuing"
            )
        # Keep every step-limited target inside the configured bound.
        desired_value = min(max(float(value), lower), upper)
        step_limited = current + min(max(desired_value - current, -max_step), max_step)
        result[key] = min(max(step_limited, lower), upper)
    return result
