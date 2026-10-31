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

"""Episode replay: executed joint targets, human corrections, and chunk TD samples."""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import Tensor


@dataclass
class Episode:
    observations: list[dict[str, Tensor]]  # T+1 raw observations, CHW uint8 cameras.
    actions: Tensor  # T x D, the targets returned by robot.send_action, in physical units.
    rewards: Tensor  # T, success only at the final step.
    interventions: Tensor  # T, preserved for diagnostics and dataset export.
    success: bool
    offline: bool = False
    timing: dict[str, float] | None = None  # Rollout latency diagnostics; absent in older checkpoints.
    storage_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        steps = len(self.actions)
        if steps == 0 or len(self.observations) != steps + 1:
            raise ValueError("An episode needs T actions and T+1 observations")
        if self.actions.ndim != 2 or self.rewards.shape != (steps,) or self.interventions.shape != (steps,):
            raise ValueError("Inconsistent action/reward/intervention shapes")
        if not torch.isfinite(self.actions).all() or not torch.isfinite(self.rewards).all():
            raise ValueError("Nonfinite episode actions or rewards")


class EpisodeReplay:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.episodes: deque[Episode] = deque()
        self.frames = 0
        self.generator = torch.Generator().manual_seed(seed)

    def add(self, episode: Episode) -> None:
        if len(episode.actions) > self.capacity:
            raise ValueError("Episode exceeds replay_capacity; increase capacity or shorten episodes")
        self.episodes.append(episode)
        self.frames += len(episode.actions)
        while self.frames > self.capacity:
            self.frames -= len(self.episodes.popleft().actions)

    def sample(self, batch_size: int, horizon: int, discount: float) -> list[dict[str, Any]]:
        """Sample contiguous chunks without crossing episode boundaries.

        Short terminal chunks repeat the last executed target (absorbing padding) and never bootstrap.
        Discount is per physical control step, including for shortened chunks.
        """
        if not self.frames:
            raise ValueError("Cannot sample empty replay")
        lengths = torch.tensor([len(e.actions) for e in self.episodes])
        cumulative = lengths.cumsum(0)
        samples = []
        for index in torch.randint(self.frames, (batch_size,), generator=self.generator).tolist():
            episode_index = int(torch.searchsorted(cumulative, index, right=True))
            episode = self.episodes[episode_index]
            start = index - (int(cumulative[episode_index - 1]) if episode_index else 0)
            count = min(horizon, len(episode.actions) - start)
            end = start + count
            actions = episode.actions[start:end]
            actions = torch.cat([actions, actions[-1:].expand(horizon - count, -1)])
            reward = (episode.rewards[start:end] * discount ** torch.arange(count)).sum()
            samples.append(
                {
                    "observation": episode.observations[start],
                    "actions": actions,
                    "next_observation": episode.observations[end],
                    "reward": reward,
                    "bootstrap": 0.0 if end == len(episode.actions) else discount**count,
                    "steps": count,
                }
            )
        return samples

    def sample_success(self, batch_size: int, horizon: int) -> list[dict[str, Any]]:
        # Full real horizons only: PI05's native loss does not mask terminal action padding.
        eligible = [e for e in self.episodes if e.success and len(e.actions) >= horizon]
        if not eligible:
            return []
        samples = []
        for _ in range(batch_size):
            episode = eligible[int(torch.randint(len(eligible), (), generator=self.generator))]
            start = int(torch.randint(len(episode.actions) - horizon + 1, (), generator=self.generator))
            samples.append(
                {
                    "observation": episode.observations[start],
                    "actions": episode.actions[start : start + horizon],
                }
            )
        return samples

    def state_dict(self) -> dict[str, Any]:
        # Plain dicts keep torch.load(weights_only=True) usable for local resume files.
        return {"episodes": [vars(e) for e in self.episodes], "generator": self.generator.get_state()}

    def save_shared(self, directory: Path) -> dict[str, Any]:
        """Persist each immutable episode once, retaining a small ordered snapshot manifest."""
        directory.mkdir(parents=True, exist_ok=True)
        ids = []
        for episode in self.episodes:
            storage_id = episode.storage_id
            self._validate_storage_id(storage_id)
            path = directory / f"{storage_id}.pt"
            if not path.exists():
                temporary = directory / f"{storage_id}.{uuid4().hex}.tmp"
                try:
                    torch.save(vars(episode), temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            ids.append(storage_id)
        return {"format": "shared_episodes_v1", "episode_ids": ids, "generator": self.generator.get_state()}

    def load_shared(self, state: dict[str, Any], directory: Path) -> None:
        self.episodes.clear()
        self.frames = 0
        for storage_id in state["episode_ids"]:
            self._validate_storage_id(storage_id)
            path = directory / f"{storage_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Shared replay episode is missing: {path}. Copy the run's replay folder "
                    "along with the checkpoint to resume training."
                )
            self.add(Episode(**torch.load(path, map_location="cpu", weights_only=True)))
        self.generator.set_state(state["generator"])

    @staticmethod
    def _validate_storage_id(storage_id: str) -> None:
        if len(storage_id) != 32 or any(char not in "0123456789abcdef" for char in storage_id):
            raise ValueError("Invalid shared replay episode ID")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.episodes.clear()
        self.frames = 0
        for entry in state["episodes"]:
            self.add(Episode(**entry))
        self.generator.set_state(state["generator"])
