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

"""PyTorch EXPO-FT: chunk TD, bounded edits, candidate selection, and native VLA flow loss.

Algorithm reference: https://arxiv.org/abs/2605.25477 (equations 1–5).
This SO-arm adaptation uses joint targets and a small CNN instead of the reference ResNet-50.
"""

import json
import os
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration import ExpoFTConfig
from .networks import ChunkCritics, EditPolicy
from .replay import EpisodeReplay


class ExpoFTLearner:
    def __init__(
        self,
        cfg: ExpoFTConfig,
        policy: nn.Module,
        preprocessor: Any,
        postprocessor: Any,
        image_keys: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        # Replay and hardware use dataset names; policy/critics see checkpoint names after preprocessing.
        # Only actual cameras participate in the critic, even if the VLA declares unused camera slots.
        self.image_keys = list(image_keys if image_keys is not None else policy.config.image_features)
        policy_image_keys = [cfg.rename_map.get(key, key) for key in self.image_keys]
        if not self.image_keys or not set(policy_image_keys) <= set(policy.config.image_features):
            raise ValueError("Dataset camera names do not map to the pretrained VLA's cameras")
        if len(set(policy_image_keys)) != len(policy_image_keys):
            raise ValueError("Each camera must map to a distinct policy input")
        self.action_dim = policy.config.output_features[ACTION].shape[0]
        self.horizon = policy.config.chunk_size
        if cfg.chunk_steps > self.horizon:
            raise ValueError("chunk_steps cannot exceed the pretrained VLA chunk_size")
        if policy.config.n_obs_steps != 1 or any(
            getattr(policy.config, name, False) for name in ("use_visual_memory", "use_proprioceptive_memory")
        ):
            raise ValueError("EXPO-FT currently requires a single-observation VLA without temporal memory")
        rtc = getattr(policy.config, "rtc_config", None)
        if rtc is not None and rtc.enabled:
            raise ValueError("Use an ordinary SFT checkpoint with RTC disabled for original EXPO-FT")
        state_dim = policy.config.input_features[OBS_STATE].shape[0]
        self.critics = ChunkCritics(
            policy_image_keys,
            state_dim,
            self.action_dim,
            cfg.chunk_steps,
            cfg.hidden_dim,
            cfg.critic_image_size,
            cfg.num_critics,
        ).to(self.device)
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.edit = EditPolicy(
            cfg.hidden_dim, self.action_dim, cfg.chunk_steps, cfg.hidden_dim, cfg.edit_scale
        ).to(self.device)
        self.log_temperature = nn.Parameter(torch.tensor(cfg.temperature, device=self.device).log())
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=cfg.critic_lr)
        self.edit_optimizer = torch.optim.Adam(self.edit.parameters(), lr=cfg.edit_lr)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=cfg.edit_lr)
        params = [p for p in policy.parameters() if p.requires_grad]
        if not params:
            raise ValueError("The VLA has no trainable parameters; online EXPO-FT must update the VLA too")
        self.vla_optimizer = torch.optim.AdamW(params, lr=cfg.vla_lr)
        self.updates = 0

    def _amp(self) -> AbstractContextManager:
        if self.device.type == "cuda" and self.policy.config.use_amp:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def prepare(self, observations: list[dict[str, Tensor]], actions: Tensor | None = None) -> dict[str, Any]:
        batch = {
            key: torch.stack([obs[key] for obs in observations]).to(self.device) for key in observations[0]
        }
        for key in self.image_keys:
            batch[key] = batch[key].float() / 255.0
        batch["task"] = [self.cfg.task] * len(observations)
        if actions is not None:
            batch[ACTION] = actions.to(self.device)
        return self.preprocessor(batch)

    @torch.no_grad()
    def candidates(self, observations: list[dict[str, Tensor]]) -> Tensor:
        """Independent noise samples from the same observation, shaped B x N x C x D.

        Use the native chunk API, never select_action's single-action queue. Microbatches bound VLA
        inference memory and work for both PI05 and SmolVLA without changing offline training.
        """
        repeated = [obs for obs in observations for _ in range(self.cfg.num_candidates)]
        chunks = []
        self.policy.eval()
        for start in range(0, len(repeated), self.cfg.candidate_batch_size):
            self.policy.reset()
            batch = self.prepare(repeated[start : start + self.cfg.candidate_batch_size])
            with self._amp():
                chunk = self.policy.predict_action_chunk(batch)
            if chunk.ndim != 3 or chunk.shape[1] < self.cfg.chunk_steps:
                raise ValueError("VLA returned an invalid action chunk")
            chunks.append(chunk[:, : self.cfg.chunk_steps].float())
        candidates = torch.cat(chunks).view(
            len(observations), self.cfg.num_candidates, self.cfg.chunk_steps, self.action_dim
        )
        if not torch.isfinite(candidates).all():
            raise ValueError("VLA generated nonfinite actions")
        return candidates

    def select(self, features: Tensor, candidates: Tensor, target: bool = True) -> Tensor:
        """Maximize conservative Q across unedited AND edited chunk candidates."""
        batch_size, count, horizon, dim = candidates.shape
        if self.cfg.num_edits:
            edit_count = self.cfg.num_edits
            expanded = features[:, None].expand(-1, edit_count, -1).reshape(-1, features.shape[-1])
            edited, _ = self.edit(expanded, candidates[:, :edit_count].reshape(-1, horizon, dim))
            candidates = torch.cat([candidates, edited.view(batch_size, edit_count, horizon, dim)], dim=1)
            count += edit_count
        expanded = features[:, None].expand(-1, count, -1).reshape(-1, features.shape[-1])
        critics = self.target_critics if target else self.critics
        qs = critics.score(expanded, candidates.reshape(-1, horizon, dim))
        subset = torch.randperm(self.cfg.num_critics, device=self.device)[: self.cfg.num_min_critics]
        values = qs[:, subset].min(-1).values.view(batch_size, count)
        indices = values.argmax(-1)
        return candidates[torch.arange(batch_size, device=self.device), indices]

    @torch.no_grad()
    def sample_action_chunk(self, observation: dict[str, Tensor]) -> Tensor:
        candidates = self.candidates([observation])
        batch = self.prepare([observation])  # Also restores the relative-action anchor after microbatching.
        features = self.target_critics.encoder(batch)
        chosen = self.select(features, candidates)
        result = self.postprocessor(chosen).detach().cpu()[0]
        if result.shape != (self.cfg.chunk_steps, self.action_dim) or not torch.isfinite(result).all():
            raise ValueError("Invalid postprocessed joint chunk")
        return result

    def _step(self, loss: Tensor, optimizer: torch.optim.Optimizer, parameters: Any) -> None:
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite EXPO-FT training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()

    def update(self, replay: EpisodeReplay) -> dict[str, float]:
        cfg = self.cfg
        samples = replay.sample(cfg.batch_size, cfg.chunk_steps, cfg.discount)
        observations = [s["observation"] for s in samples]
        batch = self.prepare(observations, torch.stack([s["actions"] for s in samples]))
        actions = batch[ACTION].float()
        next_observations = [s["next_observation"] for s in samples]
        with torch.no_grad():
            candidates = self.candidates(next_observations)
            next_batch = self.prepare(next_observations)
            next_features = self.target_critics.encoder(next_batch)
            next_actions = self.select(next_features, candidates)
            qs = self.target_critics.score(next_features, next_actions)
            subset = torch.randperm(cfg.num_critics, device=self.device)[: cfg.num_min_critics]
            rewards = torch.stack([s["reward"] for s in samples]).to(self.device)
            bootstrap = torch.tensor([s["bootstrap"] for s in samples], device=self.device)
            target = rewards + bootstrap * qs[:, subset].min(-1).values
        q_loss = functional.mse_loss(
            self.critics(batch, actions), target[:, None].expand(-1, cfg.num_critics)
        )
        self._step(q_loss, self.critic_optimizer, self.critics.parameters())

        # Gradients reach the edit actor through Q(action), never through the VLA or critic encoder.
        with torch.no_grad():
            features = self.critics.encoder(batch)
        self.critics.requires_grad_(False)
        try:
            edited, log_prob = self.edit(features, actions.detach())
            values = self.critics.score(features, edited).min(-1).values
            edit_loss = (self.log_temperature.exp().detach() * log_prob - values).mean()
            self._step(edit_loss, self.edit_optimizer, self.edit.parameters())
        finally:
            self.critics.requires_grad_(True)
        target_entropy = -cfg.chunk_steps * self.action_dim / 2
        temperature_loss = -(self.log_temperature * (log_prob.detach() + target_entropy)).mean()
        self._step(temperature_loss, self.temperature_optimizer, [self.log_temperature])

        with torch.no_grad():
            for target_param, param in zip(
                self.target_critics.parameters(), self.critics.parameters(), strict=True
            ):
                target_param.lerp_(param, cfg.tau)

        metrics = {
            "critic_loss": q_loss.item(),
            "edit_loss": edit_loss.item(),
            "temperature": self.log_temperature.exp().item(),
        }
        success_samples = replay.sample_success(cfg.vla_batch_size, self.horizon)
        if success_samples:
            self.policy.reset()
            self.policy.train()
            training_batch = self.prepare(
                [s["observation"] for s in success_samples],
                torch.stack([s["actions"] for s in success_samples]),
            )
            with self._amp():
                vla_loss, _ = self.policy(training_batch)
            self._step(vla_loss, self.vla_optimizer, self.policy.parameters())
            metrics["vla_loss"] = vla_loss.item()
        self.policy.eval()
        self.updates += 1
        return metrics

    def save(self, path: Path, replay: EpisodeReplay, episodes: int, replay_dir: Path | None = None) -> None:
        path.mkdir(parents=True, exist_ok=False)
        policy_dir = path / "pretrained_model"
        self.policy.save_pretrained(policy_dir)
        if getattr(self.policy.config, "use_peft", False):
            # PEFT writes adapters, but does not write LeRobot's policy config.json.
            self.policy.config.save_pretrained(policy_dir)
        self.preprocessor.save_pretrained(policy_dir)
        self.postprocessor.save_pretrained(policy_dir)
        replay_state = replay.state_dict() if replay_dir is None else replay.save_shared(replay_dir)
        if replay_dir is not None:
            replay_state["directory"] = os.path.relpath(replay_dir, path)
            (path / "replay_manifest.json").write_text(
                json.dumps(
                    {key: value for key, value in replay_state.items() if key != "generator"},
                    indent=2,
                )
            )
        state = {
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "edit": self.edit.state_dict(),
            "log_temperature": self.log_temperature.detach(),
            "updates": self.updates,
            "episodes": episodes,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "replay": replay_state,
            "optimizers": {
                name: getattr(self, name).state_dict()
                for name in ("critic_optimizer", "edit_optimizer", "temperature_optimizer", "vla_optimizer")
            },
        }
        torch.save(state, path / "training_state.pt")

    def restore(self, path: Path, replay: EpisodeReplay) -> int:
        # policy and processors have already been loaded from path/pretrained_model.
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        for name in ("critics", "target_critics", "edit"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"])
        for name, value in state["optimizers"].items():
            getattr(self, name).load_state_dict(value)
        self.updates = state["updates"]
        replay_state = state["replay"]
        if replay_state.get("format") == "shared_episodes_v1":
            replay.load_shared(replay_state, path / replay_state["directory"])
        else:
            replay.load_state_dict(replay_state)
        torch.set_rng_state(state["rng"])
        if self.device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state["episodes"]
