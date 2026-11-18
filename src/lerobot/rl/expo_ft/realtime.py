"""Latency-aware SmolVLA EXPO-FT, based on arXiv:2609.18207.

Slow prefix-conditioned proposals and fast current-observation edits are separate calls.
Only the slow call runs during execution; processors are never shared concurrently.
"""

import time

import torch
from torch import nn

from lerobot.utils.constants import ACTION

from .learner import ExpoFTLearner
from .networks import mlp


class RealTimeEXPOFTLearner(ExpoFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.policy.config.type != "smolvla":
            raise ValueError("real_time currently supports SmolVLA; PI05 remains supported in normal mode")
        if self.cfg.chunk_steps + self.cfg.real_time_delay_steps > self.horizon:
            raise ValueError(
                "real_time requires chunk_steps + real_time_delay_steps <= VLA chunk_size; use chunk_steps=10 for a 20-action model"
            )
        if getattr(self.policy.config, "adapt_to_pi_aloha", False):
            raise ValueError("real_time does not support adapt_to_pi_aloha")
        dim = self.horizon * self.policy.config.max_action_dim
        self.noise_filter = nn.ModuleList(
            # The released real-time config conditions Q_f on both the current
            # critic observation and the delayed observation used by the VLA.
            [mlp(2 * self.critics.feature_dim + dim, 1, self.cfg.hidden_dim) for _ in range(2)]
        ).to(self.device)
        self.filter_optimizer = torch.optim.Adam(self.noise_filter.parameters(), lr=self.cfg.critic_lr)
        self.filter_records = []

    @torch.no_grad()
    def propose(self, observations, prefixes=None, noise=None, count=None):
        """Return physical full-horizon proposals, preserving the request's normalization anchor."""
        count = self.cfg.num_candidates if count is None else count
        if prefixes is not None and (
            prefixes.ndim != 3
            or prefixes.shape[0] != len(observations)
            or prefixes.shape[2] != self.action_dim
            or prefixes.shape[1] > self.cfg.real_time_delay_steps
            or not torch.isfinite(prefixes).all()
        ):
            raise ValueError("Invalid real-time action prefix")
        repeated = [obs for obs in observations for _ in range(count)]
        chunks = []
        self.policy.eval()
        for start in range(0, len(repeated), self.cfg.candidate_batch_size):
            stop = start + self.cfg.candidate_batch_size
            self.policy.reset()
            obs = repeated[start:stop]
            batch = self.prepare(obs)
            kwargs = {}
            if prefixes is not None and prefixes.shape[1]:
                physical = prefixes.repeat_interleave(count, 0)[start:stop]
                normalized = self.prepare(obs, physical)[ACTION]
                prefix = torch.zeros(
                    len(obs), self.horizon, self.policy.config.max_action_dim, device=self.device
                )
                prefix[:, : physical.shape[1], : self.action_dim] = normalized
                mask = torch.zeros_like(prefix, dtype=torch.bool)
                mask[:, : physical.shape[1]] = True
                kwargs.update(hard_prefix=prefix, hard_prefix_mask=mask)
            if noise is not None:
                kwargs["noise"] = noise[start:stop]
            with self._amp():
                chunk = self.policy.predict_action_chunk(batch, **kwargs).float()
            chunks.append(self.postprocessor(chunk).detach().cpu())
        result = torch.cat(chunks).reshape(len(observations), count, self.horizon, self.action_dim)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite real-time VLA proposals")
        return result

    @torch.no_grad()
    def react(self, observation, physical_candidates, delay=0, diagnostics=None):
        if not 0 <= delay <= self.cfg.real_time_delay_steps:
            raise ValueError("Invalid real-time proposal delay")
        if physical_candidates.ndim != 3 or not torch.isfinite(physical_candidates).all():
            raise ValueError("Invalid real-time proposal candidates")
        window = physical_candidates[:, delay : delay + self.cfg.chunk_steps]
        if window.shape[1] != self.cfg.chunk_steps:
            raise ValueError("Real-time proposal has no complete execution window")
        obs = [observation] * len(window)
        normalized = self.prepare(obs, window)[ACTION]
        features = self.target_critics.encoder(self.prepare([observation]))
        chosen = self.select(features, normalized[None], diagnostics=diagnostics)
        result = self.postprocessor(chosen).detach().cpu()[0]
        if result.shape != (self.cfg.chunk_steps, self.action_dim) or not torch.isfinite(result).all():
            raise ValueError("Invalid postprocessed joint chunk")
        return result

    @torch.no_grad()
    def sample_action_chunk(self, observation, diagnostics=None):
        return self.react(observation, self.propose([observation])[0], diagnostics=diagnostics)

    def vla_loss(self, batch, samples=None):
        # Clean prefix at per-token flow time zero; only postfix contributes to loss.
        delay = self.cfg.real_time_delay_steps
        lengths = torch.tensor(
            [0 if sample["start"] < self.cfg.chunk_steps else delay for sample in samples], device=self.device
        )
        mask = torch.arange(self.horizon, device=self.device)[None] < lengths[:, None]
        return self.policy(batch, prefix_mask=mask)[0]

    @torch.no_grad()
    def backup_target(self, samples):
        current = [s["next_observation"] for s in samples]
        stale = [s["delayed_next_observation"] for s in samples]
        features = self.target_critics.encoder(self.prepare(current))
        delayed_features = self.target_critics.encoder(self.prepare(stale))
        filter_features = torch.cat([features, delayed_features], -1)
        pool = torch.randn(
            len(samples),
            self.cfg.real_time_filter_candidates,
            self.horizon,
            self.policy.config.max_action_dim,
            device=self.device,
        )
        expanded = filter_features[:, None].expand(-1, pool.shape[1], -1)
        inputs = torch.cat([expanded, pool.flatten(2)], -1)
        scores = torch.cat([head(inputs) for head in self.noise_filter], -1).min(-1).values
        noise = pool[torch.arange(len(samples), device=self.device), scores.argmax(-1)]
        # Terminal short chunks never bootstrap; repeat their last recorded prefix action.
        lengths = [len(s["delay_prefix"]) for s in samples]
        delay = self.cfg.real_time_delay_steps
        prefixes = torch.stack(
            [
                torch.cat([s["delay_prefix"], s["delay_prefix"][-1:].expand(delay - n, -1)])
                for s, n in zip(samples, lengths, strict=True)
            ]
        )
        proposals = self.propose(stale, prefixes, noise=noise, count=1)[
            :, 0, delay : delay + self.cfg.chunk_steps
        ]
        bases = self.prepare(current, proposals)[ACTION]
        # Restore current-state anchor after processing the delayed proposals.
        self.prepare(current)
        # Match the released filter_n_edit=-1 setting: edit the surviving base
        # num_edits times, using one common critic subset for all candidates.
        candidates = bases[:, None]
        if self.cfg.num_edits:
            count = self.cfg.num_edits
            edited, _ = self.edit(features.repeat_interleave(count, 0), bases.repeat_interleave(count, 0))
            candidates = torch.cat(
                [candidates, edited.reshape(len(samples), count, self.cfg.chunk_steps, self.action_dim)], 1
            )
        count = candidates.shape[1]
        subset = torch.randperm(self.cfg.num_critics, device=self.device)[: self.cfg.num_min_critics]
        qs = (
            self.target_critics.score(features.repeat_interleave(count, 0), candidates.flatten(0, 1))[
                :, subset
            ]
            .min(-1)
            .values.reshape(len(samples), count)
        )
        self.filter_records.append(
            (torch.cat([filter_features, noise.flatten(1)], -1).detach(), qs[:, 0].detach())
        )
        rewards = torch.stack([s["reward"] for s in samples]).to(self.device)
        bootstrap = torch.tensor([s["bootstrap"] for s in samples], device=self.device)
        return rewards + bootstrap * qs.max(-1).values

    def update(self, replay):
        self.filter_records.clear()
        metrics = super().update(replay)
        inputs = torch.cat([r[0] for r in self.filter_records])
        target = torch.cat([r[1] for r in self.filter_records])
        loss = sum((head(inputs).squeeze(-1) - target).square().mean() for head in self.noise_filter) / len(
            self.noise_filter
        )
        self._step(loss, self.filter_optimizer, self.noise_filter.parameters())
        metrics["filter_loss"] = loss.item()
        return metrics

    def save(self, path, replay, episodes, replay_dir=None):
        super().save(path, replay, episodes, replay_dir)
        torch.save(
            {
                "version": 2,
                "filter": self.noise_filter.state_dict(),
                "optimizer": self.filter_optimizer.state_dict(),
            },
            path / "realtime_state.pt",
        )

    def restore(self, path, replay):
        state = torch.load(path / "realtime_state.pt", map_location=self.device, weights_only=True)
        if state.get("version") != 2:
            raise ValueError(
                "Incompatible experimental real-time filter checkpoint; start a fresh RL run using its pretrained_model as policy_path"
            )
        episodes = super().restore(path, replay)
        self.noise_filter.load_state_dict(state["filter"])
        self.filter_optimizer.load_state_dict(state["optimizer"])
        return episodes


class RealTimePlanner:
    """One slow request at a time; invalidate prefetch on every intervention transition."""

    def __init__(self, learner, worker):
        self.learner, self.worker = learner, worker
        self.pending = None
        self.delay = 0
        self.requested = 0.0
        self.request_count = 0
        self.selected_index = None
        self.completed = 0
        self.total_latency = 0.0
        self.deadline_warned = False

    def request(self, observation, prefix=None):
        if self.pending is not None:
            raise RuntimeError("A real-time proposal is already pending")
        self.deadline_warned = False
        self.delay = 0 if prefix is None else len(prefix)
        self.requested = time.monotonic()
        self.pending = self.worker.submit(
            self.learner.propose, [observation], None if prefix is None else prefix[None]
        )
        self.request_count += 1

    def take(self, observation):
        if self.pending is None:
            return None
        if not self.pending.done():
            if self.delay and not self.deadline_warned:
                import logging

                logging.getLogger(__name__).warning(
                    "Real-time VLA missed the prefetch deadline; holding the last target. Increase real_time_delay_steps within the model horizon."
                )
                self.deadline_warned = True
            return None
        proposals = self.pending.result()[0]
        self.pending = None
        self.selected_index = self.request_count - 1
        self.completed += 1
        self.total_latency += time.monotonic() - self.requested
        # Worker owns all processor/GPU calls, including the fast phase.
        details = {} if self.learner.cfg.capture_all_plans else None
        chunk = self.worker.submit(self.learner.react, observation, proposals, self.delay, details).result()
        if details is not None:
            from .diagnostics import _pending, _writer

            path = self.learner.cfg.output_dir / "realtime_plans" / f"{time.time_ns()}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            _pending.append(
                _writer.submit(
                    torch.save,
                    {
                        "observation": observation,
                        "physical_proposals": proposals,
                        "delay": self.delay,
                        "details": details,
                        "action_chunk": chunk,
                    },
                    path,
                )
            )
        return chunk

    def cancel(self):
        if self.pending is not None:
            self.pending.cancel()
            if not self.pending.cancelled():
                self.pending.result()
            self.pending = None
