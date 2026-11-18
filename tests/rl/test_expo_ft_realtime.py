"""Real-time temporal alignment, current-state selection, and resumable filter learning."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lerobot.rl.expo_ft.learner import ExpoFTLearner
from lerobot.rl.expo_ft.realtime import RealTimeEXPOFTLearner, RealTimePlanner
from lerobot.rl.expo_ft.replay import EpisodeReplay
from tests.rl.test_expo_ft import Processor, TinyVLA, cfg as cfg, episode, observation


class PrefixVLA(TinyVLA):
    def __init__(self):
        super().__init__()
        self.config.type = "smolvla"
        self.config.max_action_dim = 6
        self.prefixes = []

    def predict_action_chunk(self, batch, noise=None, hard_prefix=None, hard_prefix_mask=None):
        n = len(batch["observation.state"])
        result = self.weight + (torch.randn(n, 4, 6) if noise is None else noise)
        if hard_prefix is not None:
            self.prefixes.append(hard_prefix.detach().clone())
            result = torch.where(hard_prefix_mask, hard_prefix, result)
        return result

    def forward(self, batch, prefix_mask=None):
        self.training_mask = prefix_mask
        return super().forward(batch)


def make(cfg):
    cfg = replace(cfg, real_time=True, real_time_delay_steps=1)
    return RealTimeEXPOFTLearner(cfg, PrefixVLA(), Processor(), Processor())


def test_prefix_inpainting_and_reactive_window(cfg):
    agent = make(cfg)
    prefix = torch.ones(1, 1, 6) * 12
    proposals = agent.propose([observation()], prefix)
    torch.testing.assert_close(proposals[:, :, 0], prefix[:, None, 0].expand(1, 2, 6))
    seen = []

    def select(features, candidates, **kwargs):
        seen.append(candidates.clone())
        return candidates[:, 0]

    agent.select = select
    result = agent.react(observation(), proposals[0], delay=1)
    torch.testing.assert_close(result, proposals[0, 0, 1:3])
    assert seen[0].shape == (1, 2, 2, 6)


def test_delayed_replay_and_filter_update_resume(cfg, tmp_path):
    agent = make(cfg)
    replay = EpisodeReplay(100)
    entry = episode(8)
    replay.add(entry)
    for sample in replay.sample(30, 2, 0.99, delay=1):
        assert len(sample["delay_prefix"]) == 1
        torch.testing.assert_close(sample["delay_prefix"][0], entry.actions[min(sample["start"] + 2, 8) - 1])
    before = [p.clone() for p in agent.noise_filter.parameters()]
    metrics = agent.update(replay)
    assert torch.isfinite(torch.tensor(metrics["filter_loss"]))
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.noise_filter.parameters(), strict=True))
    assert agent.policy.training_mask.shape == (cfg.vla_batch_size, 4)
    dest = tmp_path / "checkpoint"
    agent.save(dest, replay, 1)
    other = make(cfg)
    restored_replay = EpisodeReplay(100)
    other.restore(dest, restored_replay)
    # Production load_learner loads these before restore; the tiny policy stands in for that load.
    other.policy.load_state_dict(agent.policy.state_dict())
    for a, b in zip(agent.noise_filter.parameters(), other.noise_filter.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    rng = torch.get_rng_state()
    expected = agent.update(replay)
    torch.set_rng_state(rng)
    actual = other.update(restored_replay)
    assert expected == actual
    for a, b in zip(agent.noise_filter.parameters(), other.noise_filter.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_prefetch_nonblocking_and_current_observation(cfg):
    pending = Future()
    observations = []

    class Worker:
        def submit(self, fn, *args):
            if fn.__name__ == "propose":
                return pending
            result = Future()
            result.set_result(fn(*args))
            return result

    def propose(*args):
        pass

    def react(obs, proposals, delay, details):
        observations.append(obs)
        assert delay == 1
        return proposals[0, delay : delay + 2]

    learner = SimpleNamespace(cfg=SimpleNamespace(capture_all_plans=False), propose=propose, react=react)
    planner = RealTimePlanner(learner, Worker())
    planner.request({"old": True}, torch.ones(1, 6))
    assert planner.take({"current": True}) is None
    pending.set_result(torch.arange(24).reshape(1, 1, 4, 6).float())
    result = planner.take({"current": True})
    assert observations == [{"current": True}]
    assert result.shape == (2, 6)


def test_incompatible_horizon_rejected(cfg):
    cfg = replace(cfg, real_time=True, chunk_steps=4, real_time_delay_steps=1)
    with pytest.raises(ValueError, match="chunk_steps"):
        RealTimeEXPOFTLearner(cfg, PrefixVLA(), Processor(), Processor())


def test_disabled_realtime_settings_do_not_change_actions_updates_or_rng(cfg):
    # Unused RT settings must not affect validation, model initialization, RNG,
    # replay draws, or any optimizer when the mode is disabled.
    agents, results = [], []
    for config in (cfg, replace(cfg, real_time_delay_steps=-1, real_time_filter_candidates=0)):
        torch.manual_seed(123)
        agent = ExpoFTLearner(config, TinyVLA(), Processor(), Processor())
        replay = EpisodeReplay(100, seed=456)
        replay.add(episode())
        chunk = agent.sample_action_chunk(observation())
        metrics = agent.update(replay)
        results.append((chunk, metrics, torch.get_rng_state(), replay.generator.get_state()))
        agents.append(agent)
        assert not hasattr(agent, "noise_filter")
    for a, b in zip(results[0], results[1], strict=True):
        if isinstance(a, dict):
            assert a == b
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for name in ("policy", "critics", "target_critics", "edit"):
        for key, value in getattr(agents[0], name).state_dict().items():
            torch.testing.assert_close(value, getattr(agents[1], name).state_dict()[key], rtol=0, atol=0)


def test_normal_backup_matches_original_inline_computation(cfg):
    agent = ExpoFTLearner(cfg, TinyVLA(), Processor(), Processor())
    replay = EpisodeReplay(100)
    replay.add(episode())
    samples = replay.sample(cfg.batch_size, cfg.chunk_steps, cfg.discount)
    rng = torch.get_rng_state()
    actual = agent.backup_target(samples)
    actual_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    with torch.no_grad():
        observations = [s["next_observation"] for s in samples]
        candidates = agent.candidates(observations)
        features = agent.target_critics.encoder(agent.prepare(observations))
        selected = agent.select(features, candidates)
        qs = agent.target_critics.score(features, selected)
        subset = torch.randperm(cfg.num_critics)[: cfg.num_min_critics]
        rewards = torch.stack([s["reward"] for s in samples])
        bootstrap = torch.tensor([s["bootstrap"] for s in samples])
        expected = rewards + bootstrap * qs[:, subset].min(-1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_rng, torch.get_rng_state())


def test_invalid_reactive_actions_rejected(cfg):
    agent = make(cfg)
    agent.postprocessor = lambda chunk: chunk * float("nan")
    with pytest.raises(ValueError, match="Invalid postprocessed"):
        agent.react(observation(), torch.zeros(2, 4, 6))


@pytest.mark.parametrize("completed", [False, True])
def test_cancel_discards_old_prefetch_before_resuming(completed):
    futures = []

    class Worker:
        def submit(self, *args):
            future = Future()
            futures.append(future)
            return future

    planner = RealTimePlanner(SimpleNamespace(propose=lambda *args: None), Worker())
    planner.request({"old": True}, torch.ones(1, 6))
    if completed:
        futures[0].set_result(torch.ones(1, 2, 4, 6))
    with pytest.raises(RuntimeError, match="already pending"):
        planner.request({"replacement": True})
    planner.cancel()
    assert planner.pending is None
    assert planner.take({"human": True}) is None
    planner.request({"fresh": True})
    assert planner.delay == 0 and planner.pending is futures[1]
    assert planner.completed == 0


@pytest.mark.parametrize("edits", [0, 2])
def test_backup_filter_sees_delayed_state_and_honors_edit_count(cfg, edits):
    agent = make(replace(cfg, num_edits=edits))
    replay = EpisodeReplay(100)
    replay.add(episode())
    samples = replay.sample(2, 2, 0.99, delay=1)
    seen = []
    handle = agent.edit.register_forward_pre_hook(lambda module, args: seen.append(len(args[0])))
    agent.backup_target(samples)
    handle.remove()
    assert seen == ([len(samples) * edits] if edits else [])
    inputs, _ = agent.filter_records[-1]
    features = agent.target_critics.encoder(agent.prepare([s["next_observation"] for s in samples]))
    stale = agent.target_critics.encoder(agent.prepare([s["delayed_next_observation"] for s in samples]))
    torch.testing.assert_close(inputs[:, : features.shape[1]], features)
    torch.testing.assert_close(inputs[:, features.shape[1] : 2 * features.shape[1]], stale)


def test_prefix_and_selection_keep_physical_coordinates_with_relative_processor(cfg):
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        NormalizerProcessorStep,
        RelativeActionsProcessorStep,
        UnnormalizerProcessorStep,
        make_policy_processor_pipelines,
    )
    from lerobot.utils.constants import ACTION

    relative = RelativeActionsProcessorStep(enabled=True)
    features = {ACTION: PolicyFeature(FeatureType.ACTION, (6,))}
    stats = {ACTION: {"min": torch.full((6,), -10.0), "max": torch.full((6,), 10.0)}}
    norm_map = {FeatureType.ACTION: NormalizationMode.MIN_MAX}
    pre, post = make_policy_processor_pipelines(
        input_steps=[relative, NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)],
        output_steps=[
            UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
            AbsoluteActionsProcessorStep(enabled=True, relative_step=relative),
        ],
    )
    agent = RealTimeEXPOFTLearner(
        replace(cfg, real_time=True, real_time_delay_steps=1, candidate_batch_size=1),
        PrefixVLA(),
        pre,
        post,
    )
    physical = torch.full((2, 1, 6), 25.0)
    proposed = agent.propose([observation(20), observation(30)], physical)
    torch.testing.assert_close(proposed[:, :, :1], physical[:, None].expand(2, 2, 1, 6))
    agent.select = lambda features, candidates, **kwargs: candidates[:, 0]
    result = agent.react(observation(35), proposed[0], delay=1)
    torch.testing.assert_close(result, proposed[0, 0, 1:3])


def test_smolvla_prefix_loss_masks_only_committed_and_padded_actions():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    losses = torch.arange(1.0, 25.0).reshape(2, 4, 3).requires_grad_()
    prefix = torch.tensor([[True, False, False, False], [True, True, False, False]])
    padding = torch.tensor([[False, False, False, True], [False, False, False, False]])
    seen = []

    def forward(*args, prefix_mask=None):
        seen.append(prefix_mask)
        return losses

    fake = SimpleNamespace(
        config=SimpleNamespace(
            action_feature=SimpleNamespace(shape=(3,)), max_action_dim=3, adapt_to_pi_aloha=False
        ),
        prepare_images=lambda batch: (None, None),
        prepare_state=lambda batch: None,
        prepare_action=lambda batch: None,
        model=SimpleNamespace(forward=forward),
    )
    batch = {OBS_LANGUAGE_TOKENS: None, OBS_LANGUAGE_ATTENTION_MASK: None, "action_is_pad": padding}
    loss, _ = SmolVLAPolicy.forward(fake, batch, prefix_mask=prefix)
    valid = ~(prefix | padding)
    expected = losses[valid].mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(losses.grad[~valid], torch.zeros_like(losses.grad[~valid]))
    torch.testing.assert_close(losses.grad[valid], torch.full_like(losses.grad[valid], 1 / 12))
    assert seen[0] is prefix


def test_smolvla_per_token_time_embedding_preserves_batch_alignment():
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    fake = SimpleNamespace(
        config=SimpleNamespace(min_period=0.004, max_period=4.0, chunk_size=4),
        vlm_with_expert=SimpleNamespace(expert_hidden_size=8),
        action_in_proj=torch.nn.Linear(3, 8),
        action_time_mlp_in=torch.nn.Linear(16, 8),
        action_time_mlp_out=torch.nn.Linear(8, 8),
    )
    actions = torch.randn(2, 4, 3)
    times = torch.tensor([0.2, 0.8])
    original = VLAFlowMatching.embed_suffix(fake, actions, times)[0]
    per_token = times[:, None].expand(2, 4).clone()
    expanded = VLAFlowMatching.embed_suffix(fake, actions, per_token)[0]
    torch.testing.assert_close(original, expanded, atol=0, rtol=0)
    per_token[:, :1] = 0
    conditioned = VLAFlowMatching.embed_suffix(fake, actions, per_token)[0]
    torch.testing.assert_close(original[:, 1:], conditioned[:, 1:], atol=0, rtol=0)
    assert not torch.equal(original[:, :1], conditioned[:, :1])


def test_control_prefetch_overlaps_execution_and_reacts_after_prefix(cfg, monkeypatch):
    import numpy as np

    from lerobot.lerobot_types import TransitionKey
    from lerobot.rl.expo_ft import train
    from lerobot.teleoperators.utils import TeleopEvents

    cfg = replace(cfg, real_time=True, real_time_delay_steps=1, episode_time_s=100)
    names = [f"joint{i}.pos" for i in range(6)]
    events = []

    class Robot:
        commands = []

        def get_observation(self):
            return {
                **dict.fromkeys(names, float(len(self.commands))),
                "wrist": np.zeros((32, 32, 3), dtype=np.uint8),
            }

        def send_action(self, action):
            self.commands.append(action)
            events.append(("send", len(self.commands)))
            return action

    robot = Robot()

    class Worker:
        def submit(self, fn, *args):
            result = Future()
            result.set_result(fn(*args))
            return result

    class Controller:
        phase = train.LeaderPhase.AUTONOMOUS

        def reset(self):
            pass

        def __call__(self, transition):
            return {
                TransitionKey.INFO: {
                    TeleopEvents.SUCCESS: len(robot.commands) >= 6,
                    TeleopEvents.TERMINATE_EPISODE: False,
                    TeleopEvents.RERECORD_EPISODE: False,
                    TeleopEvents.IS_INTERVENTION: False,
                }
            }

    def propose(observations, prefix):
        events.append(("propose", len(robot.commands), None if prefix is None else prefix.clone()))
        return torch.ones(1, 2, 4, 6)

    def react(obs, proposals, delay, details):
        events.append(("react", len(robot.commands), obs["observation.state"].clone()))
        return proposals[0, delay : delay + 2]

    agent = SimpleNamespace(cfg=cfg, image_keys=["observation.images.wrist"], propose=propose, react=react)
    monkeypatch.setattr(train, "precise_sleep", lambda _: None)
    result = train.run_episode(
        cfg,
        agent,
        Worker(),
        robot,
        SimpleNamespace(get_action=lambda: dict.fromkeys(names, 0.0)),
        Controller(),
        names,
        names,
    )
    assert result.success
    calls = [e for e in events if e[0] == "propose"]
    assert [e[1] for e in calls] == [0, 1, 3, 5]
    assert calls[0][2] is None and calls[1][2].shape == (1, 1, 6)
    reactions = [e for e in events if e[0] == "react"]
    assert [e[1] for e in reactions] == [0, 2, 4]
    torch.testing.assert_close(reactions[1][2], torch.full((6,), 2.0))
