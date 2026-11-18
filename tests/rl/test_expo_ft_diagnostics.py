"""Live planning observations must be fresh, and diagnostic captures must replay exactly."""

import json
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.rl.expo_ft import train
from lerobot.rl.expo_ft.diagnostics import capture_initial_plan, flush_captures, sample_debug_plan
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_STATE
from tests.rl.test_expo_ft import IMAGE, cfg as cfg, learner, observation


@pytest.mark.parametrize("capture_all", [False, True])
def test_initial_plan_capture_preserves_rng_and_selected_chunk(cfg, capture_all):
    torch.set_num_threads(1)
    cfg.eval_only = True
    cfg.capture_all_plans = capture_all
    agent = learner(cfg)
    result = capture_initial_plan(agent, observation())
    flush_captures()
    directory = next((cfg.output_dir / ("policy_plans" if capture_all else "initial_plans")).iterdir())
    captured = torch.load(directory / "plan.pt", weights_only=True)
    torch.set_rng_state(captured["rng"])
    repeated = agent.sample_action_chunk(captured["observation"])
    torch.testing.assert_close(result, repeated, rtol=0, atol=0)
    torch.testing.assert_close(captured["action_chunk"], repeated, rtol=0, atol=0)
    assert (directory / "wrist.png").is_file()
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["eval_only"] is True and metadata["updates"] == 0
    assert len(metadata["candidate_scores"][0]) == cfg.num_candidates + cfg.num_edits
    assert len(metadata["critic_subset"]) == cfg.num_min_critics


def test_first_plan_refreshes_observation_after_blocking_intervention(cfg, monkeypatch):
    cfg.capture_control_trace = True
    monkeypatch.setattr(train, "precise_sleep", lambda _: None)
    names = [f"joint{i}.pos" for i in range(6)]
    seen = []

    class Robot:
        pose = 0.0

        def __init__(self):
            self.commands = []

        def get_observation(self):
            return {**dict.fromkeys(names, self.pose), "wrist": np.zeros((32, 32, 3), dtype=np.uint8)}

        def send_action(self, action):
            self.commands.append(action)
            return action

    robot = Robot()

    class Controller:
        phase = train.LeaderPhase.AUTONOMOUS

        def reset(self):
            pass

        def __call__(self, transition):
            # The scene/state can change while a synchronous catch-up command blocks.
            robot.pose = 10.0
            return {
                TransitionKey.INFO: {
                    TeleopEvents.SUCCESS: len(robot.commands) >= cfg.chunk_steps,
                    TeleopEvents.TERMINATE_EPISODE: False,
                    TeleopEvents.RERECORD_EPISODE: False,
                    TeleopEvents.IS_INTERVENTION: False,
                }
            }

    class Worker:
        def submit(self, fn, *args):
            future = Future()
            future.set_result(fn(*args))
            return future

    def plan(obs):
        seen.append(obs[OBS_STATE].clone())
        return obs[OBS_STATE].expand(cfg.chunk_steps, -1).clone()

    agent = SimpleNamespace(cfg=cfg, image_keys=[IMAGE], sample_action_chunk=plan)
    leader = SimpleNamespace(get_action=lambda: dict.fromkeys(names, 0.0))
    result = train.run_episode(cfg, agent, Worker(), robot, leader, Controller(), names, names)
    assert result.success
    torch.testing.assert_close(seen[0], torch.full((6,), 10.0))
    torch.testing.assert_close(result.actions, torch.full((cfg.chunk_steps, 6), 10.0))
    flush_captures()
    records = [
        json.loads(line)
        for line in next((cfg.output_dir / "control_traces").glob("*.jsonl")).read_text().splitlines()
    ]
    commands = [row for row in records if row["event"] == "command"]
    assert len(commands) == cfg.chunk_steps
    assert [row["action_index"] for row in commands] == list(range(cfg.chunk_steps))
    assert all(row["requested"] == row["limited"] == row["sent"] for row in commands)
    assert all(row["send_started_ns"] <= row["monotonic_ns"] for row in commands)


def test_debug_seed_overrides_prior_rng_and_matches_modes(cfg):
    cfg.debug_plan_seed = 2026
    agent = SimpleNamespace(cfg=cfg, sample_action_chunk=lambda obs: torch.randn(3, 6))
    first = sample_debug_plan(agent, {}, 0, "train")
    torch.randn(1000)  # Simulate checkpoint/setup RNG consumption.
    cfg.eval_only = True
    repeated = sample_debug_plan(agent, {}, 0, "eval")
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.equal(first, sample_debug_plan(agent, {}, 1, "eval"))


def test_frozen_debug_requires_checkpoint_and_cannot_enable_check_only(cfg, tmp_path):
    with pytest.raises(ValueError, match="debug_frozen_training"):
        replace(cfg, debug_frozen_training=True)
    with pytest.raises(ValueError, match="debug_frozen_training"):
        replace(cfg, debug_frozen_training=True, resume=tmp_path, check_only=True)
    replace(cfg, debug_frozen_training=True, resume=tmp_path)


def test_frozen_training_runs_new_episode_without_updates_or_checkpoint(cfg, monkeypatch, tmp_path):
    cfg = replace(cfg, resume=tmp_path, debug_frozen_training=True, num_episodes=1)
    names = [f"joint{i}.pos" for i in range(6)]
    cfg.robot.cameras = {"wrist": None}
    calls = []

    class Arm:
        is_connected = False
        is_calibrated = True
        action_features = dict.fromkeys(names, float)
        cameras = {"wrist": None}

        def connect(self, **kwargs):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def get_observation(self):
            return dict.fromkeys(names, 0.0)

        get_action = get_observation

    replay = SimpleNamespace(add=lambda episode: calls.append("add"))
    agent = SimpleNamespace(image_keys=[IMAGE])
    metadata = SimpleNamespace(features={key: {"names": names} for key in ["action", OBS_STATE]})
    monkeypatch.setattr(train, "load_learner", lambda cfg: (agent, metadata, replay, 71))
    monkeypatch.setattr(train, "make_robot_from_config", lambda cfg: Arm())
    monkeypatch.setattr(train, "make_teleoperator_from_config", lambda cfg: Arm())
    monkeypatch.setattr(train, "capture_home_poses", lambda *args: dict.fromkeys(names, 0.0))
    monkeypatch.setattr(train, "move_follower", lambda *args: None)
    monkeypatch.setattr(train, "precise_sleep", lambda *args: None)
    monkeypatch.setattr(
        train,
        "LeaderArmInterventionStep",
        lambda **kwargs: SimpleNamespace(reset=lambda: None, close=lambda: None, _follow=lambda *args: None),
    )
    monkeypatch.setattr(train, "run_episode", lambda *args: SimpleNamespace(success=True))

    def updates(agent, replay, count):
        assert count == 0
        calls.append("zero_updates")

    monkeypatch.setattr(train, "train_updates", updates)
    monkeypatch.setattr(train, "save_checkpoint", lambda *args: pytest.fail("Frozen run saved a checkpoint"))
    train.train(cfg)
    assert calls == ["add", "zero_updates"]
