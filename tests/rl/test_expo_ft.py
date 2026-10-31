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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.envs.configs import LeaderInterventionConfig
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    create_transition,
    make_policy_processor_pipelines,
)
from lerobot.processor.factory import load_pretrained_policy_processors
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.rl.expo_ft.configuration import ExpoFTConfig
from lerobot.rl.expo_ft.control import limit_joint_target
from lerobot.rl.expo_ft.learner import ExpoFTLearner
from lerobot.rl.expo_ft.replay import Episode, EpisodeReplay
from lerobot.rl.expo_ft.train import run_episode
from lerobot.rl.leader_intervention import INTERVENTION_JOINT_TARGET_KEY, LeaderArmInterventionStep
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.rl.test_leader_intervention import MOTORS, FakeLeader, SilentEpisodeEvents, joints

IMAGE = "observation.images.wrist"


def observation(value=0):
    return {
        OBS_STATE: torch.full((6,), float(value)),
        IMAGE: torch.full((3, 32, 32), value, dtype=torch.uint8),
    }


def episode(steps=8, success=True):
    rewards = torch.zeros(steps)
    rewards[-1] = float(success)
    return Episode(
        [observation(i) for i in range(steps + 1)],
        torch.arange(steps * 6, dtype=torch.float32).view(steps, 6) / 100,
        rewards,
        torch.zeros(steps, dtype=torch.bool),
        success,
    )


@pytest.fixture
def cfg(tmp_path):
    return ExpoFTConfig(
        policy_path="unused",
        dataset_repo_id="local/test",
        robot=SOFollowerRobotConfig(port="unused"),
        teleop=SOLeaderTeleopConfig(port="unused"),
        reset_pose=[0.0] * 6,
        home_pose=[0.0] * 6,
        output_dir=tmp_path,
        device="cpu",
        num_candidates=2,
        num_edits=2,
        candidate_batch_size=2,
        hidden_dim=16,
        batch_size=2,
        num_critics=2,
        num_min_critics=2,
        chunk_steps=2,
        critic_image_size=32,
        storage_image_size=32,
    )


class TinyVLA(nn.Module):
    """A flow-policy stand-in: stochastic chunks and a supervised loss with real gradients."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.config = SimpleNamespace(
            image_features={IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 32, 32))},
            input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (6,))},
            output_features={ACTION: PolicyFeature(FeatureType.ACTION, (6,))},
            chunk_size=4,
            n_obs_steps=1,
            use_amp=False,
        )
        self.batch_sizes = []

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        batch_size = len(batch[OBS_STATE])
        self.batch_sizes.append(batch_size)
        return self.weight + torch.randn(batch_size, 4, 6)

    def forward(self, batch):
        return ((self.weight - batch[ACTION]) ** 2).mean(), {}

    def save_pretrained(self, path):
        path.mkdir(parents=True)
        torch.save(self.state_dict(), path / "tiny.pt")


class Processor:
    def __call__(self, value):
        return value

    def save_pretrained(self, path):
        pass


def learner(cfg):
    return ExpoFTLearner(cfg, TinyVLA(), Processor(), Processor())


def test_joint_interventions_need_no_ik_and_preserve_follow_mode():
    leader, events = FakeLeader(), SilentEpisodeEvents()
    step = LeaderArmInterventionStep(
        teleop_device=leader,
        kinematics=None,
        motor_names=MOTORS,
        end_effector_step_sizes={},
        config=LeaderInterventionConfig(follow_follower=True),
        episode_events=events,
        action_representation="joint_positions",
    )
    try:
        follower = joints()
        step(create_transition(observation=follower, complementary_data={TELEOP_ACTION_KEY: joints()}))
        assert leader.feedback[-1] == follower
        events.dispatch("space")
        leader.position = joints(wrist_roll=40, gripper=80)
        transition = step(
            create_transition(
                observation=follower, complementary_data={TELEOP_ACTION_KEY: leader.get_action()}
            )
        )
        targets = transition[TransitionKey.COMPLEMENTARY_DATA][INTERVENTION_JOINT_TARGET_KEY]
        labels = transition[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]
        np.testing.assert_allclose(labels, [targets[f"{m}.pos"] for m in MOTORS])
        assert targets["wrist_roll.pos"] == 9
        assert targets["gripper.pos"] == 9
        assert transition[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION]
        assert not leader.torque
        events.dispatch("space")
        transition = step(
            create_transition(
                observation=follower, complementary_data={TELEOP_ACTION_KEY: leader.get_action()}
            )
        )
        assert not transition[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION]
    finally:
        step.close()


def test_replay_discounts_short_terminal_chunks_and_never_crosses_episodes():
    replay = EpisodeReplay(20, seed=0)
    replay.add(episode(3))
    samples = replay.sample(100, horizon=5, discount=0.5)
    assert {s["steps"] for s in samples} == {1, 2, 3}
    for sample in samples:
        assert sample["bootstrap"] == 0
        assert sample["reward"] == 0.5 ** (sample["steps"] - 1)
        assert sample["actions"].shape == (5, 6)
        assert torch.equal(sample["actions"][-1], replay.episodes[0].actions[-1])


def test_success_replay_uses_only_full_real_horizons_and_eviction():
    replay = EpisodeReplay(10)
    replay.add(episode(8, success=True))
    replay.add(episode(3, success=False))
    assert replay.frames == 3
    assert replay.sample_success(5, 2) == []
    replay.add(episode(5, success=True))
    assert replay.sample_success(2, 6) == []
    assert len(replay.sample_success(2, 4)) == 2


def test_vla_samples_multiple_chunks_with_bounded_microbatches(cfg):
    agent = learner(cfg)
    candidates = agent.candidates([observation(), observation(1)])
    assert candidates.shape == (2, 2, 2, 6)
    assert not torch.equal(candidates[0, 0], candidates[0, 1])
    assert max(agent.policy.batch_sizes) <= cfg.candidate_batch_size
    assert agent.sample_action_chunk(observation()).shape == (2, 6)


def test_selector_keeps_unedited_candidates_and_can_choose_edits(cfg, monkeypatch):
    agent = learner(cfg)
    features = torch.zeros(1, cfg.hidden_dim)
    base = torch.zeros(1, 2, 2, 6)
    base[:, 1] = 1

    def scores(features, actions):
        return actions[:, 0, :1].expand(-1, cfg.num_critics)

    monkeypatch.setattr(agent.target_critics, "score", scores)
    monkeypatch.setattr(
        agent.edit, "forward", lambda features, actions: (actions - 2, torch.zeros(len(actions)))
    )
    assert torch.equal(agent.select(features, base), base[:, 1])
    monkeypatch.setattr(
        agent.edit, "forward", lambda features, actions: (actions + 2, torch.zeros(len(actions)))
    )
    assert torch.equal(agent.select(features, base), base[:, 1] + 2)


def test_update_changes_vla_edit_critics_and_target(cfg):
    torch.set_num_threads(1)
    agent = learner(cfg)
    replay = EpisodeReplay(100)
    replay.add(episode())
    before = {
        name: [p.detach().clone() for p in getattr(agent, name).parameters()]
        for name in ("policy", "edit", "critics", "target_critics")
    }
    metrics = agent.update(replay)
    assert set(metrics) == {"critic_loss", "edit_loss", "temperature", "vla_loss"}
    for name, params in before.items():
        assert any(
            not torch.equal(old, new)
            for old, new in zip(params, getattr(agent, name).parameters(), strict=True)
        )
    assert all(p.grad is None for p in agent.target_critics.parameters())


def test_failed_episode_does_not_update_vla(cfg):
    agent = learner(cfg)
    replay = EpisodeReplay(100)
    replay.add(episode(success=False))
    before = agent.policy.weight.detach().clone()
    assert "vla_loss" not in agent.update(replay)
    assert torch.equal(before, agent.policy.weight)


def test_checkpoint_restores_models_optimizers_and_replay(cfg, tmp_path):
    agent = learner(cfg)
    replay = EpisodeReplay(100)
    replay.add(episode())
    agent.update(replay)
    path = tmp_path / "checkpoint"
    agent.save(path, replay, 3)
    restored = learner(cfg)
    restored.policy.load_state_dict(torch.load(path / "pretrained_model/tiny.pt", weights_only=True))
    restored_replay = EpisodeReplay(100)
    assert restored.restore(path, restored_replay) == 3
    assert restored.updates == 1
    assert restored_replay.frames == replay.frames
    assert all(
        torch.equal(old, new)
        for old, new in zip(agent.edit.parameters(), restored.edit.parameters(), strict=True)
    )
    assert torch.equal(replay.generator.get_state(), restored_replay.generator.get_state())


def test_shared_checkpoints_deduplicate_replay_and_resume_after_relocation(cfg, tmp_path):
    import shutil

    from lerobot.rl.expo_ft.train import save_checkpoint

    cfg.output_dir = tmp_path / "run"
    agent = learner(cfg)
    replay = EpisodeReplay(100)
    replay.add(episode())
    agent.update(replay)
    first = save_checkpoint(agent, replay, 1)
    ids = [e.storage_id for e in replay.episodes]
    paths = [cfg.output_dir / "replay" / f"{storage_id}.pt" for storage_id in ids]
    original_mtimes = [p.stat().st_mtime_ns for p in paths]
    second = save_checkpoint(agent, replay, 2)
    assert len(list((cfg.output_dir / "replay").glob("*.pt"))) == 1
    assert [p.stat().st_mtime_ns for p in paths] == original_mtimes
    state = torch.load(second / "training_state.pt", map_location="cpu", weights_only=True)
    assert state["replay"]["episode_ids"] == ids
    assert "episodes" not in state["replay"]
    assert first.exists()
    relocated_root = tmp_path / "relocated"
    shutil.copytree(cfg.output_dir, relocated_root)
    restored = learner(cfg)
    restored_replay = EpisodeReplay(100)
    assert restored.restore(relocated_root / "checkpoints" / second.name, restored_replay) == 2
    assert restored_replay.frames == replay.frames
    assert torch.equal(restored_replay.episodes[0].actions, replay.episodes[0].actions)
    assert torch.equal(restored_replay.generator.get_state(), replay.generator.get_state())


def test_retention_preserves_unmanaged_checkpoints_and_live_replay(cfg):
    from lerobot.rl.expo_ft.train import save_checkpoint

    cfg.checkpoint_keep_last = 2
    legacy = cfg.output_dir / "checkpoints/episode_000000_update_00000000"
    legacy.mkdir(parents=True)
    (legacy / "user_file.txt").write_text("leave this untouched")
    agent = learner(cfg)
    replay = EpisodeReplay(8)
    saved = []
    episode_ids = []
    for number in range(1, 4):
        entry = episode()
        episode_ids.append(entry.storage_id)
        replay.add(entry)  # Evicts the previous episode from active RAM replay.
        saved.append(save_checkpoint(agent, replay, number))
    assert legacy.is_dir()
    assert not saved[0].exists()
    assert saved[1].exists() and saved[2].exists()
    remaining = {p.stem for p in (cfg.output_dir / "replay").glob("*.pt")}
    assert remaining == set(episode_ids[1:])
    for path in saved[1:]:
        restored = EpisodeReplay(8)
        agent.restore(path, restored)
        assert restored.frames == 8


def test_missing_shared_replay_reports_required_directory(cfg, tmp_path):
    replay = EpisodeReplay(100)
    replay.add(episode())
    state = replay.save_shared(tmp_path / "replay")
    with pytest.raises(FileNotFoundError, match="Copy the run's replay folder"):
        EpisodeReplay(100).load_shared(state, tmp_path / "absent")


def test_failed_save_removes_only_its_own_temporary_directory(cfg, monkeypatch):
    from lerobot.rl.expo_ft.train import save_checkpoint

    agent = learner(cfg)
    other = cfg.output_dir / "checkpoints/unrelated.tmp"
    other.mkdir(parents=True)

    def fail(path, *args, **kwargs):
        path.mkdir()
        (path / "partial.bin").write_bytes(b"partial")
        raise OSError("simulated full disk")

    monkeypatch.setattr(agent, "save", fail)
    with pytest.raises(OSError, match="simulated full disk"):
        save_checkpoint(agent, EpisodeReplay(100), 1)
    assert list((cfg.output_dir / "checkpoints").iterdir()) == [other]


def test_existing_run_requires_explicit_resume(cfg):
    from lerobot.rl.expo_ft.train import load_learner

    existing = cfg.output_dir / "checkpoints/episode_000001_update_00000001"
    existing.mkdir(parents=True)
    (existing / "training_state.pt").write_bytes(b"existing checkpoint")
    with pytest.raises(FileExistsError, match="Use --resume"):
        load_learner(cfg)


def test_joint_limits_cap_policy_and_corrections_and_reject_nonfinite():
    assert limit_joint_target({"a": 200}, {"a": 0}, 9, {"a": [-20, 20]}) == {"a": 9}
    assert limit_joint_target({"a": 200}, {"a": 40}, 9, {"a": [-20, 20]}) == {"a": 31}
    with pytest.raises(ValueError, match="Nonfinite"):
        limit_joint_target({"a": float("nan")}, {"a": 0}, 9, {})


def test_episode_records_actual_sent_targets_and_manual_reward(cfg, monkeypatch):
    from lerobot.rl.expo_ft import train

    monkeypatch.setattr(train, "precise_sleep", lambda _: None)
    names = [f"{m}.pos" for m in MOTORS]
    agent = learner(cfg)
    leader = FakeLeader()

    class Robot:
        def __init__(self):
            self.pose = dict.fromkeys(names, 0.0)
            self.commands = []

        def get_observation(self):
            return {**self.pose, "wrist": np.zeros((32, 32, 3), dtype=np.uint8)}

        def send_action(self, desired):
            self.pose = dict.fromkeys(names, 0.25)  # Simulate a hardware-side target cap.
            self.commands.append(self.pose)
            return self.pose

    robot = Robot()

    class Controller:
        phase = train.LeaderPhase.CORRECTING

        def reset(self):
            pass

        def __call__(self, transition):
            success = len(robot.commands) == 2
            return {
                TransitionKey.INFO: {
                    TeleopEvents.SUCCESS: success,
                    TeleopEvents.TERMINATE_EPISODE: False,
                    TeleopEvents.RERECORD_EPISODE: False,
                    TeleopEvents.IS_INTERVENTION: True,
                },
                TransitionKey.COMPLEMENTARY_DATA: {INTERVENTION_JOINT_TARGET_KEY: dict.fromkeys(names, 30.0)},
            }

    with ThreadPoolExecutor(max_workers=1) as worker:
        result = run_episode(cfg, agent, worker, robot, leader, Controller(), names, names)
    assert result.success
    assert result.actions.shape == (2, 6)
    assert (result.actions == 0.25).all()
    assert result.rewards.tolist() == [0.0, 1.0]
    assert result.interventions.all()


def test_configuration_rejects_incompatible_recording_units(cfg):
    cfg.dataset_use_degrees = False
    with pytest.raises(ValueError, match="recording units"):
        cfg.__post_init__()


def test_renamed_camera_and_unused_vla_slots_support_sampling_and_learning(cfg):
    cfg.rename_map = {IMAGE: "observation.images.camera1"}
    policy = TinyVLA()
    policy.config.image_features = {
        f"observation.images.camera{i}": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)) for i in range(1, 4)
    }
    pre, post = make_policy_processor_pipelines(
        input_steps=[RenameObservationsProcessorStep(rename_map=cfg.rename_map)],
        output_steps=[],
    )
    agent = ExpoFTLearner(cfg, policy, pre, post, image_keys=[IMAGE])
    replay = EpisodeReplay(100)
    replay.add(episode())
    assert agent.critics.encoder.image_keys == ["observation.images.camera1"]
    assert agent.sample_action_chunk(observation()).shape == (2, 6)
    assert "vla_loss" in agent.update(replay)


def test_check_only_does_not_connect_hardware_or_save(cfg, monkeypatch):
    from lerobot.rl.expo_ft import train

    cfg.check_only = True
    cfg.vla_batch_size = 1
    agent = learner(cfg)
    replay = EpisodeReplay(100)
    replay.add(episode())
    names = [f"{m}.pos" for m in MOTORS]
    hardware = SimpleNamespace(action_features=dict.fromkeys(names, float), cameras={"wrist": None})
    metadata = SimpleNamespace(features={ACTION: {"names": names}, OBS_STATE: {"names": names}})
    monkeypatch.setattr(train, "load_learner", lambda cfg: (agent, metadata, replay, 0))
    monkeypatch.setattr(train, "make_robot_from_config", lambda cfg: hardware)
    monkeypatch.setattr(train, "make_teleoperator_from_config", lambda cfg: hardware)
    monkeypatch.setattr(train, "load_demonstrations", lambda *args: None)
    monkeypatch.setattr(train, "save_checkpoint", lambda *args: pytest.fail("check_only must not save"))
    train.train(cfg)  # A connect() attempt would also fail: the hardware stub does not provide it.
    assert agent.updates == 1


def test_example_config_roundtrip():
    path = Path(__file__).parents[2] / "examples/expo_ft_so101/configs/train.json"
    config = draccus.parse(ExpoFTConfig, config_path=path, args=[])
    restored = draccus.decode(ExpoFTConfig, draccus.encode(config))
    assert restored.leader_intervention.follow_follower
    assert restored.num_candidates == 8
    assert restored.robot.use_degrees == restored.dataset_use_degrees


def test_jenga_config_and_startup_poses_keep_separate_leader_home(cfg):
    from lerobot.rl.expo_ft.train import capture_home_poses

    path = Path(__file__).parents[2] / "examples/expo_ft_so101/configs/jenga_smolvla.json"
    config = draccus.parse(ExpoFTConfig, config_path=path, args=[])
    assert config.use_startup_poses and config.reset_pose is None and config.home_pose is None
    names = [f"{m}.pos" for m in MOTORS]
    robot = SimpleNamespace(get_observation=lambda: dict.fromkeys(names, 12.0))
    leader = SimpleNamespace(get_action=lambda: dict.fromkeys(names, -20.0))
    home = capture_home_poses(config, robot, leader, names)
    assert config.reset_pose == config.home_pose == [12.0] * 6
    assert home == dict.fromkeys(names, -20.0)


def test_relative_joint_checkpoint_preserves_chunk_anchor_and_training_normalization(cfg, tmp_path):
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
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre, post = load_pretrained_policy_processors(tmp_path)
    agent = ExpoFTLearner(cfg, TinyVLA(), pre, post)
    # Repeated candidate observations use a B*N anchor; final selection must restore the B anchor.
    current = observation(20)
    candidates = agent.candidates([current])
    agent.prepare([current])
    torch.testing.assert_close(post(candidates[:, 0]), candidates[:, 0] * 10 + 20)
    physical = torch.full((1, 2, 6), 25.0)
    batch = agent.prepare([current], physical)
    torch.testing.assert_close(batch[ACTION], torch.full((1, 2, 6), 0.5))


def test_reset_finishes_gripper_after_arm_and_reaches_target(monkeypatch):
    from lerobot.rl.expo_ft import train

    monkeypatch.setattr(train, "precise_sleep", lambda _: None)
    names = [f"{m}.pos" for m in MOTORS]

    class Robot:
        def __init__(self):
            self.pose = dict.fromkeys(names, 0.0)
            self.commands = []

        def get_observation(self):
            return self.pose.copy()

        def send_action(self, command):
            self.pose = limit_joint_target(command, self.pose, 2.0, {})
            self.commands.append(self.pose.copy())
            return self.pose

    robot = Robot()
    train.move_follower(robot, [9.0] * 6, names, 1.0, {})
    assert robot.pose == dict.fromkeys(names, 9.0)
    assert all(command["gripper.pos"] == 0 for command in robot.commands[:30])


def test_takeover_discards_inflight_chunk_and_replans(cfg, monkeypatch):
    from concurrent.futures import Future

    from lerobot.rl.expo_ft import train

    monkeypatch.setattr(train, "precise_sleep", lambda _: None)
    names = [f"{m}.pos" for m in MOTORS]
    agent = learner(cfg)

    class Robot:
        def __init__(self):
            self.pose = dict.fromkeys(names, 0.0)
            self.commands = []

        def get_observation(self):
            return {**self.pose, "wrist": np.zeros((32, 32, 3), dtype=np.uint8)}

        def send_action(self, command):
            self.pose = dict(command)
            self.commands.append(self.pose)
            return self.pose

    class Worker:
        def __init__(self):
            self.futures = []
            self.observations = []

        def submit(self, function, observation):
            future = Future()
            self.futures.append(future)
            self.observations.append(observation)
            if len(self.futures) == 2:
                future.set_result(torch.full((2, 6), 2.0))
            return future

    worker, robot = Worker(), Robot()

    class Controller:
        phase = train.LeaderPhase.AUTONOMOUS
        ticks = 0

        def reset(self):
            pass

        def __call__(self, transition):
            self.ticks += 1
            if self.ticks == 2:
                self.phase = train.LeaderPhase.CORRECTING
                worker.futures[0].set_result(torch.full((2, 6), -9.0))
            elif self.ticks == 3:
                self.phase = train.LeaderPhase.AUTONOMOUS
            return {
                TransitionKey.INFO: {
                    TeleopEvents.SUCCESS: self.ticks >= 5,
                    TeleopEvents.TERMINATE_EPISODE: False,
                    TeleopEvents.RERECORD_EPISODE: False,
                    TeleopEvents.IS_INTERVENTION: self.phase is train.LeaderPhase.CORRECTING,
                },
                TransitionKey.COMPLEMENTARY_DATA: {INTERVENTION_JOINT_TARGET_KEY: dict.fromkeys(names, 1.0)},
            }

    result = run_episode(cfg, agent, worker, robot, FakeLeader(), Controller(), names, names)
    assert result.success
    assert len(worker.futures) == 2
    assert torch.equal(worker.observations[1][OBS_STATE], torch.ones(6))
    assert [command[names[0]] for command in robot.commands] == [1.0, 2.0, 2.0]


def test_ready_plans_do_not_insert_idle_ticks_between_chunks(cfg, monkeypatch):
    from concurrent.futures import Future

    from lerobot.rl.expo_ft import train

    monkeypatch.setattr(train, "precise_sleep", lambda _: None)
    names = [f"{m}.pos" for m in MOTORS]

    class Robot:
        commands = 0

        def get_observation(self):
            return {**dict.fromkeys(names, 0.0), "wrist": np.zeros((32, 32, 3), dtype=np.uint8)}

        def send_action(self, command):
            self.commands += 1
            return command

    class Worker:
        requests = 0

        def submit(self, function, observation):
            self.requests += 1
            future = Future()
            future.set_result(torch.ones(2, 6))
            return future

    class Controller:
        phase = train.LeaderPhase.AUTONOMOUS
        ticks = 0

        def reset(self):
            pass

        def __call__(self, transition):
            self.ticks += 1
            return {
                TransitionKey.INFO: {
                    TeleopEvents.SUCCESS: self.ticks == 7,
                    TeleopEvents.TERMINATE_EPISODE: False,
                    TeleopEvents.RERECORD_EPISODE: False,
                    TeleopEvents.IS_INTERVENTION: False,
                }
            }

    robot, worker = Robot(), Worker()
    result = run_episode(cfg, learner(cfg), worker, robot, FakeLeader(), Controller(), names, names)
    assert robot.commands == len(result.actions) == 6
    assert worker.requests == 3
    assert result.timing["planning_wait_s"] == 0


def test_gripper_targets_stay_in_calibrated_range():
    for requested, measured, expected in [(-10.0, 5.0, 0.0), (110.0, 98.0, 100.0), (-1.0, -2.0, 0.0)]:
        result = limit_joint_target({"gripper.pos": requested}, {"gripper.pos": measured}, 9.0, {})
        assert result["gripper.pos"] == expected


def test_resume_rejects_changed_execution_horizon_before_loading_weights(cfg, tmp_path):
    import json

    from lerobot.rl.expo_ft.train import load_learner

    cfg.resume = tmp_path
    (tmp_path / "expo_ft_config.json").write_text(json.dumps({"chunk_steps": cfg.chunk_steps + 1}))
    with pytest.raises(ValueError, match="chunk_steps differs"):
        load_learner(cfg)
