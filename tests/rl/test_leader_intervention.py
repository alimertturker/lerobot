#!/usr/bin/env python

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

import numpy as np
import pytest
import torch

from lerobot.envs.configs import LeaderInterventionConfig
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    AddTeleopActionAsComplimentaryDataStep,
    DataProcessorPipeline,
    InterventionActionProcessorStep,
    create_transition,
    identity_transition,
)
from lerobot.rl.leader_intervention import (
    GRIPPER_DECREASE,
    GRIPPER_INCREASE,
    GRIPPER_STAY,
    INTERVENTION_JOINT_TARGET_KEY,
    LEADER_PHASE_KEY,
    KeyboardEpisodeEvents,
    LeaderArmInterventionStep,
    LeaderJointActionOverrideStep,
    LeaderPhase,
    ee_delta_action_from_joints,
)
from lerobot.teleoperators.utils import TeleopEvents

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STEP_SIZES = {"x": 0.02, "y": 0.02, "z": 0.02}


class FakeKinematics:
    """End-effector position = first three joints * 1 cm/degree."""

    def forward_kinematics(self, q):
        t = np.eye(4)
        t[:3, 3] = np.asarray(q[:3], dtype=float) * 0.01
        return t


class FakeLeader:
    """Actuated leader whose position is set by the test; records torque and feedback calls."""

    def __init__(self):
        self.position = dict.fromkeys((f"{m}.pos" for m in MOTORS), 0.0)
        self.torque = False
        self.feedback: list[dict] = []
        self.is_connected = True

    @property
    def action_features(self):
        return dict.fromkeys(self.position, float)

    @property
    def feedback_features(self):
        return self.action_features

    def get_action(self):
        return dict(self.position)

    def send_feedback(self, feedback):
        self.feedback.append(dict(feedback))

    def enable_torque(self):
        self.torque = True

    def disable_torque(self):
        self.torque = False


class SilentEpisodeEvents(KeyboardEpisodeEvents):
    def __init__(self):
        super().__init__("s", ("q", "esc"), "r")

    def start(self):
        pass


class ResetCounter:
    def __init__(self):
        self.count = 0

    def reset(self):
        self.count += 1


def joints(value=0.0, **overrides):
    out = dict.fromkeys((f"{m}.pos" for m in MOTORS), value)
    out.update({f"{k}.pos": v for k, v in overrides.items()})
    return out


@pytest.fixture
def setup():
    leader = FakeLeader()
    events = SilentEpisodeEvents()
    config = LeaderInterventionConfig(handover_duration_s=0.0, max_joint_speed=90.0)
    step = LeaderArmInterventionStep(
        teleop_device=leader,
        kinematics=FakeKinematics(),
        motor_names=MOTORS,
        end_effector_step_sizes=STEP_SIZES,
        use_gripper=True,
        fps=10,
        config=config,
        episode_events=events,
        home_pose=joints(-90.0),
    )
    resets = ResetCounter()
    pipeline = DataProcessorPipeline(
        steps=[
            AddTeleopActionAsComplimentaryDataStep(teleop_device=leader),
            step,
            InterventionActionProcessorStep(use_gripper=True, terminate_on_success=True),
            LeaderJointActionOverrideStep(motor_names=MOTORS, steps_to_reset=[resets]),
        ],
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    return leader, events, step, pipeline, resets


def run(pipeline, follower, policy_action=(0.1, 0.0, 0.0, 1.0)):
    transition = create_transition(
        observation=follower, action=torch.tensor(policy_action, dtype=torch.float32), complementary_data={}
    )
    return pipeline(transition)


def start_correction(leader, events, pipeline, follower):
    """AUTONOMOUS -> PAUSED -> CORRECTING with the leader sitting on the follower."""
    events.dispatch("space")
    run(pipeline, follower)
    leader.position = dict(follower)
    events.dispatch("tab")
    return run(pipeline, follower)


def test_ee_delta_action_scales_clips_and_maps_gripper():
    follower = joints(gripper=50.0)
    leader = joints(shoulder_pan=1.0, shoulder_lift=-10.0, gripper=60.0)  # +1 cm x, -10 cm y
    action = ee_delta_action_from_joints(FakeKinematics(), MOTORS, follower, leader, STEP_SIZES, True, 3.0)
    np.testing.assert_allclose(action, [0.5, -1.0, 0.0, GRIPPER_INCREASE], atol=1e-6)

    leader = joints(gripper=40.0)
    action = ee_delta_action_from_joints(FakeKinematics(), MOTORS, follower, leader, STEP_SIZES, True, 3.0)
    assert action[-1] == GRIPPER_DECREASE

    leader = joints(gripper=51.0)
    action = ee_delta_action_from_joints(FakeKinematics(), MOTORS, follower, leader, STEP_SIZES, False, 3.0)
    assert action.shape == (3,)

    # The gripper command is relative to the measured follower gripper when given
    leader = joints(gripper=50.0)
    action = ee_delta_action_from_joints(
        FakeKinematics(), MOTORS, follower, leader, STEP_SIZES, True, 3.0, follower_gripper=45.0
    )
    assert action[-1] == GRIPPER_INCREASE


def test_autonomous_passes_policy_action_and_leaves_leader_alone(setup):
    leader, _, step, pipeline, resets = setup
    leader.position = joints(30.0)  # far from the follower: must not matter
    for _ in range(3):
        out = run(pipeline, joints(10.0))
    assert step.phase is LeaderPhase.AUTONOMOUS
    assert not leader.torque
    assert leader.feedback == []
    assert out[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is False
    torch.testing.assert_close(out[TransitionKey.ACTION], torch.tensor([0.1, 0.0, 0.0, 1.0]))
    assert INTERVENTION_JOINT_TARGET_KEY not in out[TransitionKey.COMPLEMENTARY_DATA]
    assert resets.count == 0


def test_pause_holds_follower_and_brings_leader_onto_it(setup):
    leader, events, step, pipeline, resets = setup
    follower = joints(10.0)
    events.dispatch("space")
    out = run(pipeline, follower)

    assert step.phase is LeaderPhase.PAUSED
    assert leader.torque
    assert leader.feedback[-1] == follower
    assert out[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is True
    torch.testing.assert_close(
        out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"], torch.tensor([0.0, 0.0, 0.0, GRIPPER_STAY])
    )
    torch.testing.assert_close(out[TransitionKey.ACTION], torch.full((6,), 10.0))
    np.testing.assert_allclose(out[TransitionKey.COMPLEMENTARY_DATA]["IK_solution"], [10.0] * 6)
    assert resets.count == 1


def test_pause_records_zero_motion_even_if_follower_sags(setup):
    leader, events, step, pipeline, _ = setup
    follower = joints(10.0)
    events.dispatch("space")
    run(pipeline, follower)
    sagged = joints(10.0, shoulder_lift=9.0)  # the follower settles below its hold target
    out = run(pipeline, sagged)
    torch.testing.assert_close(
        out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"], torch.tensor([0.0, 0.0, 0.0, GRIPPER_STAY])
    )
    torch.testing.assert_close(out[TransitionKey.ACTION], torch.full((6,), 10.0))


def test_correction_releases_leader_and_follower_tracks_it_at_limited_speed(setup):
    leader, events, step, pipeline, _ = setup
    follower = joints(0.0, gripper=50.0)
    start_correction(leader, events, pipeline, follower)
    assert step.phase is LeaderPhase.CORRECTING
    assert not leader.torque

    leader.position = joints(0.0, shoulder_pan=20.0, gripper=50.0)
    out = run(pipeline, follower)
    # 90 deg/s at 10 fps -> at most 9 degrees per step
    expected = [9.0, 0.0, 0.0, 0.0, 0.0, 50.0]
    torch.testing.assert_close(out[TransitionKey.ACTION], torch.tensor(expected))
    # 9 cm in x -> clipped to the max step; the recorded label is non-zero
    torch.testing.assert_close(
        out[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"], torch.tensor([1.0, 0.0, 0.0, GRIPPER_STAY])
    )
    out = run(pipeline, follower)
    torch.testing.assert_close(out[TransitionKey.ACTION][0], torch.tensor(18.0))
    out = run(pipeline, follower)
    torch.testing.assert_close(out[TransitionKey.ACTION][0], torch.tensor(20.0))


def test_stop_correction_holds_leader_then_resume_parks_it(setup):
    leader, events, step, pipeline, _ = setup
    follower = joints(0.0)
    start_correction(leader, events, pipeline, follower)
    leader.position = joints(0.0, elbow_flex=5.0)

    events.dispatch("tab")
    out = run(pipeline, follower)
    assert step.phase is LeaderPhase.PAUSED
    assert leader.torque
    assert leader.feedback[-1] == joints(0.0, elbow_flex=5.0)
    assert out[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is True

    events.dispatch("space")
    out = run(pipeline, follower)
    assert step.phase is LeaderPhase.AUTONOMOUS
    assert leader.feedback[-1] == joints(-90.0)  # parked at home
    assert not leader.torque
    assert out[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is False
    torch.testing.assert_close(out[TransitionKey.ACTION], torch.tensor([0.1, 0.0, 0.0, 1.0]))


def test_invalid_switches_are_ignored(setup):
    leader, events, step, pipeline, _ = setup
    events.dispatch("tab")  # cannot start a correction without pausing first
    run(pipeline, joints(0.0))
    assert step.phase is LeaderPhase.AUTONOMOUS
    assert not leader.torque

    start_correction(leader, events, pipeline, joints(0.0))
    events.dispatch("space")  # cannot resume while correcting
    run(pipeline, joints(0.0))
    assert step.phase is LeaderPhase.CORRECTING
    assert not leader.torque


@pytest.mark.parametrize(
    "key, done, reward, rerecord",
    [("s", True, 1.0, False), ("q", True, 0.0, False), ("esc", True, 0.0, False), ("r", True, 0.0, True)],
)
def test_episode_keys(setup, key, done, reward, rerecord):
    _, events, _, pipeline, _ = setup
    run(pipeline, joints(0.0))
    events.dispatch(key)
    out = run(pipeline, joints(0.0))
    assert out[TransitionKey.DONE] is done
    assert out[TransitionKey.REWARD] == reward
    assert out[TransitionKey.INFO][TeleopEvents.RERECORD_EPISODE] is rerecord
    # events are consumed
    out = run(pipeline, joints(0.0))
    assert out[TransitionKey.DONE] is False


def test_movement_keys_are_ignored(setup):
    leader, events, step, pipeline, _ = setup
    run(pipeline, joints(0.0))
    for key in ("up", "left", "w"):
        events.dispatch(key)
    out = run(pipeline, joints(0.0))
    assert out[TransitionKey.DONE] is False
    assert step.phase is LeaderPhase.AUTONOMOUS
    assert leader.feedback == []


@pytest.mark.parametrize("key", ["s", "q"])
def test_episode_end_while_correcting_holds_leader(setup, key):
    leader, events, step, pipeline, _ = setup
    start_correction(leader, events, pipeline, joints(0.0))
    leader.position = joints(3.0)
    events.dispatch(key)
    out = run(pipeline, joints(0.0))
    assert out[TransitionKey.DONE] is True
    assert leader.torque
    assert leader.feedback[-1] == joints(3.0)
    assert step.phase is LeaderPhase.PAUSED


def test_reset_hands_back_to_policy_and_parks_leader(setup):
    leader, events, step, pipeline, _ = setup
    start_correction(leader, events, pipeline, joints(0.0))
    pipeline.reset()
    assert step.phase is LeaderPhase.AUTONOMOUS
    assert leader.feedback[-1] == joints(-90.0)
    assert not leader.torque
    out = run(pipeline, joints(5.0))
    assert out[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is False


def test_start_paused_brings_leader_onto_follower_every_episode(setup):
    leader, events, step, pipeline, _ = setup
    step.config.start_paused = True
    pipeline.reset()

    out = run(pipeline, joints(10.0))
    assert step.phase is LeaderPhase.PAUSED
    assert leader.feedback[-1] == joints(10.0)
    assert out[TransitionKey.INFO][LEADER_PHASE_KEY] == "paused"

    leader.position = joints(10.0)
    events.dispatch("tab")
    out = run(pipeline, joints(10.0))
    assert out[TransitionKey.INFO][LEADER_PHASE_KEY] == "correcting"

    # Next episode: the leader is held (not parked) during the reset, then moved onto the new follower pose
    leader.position = joints(12.0)
    pipeline.reset()
    assert step.phase is LeaderPhase.AUTONOMOUS
    assert leader.torque
    assert joints(-90.0) not in leader.feedback
    run(pipeline, joints(0.0))
    assert step.phase is LeaderPhase.PAUSED
    assert leader.feedback[-1] == joints(0.0)


def test_close_moves_leader_home_once_then_releases_torque(setup):
    leader, _, step, pipeline, _ = setup
    run(pipeline, joints(10.0))

    step.close()
    assert leader.feedback[-1] == joints(-90.0)
    assert not leader.torque

    leader.feedback.clear()
    step.close()  # second call (e.g. from atexit) does nothing
    assert leader.feedback == []


def test_rejects_non_actuated_or_mismatched_leader():
    class NoFeedback(FakeLeader):
        @property
        def feedback_features(self):
            return {}

    kwargs = {
        "kinematics": FakeKinematics(),
        "end_effector_step_sizes": STEP_SIZES,
        "episode_events": SilentEpisodeEvents(),
    }
    with pytest.raises(TypeError, match="actuated leader"):
        LeaderArmInterventionStep(teleop_device=NoFeedback(), motor_names=MOTORS, **kwargs)
    with pytest.raises(ValueError, match="do not match"):
        LeaderArmInterventionStep(teleop_device=FakeLeader(), motor_names=[*MOTORS, "extra"], **kwargs)


def test_leader_phase_reaches_the_control_loop_transition(setup):
    from lerobot.rl.gym_manipulator import step_env_and_process_transition

    class StubEnv:
        def get_raw_joint_positions(self):
            return joints(0.0)

        def step(self, action):
            return {}, 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False}

    leader, events, _, pipeline, _ = setup
    identity = DataProcessorPipeline(
        steps=[], to_transition=identity_transition, to_output=identity_transition
    )
    follower = joints(0.0)
    start_correction(leader, events, pipeline, follower)
    out = step_env_and_process_transition(
        env=StubEnv(),
        transition=create_transition(observation=follower, complementary_data={}),
        action=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        env_processor=identity,
        action_processor=pipeline,
    )
    assert out[TransitionKey.INFO][LEADER_PHASE_KEY] == "correcting"
    assert out[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is True
