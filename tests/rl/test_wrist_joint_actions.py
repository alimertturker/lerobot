#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Policy actions with joint deltas for the wrist: [dx, dy, dz, d_wrist_flex, d_wrist_roll, gripper]."""

import numpy as np
import pytest
import torch

from lerobot.processor import (
    DataProcessorPipeline,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    create_transition,
    identity_transition,
)
from lerobot.utils.import_utils import _placo_available

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = MOTORS[:-1]
WRIST = ("wrist_flex", "wrist_roll")
EE_STEPS = {"x": 0.02, "y": 0.02, "z": 0.02}
JOINT_STEPS = {"wrist_flex": 9.0, "wrist_roll": 9.0}

requires_placo = pytest.mark.skipif(not _placo_available, reason="placo is not installed")


def _link(name):
    inertia = '<inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>'
    return f'<link name="{name}"><inertial><mass value="0.1"/>{inertia}</inertial></link>'


def _joint(name, parent, child, xyz, axis):
    return (
        f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
        f'<origin xyz="{xyz}"/><axis xyz="{axis}"/>'
        '<limit lower="-2.5" upper="2.5" effort="1" velocity="1"/></joint>'
    )


@pytest.fixture
def kinematics(tmp_path):
    """A 5-joint arm shaped like the SO-101: pan, lift, elbow, wrist flex, wrist roll, offset tip."""
    from lerobot.model import RobotKinematics

    urdf = (
        '<robot name="arm">'
        + "".join(_link(n) for n in ["base", "l1", "l2", "l3", "l4", "l5", "tip"])
        + _joint("shoulder_pan", "base", "l1", "0 0 0.05", "0 0 1")
        + _joint("shoulder_lift", "l1", "l2", "0 0 0.05", "0 1 0")
        + _joint("elbow_flex", "l2", "l3", "0.12 0 0", "0 1 0")
        + _joint("wrist_flex", "l3", "l4", "0.12 0 0", "0 1 0")
        + _joint("wrist_roll", "l4", "l5", "0.03 0 0", "1 0 0")
        + '<joint name="tip_joint" type="fixed"><parent link="l5"/><child link="tip"/>'
        '<origin xyz="0.05 0 -0.01"/></joint></robot>'
    )
    path = tmp_path / "arm.urdf"
    path.write_text(urdf)
    return RobotKinematics(str(path), "tip", ARM)


def joints(**overrides):
    q = {"shoulder_pan": 10.0, "shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": -20.0}
    q |= {"wrist_roll": 5.0, "gripper": 30.0}
    q |= overrides
    return {f"{m}.pos": v for m, v in q.items()}


def test_delta_steps_carry_joint_deltas_between_xyz_and_gripper():
    to_dict = MapTensorToDeltaActionDictStep(use_gripper=True, joint_names=WRIST)
    delta = to_dict.action(torch.tensor([0.1, 0.2, 0.3, 0.5, -0.5, 2.0]))
    assert delta == pytest.approx(
        {"delta_x": 0.1, "delta_y": 0.2, "delta_z": 0.3, "delta_wrist_flex": 0.5, "delta_wrist_roll": -0.5}
        | {"gripper": 2.0}
    )

    to_robot = MapDeltaActionToRobotActionStep(joint_names=WRIST)
    robot_action = to_robot.action(dict(delta))
    assert robot_action["joint_delta.wrist_flex"] == pytest.approx(0.5)
    assert robot_action["joint_delta.wrist_roll"] == pytest.approx(-0.5)
    assert robot_action["gripper_vel"] == 2.0
    assert robot_action["enabled"]

    # A wrist-only command is still an enabled command
    wrist_only = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "delta_wrist_flex": 0.0}
    wrist_only |= {"delta_wrist_roll": 0.3, "gripper": 1.0}
    assert to_robot.action(wrist_only)["enabled"]


def test_delta_steps_without_joints_keep_the_xyz_layout():
    delta = MapTensorToDeltaActionDictStep(use_gripper=True).action(torch.tensor([0.1, 0.2, 0.3, 2.0]))
    assert set(delta) == {"delta_x", "delta_y", "delta_z", "gripper"}
    robot_action = MapDeltaActionToRobotActionStep().action(delta)
    assert not any(k.startswith("joint_delta.") for k in robot_action)


@requires_placo
def test_locked_joints_keep_their_value_and_ik_reaches_the_tip(kinematics):
    q = np.array([10.0, -20.0, 40.0, -20.0, 5.0])
    target = kinematics.forward_kinematics(q)
    q_guess = q.copy()
    q_guess[3:] += [9.0, -9.0]
    solution = kinematics.inverse_kinematics(q_guess, target, orientation_weight=0.0, locked_joints=WRIST)
    np.testing.assert_allclose(solution[3:], q_guess[3:], atol=1e-6)
    np.testing.assert_allclose(kinematics.forward_kinematics(solution)[:3, 3], target[:3, 3], atol=1e-5)

    # A locked joint past its URDF limit (2.5 rad here) keeps its value; the real range can be wider
    q_guess[3] = 150.0
    solution = kinematics.inverse_kinematics(
        q_guess, kinematics.forward_kinematics(q_guess), orientation_weight=0.0, locked_joints=WRIST
    )
    assert solution[3] == pytest.approx(150.0)
    np.testing.assert_allclose(
        np.rad2deg(kinematics.robot.get_joint_limits("wrist_flex")), [-143.24, 143.24], atol=0.01
    )

    # The joints are unlocked again afterwards
    free = kinematics.inverse_kinematics(q_guess, target, orientation_weight=0.0)
    assert not np.allclose(free[3:], q_guess[3:])


def _policy_pipeline(kinematics, bounds=None, **limits):
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        EEBoundsAndSafety,
        EEReferenceAndDelta,
        GripperVelocityToJoint,
        InverseKinematicsRLStep,
    )

    steps = [
        MapTensorToDeltaActionDictStep(use_gripper=True, joint_names=WRIST),
        MapDeltaActionToRobotActionStep(joint_names=WRIST),
        EEReferenceAndDelta(
            kinematics=kinematics,
            end_effector_step_sizes=EE_STEPS,
            motor_names=ARM,
            use_latched_reference=False,
            use_ik_solution=True,
        ),
        EEBoundsAndSafety(
            end_effector_bounds=bounds or {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=None if limits else 0.05,
        ),
        GripperVelocityToJoint(clip_max=100.0, speed_factor=1.0, discrete_gripper=True),
        InverseKinematicsRLStep(
            kinematics=kinematics,
            motor_names=MOTORS,
            initial_guess_current_joints=False,
            joint_step_sizes=JOINT_STEPS,
            joint_bounds={"wrist_roll": [-30.0, 30.0]},
            **limits,
        ),
    ]
    return DataProcessorPipeline(
        steps=steps, to_transition=identity_transition, to_output=identity_transition
    )


@requires_placo
def test_recorded_take_over_action_replays_to_the_same_joints(kinematics):
    """`ee_delta_action_from_joints` and the policy pipeline are inverses for wrist moves within one step."""
    from lerobot.rl.leader_intervention import ee_delta_action_from_joints

    follower = joints()
    target = joints(shoulder_lift=-17.0, elbow_flex=37.0, wrist_flex=-14.0, wrist_roll=-2.0)
    action = ee_delta_action_from_joints(
        kinematics, ARM, follower, target, EE_STEPS, True, 3.0, joint_step_sizes=JOINT_STEPS
    )
    assert action.shape == (6,)
    np.testing.assert_allclose(action[3:5], [6.0 / 9.0, -7.0 / 9.0], atol=1e-6)

    transition = create_transition(
        observation=follower, action=torch.from_numpy(action), complementary_data={}
    )
    result = _policy_pipeline(kinematics)(transition)
    for m in ARM:
        assert result["action"][f"{m}.pos"] == pytest.approx(target[f"{m}.pos"], abs=0.05), m


@requires_placo
def test_wrist_only_action_turns_the_wrist_in_place_within_bounds(kinematics):
    follower = joints(wrist_roll=25.0)
    tip = kinematics.forward_kinematics(np.array([follower[f"{m}.pos"] for m in ARM]))[:3, 3]
    action = torch.tensor([0.0, 0.0, 0.0, -1.0, 1.0, 1.0])  # flex -9 deg, roll +9 deg (past the 30 deg bound)

    result = _policy_pipeline(kinematics)(create_transition(observation=follower, action=action))
    commanded = np.array([result["action"][f"{m}.pos"] for m in ARM])
    assert commanded[3] == pytest.approx(-29.0, abs=1e-6)
    assert commanded[4] == pytest.approx(30.0, abs=1e-6)
    np.testing.assert_allclose(kinematics.forward_kinematics(commanded)[:3, 3], tip, atol=1e-5)


def _commanded(result):
    return np.array([result["action"][f"{m}.pos"] for m in ARM])


@requires_placo
def test_step_over_the_joint_limit_is_shrunk_along_the_straight_line(kinematics):
    follower = joints()
    q = np.array([follower[f"{m}.pos"] for m in ARM])
    tip = kinematics.forward_kinematics(q)[:3, 3]
    action = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])  # 2 cm on each axis, wrist +9 / -9 deg

    full = _commanded(_policy_pipeline(kinematics)(create_transition(observation=follower, action=action)))
    assert np.abs(full - q).max() > 3.0

    limited = _policy_pipeline(kinematics, max_joint_step_deg=3.0, max_ik_error_m=0.005)
    commanded = _commanded(limited(create_transition(observation=follower, action=action)))
    assert np.abs(commanded - q).max() <= 3.0 + 1e-6
    # The whole step is scaled by the same factor: wrist and tip move the same fraction of the way
    scale = (commanded[3] - q[3]) / 9.0
    assert scale in (pytest.approx(0.5), pytest.approx(0.25), pytest.approx(0.125))
    assert commanded[4] - q[4] == pytest.approx(-9.0 * scale)
    np.testing.assert_allclose(
        kinematics.forward_kinematics(commanded)[:3, 3],
        tip + scale * np.array([0.02, -0.02, 0.02]),
        atol=1e-4,
    )


@requires_placo
def test_unreachable_target_holds_the_previous_command(kinematics):
    follower = joints()
    q = np.array([follower[f"{m}.pos"] for m in ARM])
    # Steps of 1 m per unit of action: even 1/8 of the move is out of the arm's reach
    pipeline = _policy_pipeline(kinematics, max_joint_step_deg=90.0, max_ik_error_m=0.005)
    pipeline.steps[2].end_effector_step_sizes = {"x": 1.0, "y": 1.0, "z": 1.0}
    action = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

    commanded = _commanded(pipeline(create_transition(observation=follower, action=action)))
    np.testing.assert_allclose(commanded, q, atol=1e-9)


def test_ee_jump_check_can_be_disabled():
    from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety

    pose = {"ee.wx": 0.0, "ee.wy": 0.0, "ee.wz": 0.0}
    bounds = {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.5]}
    checked = EEBoundsAndSafety(end_effector_bounds=bounds)
    unchecked = EEBoundsAndSafety(end_effector_bounds=bounds, max_ee_step_m=None)
    for step in (checked, unchecked):
        step.action({"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.1} | pose)
    with pytest.raises(ValueError, match="EE jump"):
        checked.action({"ee.x": 0.3, "ee.y": 0.0, "ee.z": 0.1} | pose)
    clipped = unchecked.action({"ee.x": 0.3, "ee.y": 0.0, "ee.z": -0.2} | pose)
    assert (clipped["ee.x"], clipped["ee.z"]) == (0.3, 0.0)  # box clipping still applies


@requires_placo
def test_wrist_left_outside_its_bounds_is_not_snapped_back(kinematics):
    """After a take-over the wrist can be past `joint_bounds`; the policy must still be able to move."""
    follower = joints(wrist_roll=50.0)  # bounds in _policy_pipeline: wrist_roll [-30, 30]
    pipeline = _policy_pipeline(kinematics, max_joint_step_deg=9.0, max_ik_error_m=0.005)

    commanded = _commanded(
        pipeline(create_transition(observation=follower, action=torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 1.0])))
    )
    assert commanded[4] == pytest.approx(50.0)  # stays, x still moves
    arm_before = np.array([follower[f"{m}.pos"] for m in ARM[:3]])
    assert np.abs(commanded[:3] - arm_before).max() > 0.1

    pipeline.reset()
    back = torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 1.0])
    assert _commanded(pipeline(create_transition(observation=follower, action=back)))[4] == pytest.approx(
        41.0
    )
    further = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    pipeline.reset()
    assert _commanded(pipeline(create_transition(observation=follower, action=further)))[4] == pytest.approx(
        50.0
    )
