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

"""Human interventions with an actuated leader arm for HIL-SERL (`control_mode: "leader"`).

The policy acts in end-effector delta space (`[delta_x, delta_y, delta_z, gripper]`). A leader arm
outputs joint positions, so two processor steps bridge the two:

* :class:`LeaderArmInterventionStep` runs at the start of the action pipeline. Like the DAgger rollout
  strategy, interventions are switched with explicit keys and the leader is only released while the
  operator holds it:

      AUTONOMOUS --pause--> PAUSED --correction--> CORRECTING --correction--> PAUSED --pause--> AUTONOMOUS

  In AUTONOMOUS the policy drives the follower and the leader is parked (torque off at its home pose).
  Entering PAUSED holds the follower still and smoothly drives the leader onto the follower's pose, where
  it holds with torque. In CORRECTING the leader's torque is off and the follower copies it joint by joint
  at a limited speed. During PAUSED and CORRECTING the step records, as the intervention action, the
  end-effector delta of the commanded move expressed in the policy's action units. Episode-ending events
  (success / failure / re-record) also come from keys.
* :class:`LeaderJointActionOverrideStep` runs at the end of the action pipeline. During an intervention
  it replaces the IK joint targets with the step's joint targets and resets the stateful IK steps so the
  policy resumes from the follower's new pose.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.common.control_utils import teleop_smooth_move_to, teleop_supports_feedback
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.envs.configs import LeaderInterventionConfig
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.keyboard_input import create_key_listener

if TYPE_CHECKING:
    from lerobot.model import RobotKinematics
    from lerobot.teleoperators.teleoperator import Teleoperator

logger = logging.getLogger(__name__)

# Complementary-data key carrying the follower joint targets while an intervention is active.
INTERVENTION_JOINT_TARGET_KEY = "intervention_joint_target"
# Complementary-data key written by `InverseKinematicsRLStep` and read by `EEReferenceAndDelta`.
IK_SOLUTION_KEY = "IK_solution"
# Info key carrying the current `LeaderPhase` value.
LEADER_PHASE_KEY = "leader_phase"

# Discrete gripper commands, as interpreted by `GripperVelocityToJoint(discrete_gripper=True)`:
# 0 increases the gripper joint position, 2 decreases it, 1 keeps it.
GRIPPER_INCREASE = 0.0
GRIPPER_STAY = 1.0
GRIPPER_DECREASE = 2.0

LEADER_PARK_DURATION_S = 2.0


class LeaderPhase(Enum):
    AUTONOMOUS = "autonomous"  # Policy in control, leader parked with torque off
    PAUSED = "paused"  # Follower holds still, leader holds the follower's pose with torque
    CORRECTING = "correcting"  # Leader torque off, follower copies the leader


# Valid (phase, key event) -> next phase. Any other combination is ignored.
_TRANSITIONS: dict[tuple[LeaderPhase, str], LeaderPhase] = {
    (LeaderPhase.AUTONOMOUS, "pause"): LeaderPhase.PAUSED,
    (LeaderPhase.PAUSED, "pause"): LeaderPhase.AUTONOMOUS,
    (LeaderPhase.PAUSED, "correction"): LeaderPhase.CORRECTING,
    (LeaderPhase.CORRECTING, "correction"): LeaderPhase.PAUSED,
}


class KeyboardEpisodeEvents:
    """Collects key presses from a background listener: intervention mode switches and episode endings.

    Keys only switch modes or end episodes; movement keys are ignored, so the keyboard never moves the
    robot.
    """

    def __init__(
        self,
        success_key: str,
        failure_keys: tuple[str, ...],
        rerecord_key: str,
        pause_key: str = "space",
        correction_key: str = "tab",
    ):
        self.success_key = success_key
        self.failure_keys = tuple(failure_keys)
        self.rerecord_key = rerecord_key
        self.pause_key = pause_key
        self.correction_key = correction_key
        self._lock = threading.Lock()
        self._pending: dict[TeleopEvents, bool] = {}
        self._mode_events: list[str] = []
        self._listener: Any = None

    def start(self) -> None:
        if self._listener is not None:
            return
        failure = "/".join(f"'{k}'" for k in self.failure_keys)
        self._listener = create_key_listener(
            self.dispatch,
            controls_help=(
                f"'{self.pause_key}'=pause/resume policy, '{self.correction_key}'=start/stop correction, "
                f"'{self.success_key}'=success, {failure}=failure, '{self.rerecord_key}'=re-record"
            ),
        )

    def stop(self) -> None:
        if self._listener is not None:
            with contextlib.suppress(Exception):
                self._listener.stop()
            self._listener = None

    def dispatch(self, name: str) -> None:
        with self._lock:
            if name == self.pause_key:
                self._mode_events.append("pause")
            elif name == self.correction_key:
                self._mode_events.append("correction")
            elif name == self.success_key:
                self._pending[TeleopEvents.SUCCESS] = True
            elif name in self.failure_keys:
                self._pending[TeleopEvents.TERMINATE_EPISODE] = True
            elif name == self.rerecord_key:
                self._pending[TeleopEvents.TERMINATE_EPISODE] = True
                self._pending[TeleopEvents.RERECORD_EPISODE] = True

    def pop(self) -> dict[TeleopEvents, bool]:
        """Return the episode events received since the last call and clear them."""
        with self._lock:
            events = {
                TeleopEvents.SUCCESS: self._pending.get(TeleopEvents.SUCCESS, False),
                TeleopEvents.TERMINATE_EPISODE: self._pending.get(TeleopEvents.TERMINATE_EPISODE, False),
                TeleopEvents.RERECORD_EPISODE: self._pending.get(TeleopEvents.RERECORD_EPISODE, False),
            }
            self._pending.clear()
        return events

    def pop_mode_event(self) -> str | None:
        """Return the oldest pending mode switch ("pause" or "correction"), one per control step."""
        with self._lock:
            return self._mode_events.pop(0) if self._mode_events else None

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._mode_events.clear()


def ee_delta_action_from_joints(
    kinematics: RobotKinematics,
    motor_names: list[str],
    reference_joints: dict[str, float],
    target_joints: dict[str, float],
    end_effector_step_sizes: dict[str, float],
    use_gripper: bool,
    gripper_threshold: float,
    follower_gripper: float | None = None,
) -> np.ndarray:
    """Express a commanded move of the follower as a policy action.

    The xyz part is the end-effector displacement from `reference_joints` to `target_joints` divided by
    the step sizes and clipped to [-1, 1]. With the previous joint command as reference this is the delta
    the policy would output for `EEReferenceAndDelta` (which works from the previous IK solution, not from
    the measured pose) to reach the same target. The gripper part is the discrete command that moves the
    follower gripper (`follower_gripper`, default the reference's) towards the target gripper; like
    `GripperVelocityToJoint`, it is relative to the measured gripper position.
    """
    q_reference = np.array([reference_joints[f"{m}.pos"] for m in motor_names], dtype=float)
    q_target = np.array([target_joints[f"{m}.pos"] for m in motor_names], dtype=float)
    p_reference = kinematics.forward_kinematics(q_reference)[:3, 3]
    p_target = kinematics.forward_kinematics(q_target)[:3, 3]
    steps = np.array([end_effector_step_sizes[axis] for axis in ("x", "y", "z")], dtype=float)
    delta = np.clip((p_target - p_reference) / steps, -1.0, 1.0)

    action = delta.tolist()
    if use_gripper:
        current = reference_joints["gripper.pos"] if follower_gripper is None else follower_gripper
        gripper_diff = target_joints["gripper.pos"] - current
        if gripper_diff > gripper_threshold:
            action.append(GRIPPER_INCREASE)
        elif gripper_diff < -gripper_threshold:
            action.append(GRIPPER_DECREASE)
        else:
            action.append(GRIPPER_STAY)
    return np.asarray(action, dtype=np.float32)


@ProcessorStepRegistry.register("leader_arm_intervention")
@dataclass
class LeaderArmInterventionStep(ProcessorStep):
    """Key-switched leader-arm interventions; turns the commanded follower move into a policy action.

    Expects the leader's joint positions in `complementary_data["teleop_action"]` (written by
    `AddTeleopActionAsComplimentaryDataStep`) and the follower's joint positions as the observation.
    During an intervention (PAUSED or CORRECTING) it replaces `complementary_data["teleop_action"]` with
    the end-effector delta action between consecutive joint targets and puts the follower joint targets in
    `complementary_data["intervention_joint_target"]`; otherwise it sets `teleop_action` to None so the
    policy action passes through. Also writes `is_intervention` and the episode events to `info`.

    Attributes:
        teleop_device: Actuated leader arm (position feedback and torque control).
        kinematics: Kinematic model of the follower, used for forward kinematics.
        motor_names: Follower motor names, in joint order.
        end_effector_step_sizes: Metres per unit of policy action along x, y and z.
        use_gripper: Whether the policy action includes the discrete gripper command.
        fps: Control frequency, used to turn `max_joint_speed` into a per-step limit.
        config: Keys, handover duration and speed limit.
        episode_events: Source of key events; a keyboard listener by default.
        home_pose: Leader joint positions (`{motor}.pos`) where the leader is parked (torque off) while the
            policy runs and when closing. If None, the leader keeps holding its last pose with torque.
    """

    teleop_device: Teleoperator
    kinematics: RobotKinematics
    motor_names: list[str]
    end_effector_step_sizes: dict[str, float]
    use_gripper: bool = True
    fps: int = 10
    config: LeaderInterventionConfig = field(default_factory=LeaderInterventionConfig)
    episode_events: KeyboardEpisodeEvents | None = None
    home_pose: dict[str, float] | None = None

    phase: LeaderPhase = field(default=LeaderPhase.AUTONOMOUS, init=False)
    _target: dict[str, float] | None = field(default=None, init=False, repr=False)
    _torque_enabled: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _pause_pending: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not teleop_supports_feedback(self.teleop_device):
            raise TypeError(
                f'control_mode "leader" needs an actuated leader arm with position feedback and torque '
                f"control; {type(self.teleop_device).__name__} does not provide them."
            )
        missing = {f"{m}.pos" for m in self.motor_names} - set(self.teleop_device.action_features)
        if missing:
            raise ValueError(
                f"Leader arm joints do not match the follower: leader is missing {sorted(missing)}. "
                f"Leader joints: {sorted(self.teleop_device.action_features)}."
            )
        if self.episode_events is None:
            self.episode_events = KeyboardEpisodeEvents(
                self.config.success_key,
                self.config.failure_keys,
                self.config.rerecord_key,
                self.config.pause_key,
                self.config.correction_key,
            )
        self.episode_events.start()
        self._max_step = self.config.max_joint_speed / self.fps
        self._pause_pending = self.config.start_paused
        atexit.register(self.close)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        assert self.episode_events is not None
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION) or {}
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary_data.pop(INTERVENTION_JOINT_TARGET_KEY, None)

        follower = {f"{m}.pos": float(observation[f"{m}.pos"]) for m in self.motor_names}
        leader = complementary_data.get(TELEOP_ACTION_KEY)
        if not isinstance(leader, dict):
            raise ValueError("LeaderArmInterventionStep expects the leader joint positions as teleop_action")
        leader = {f"{m}.pos": float(leader[f"{m}.pos"]) for m in self.motor_names}

        if self._pause_pending:
            self._pause_pending = False
            if self.phase is LeaderPhase.AUTONOMOUS:
                self._switch("pause", follower, leader)

        mode_event = self.episode_events.pop_mode_event()
        if mode_event is not None:
            self._switch(mode_event, follower, leader)

        episode_events = self.episode_events.pop()
        episode_ends = episode_events[TeleopEvents.TERMINATE_EPISODE] or episode_events[TeleopEvents.SUCCESS]
        if self.phase is LeaderPhase.CORRECTING and episode_ends:
            # Hold the leader where the operator left it before the environment resets.
            self._hold_leader_at(leader)
            self.phase = LeaderPhase.PAUSED

        is_intervention = self.phase is not LeaderPhase.AUTONOMOUS
        if is_intervention:
            assert self._target is not None
            previous_target = self._target
            if self.phase is LeaderPhase.CORRECTING:
                self._target = self._rate_limited(previous_target, leader)
            complementary_data[TELEOP_ACTION_KEY] = ee_delta_action_from_joints(
                self.kinematics,
                self.motor_names,
                previous_target,
                self._target,
                self.end_effector_step_sizes,
                self.use_gripper,
                self.config.gripper_threshold,
                follower_gripper=follower["gripper.pos"],
            )
            complementary_data[INTERVENTION_JOINT_TARGET_KEY] = dict(self._target)
        else:
            complementary_data[TELEOP_ACTION_KEY] = None
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        info = dict(new_transition.get(TransitionKey.INFO) or {})
        info.update(episode_events)
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[LEADER_PHASE_KEY] = self.phase.value
        new_transition[TransitionKey.INFO] = info
        return new_transition

    def _switch(self, event: str, follower: dict[str, float], leader: dict[str, float]) -> None:
        new_phase = _TRANSITIONS.get((self.phase, event))
        if new_phase is None:
            key = self.config.pause_key if event == "pause" else self.config.correction_key
            logger.info("Ignoring '%s' while %s", key, self.phase.value)
            return
        logger.info("Intervention phase: %s -> %s", self.phase.value, new_phase.value)

        if self.phase is LeaderPhase.AUTONOMOUS:
            # Hold the follower where it is and bring the leader onto it.
            self._target = dict(follower)
            logger.info(
                "Moving the leader onto the follower; hold it and press '%s' to take over",
                self.config.correction_key,
            )
            teleop_smooth_move_to(
                self.teleop_device,
                follower,
                duration_s=self.config.handover_duration_s,
                fps=max(self.fps, 30),
            )
            self._torque_enabled = True
        elif new_phase is LeaderPhase.CORRECTING:
            # The leader sits on the follower's pose, so releasing it starts the correction without a jump.
            self.teleop_device.disable_torque()
            self._torque_enabled = False
        elif self.phase is LeaderPhase.CORRECTING:
            self._hold_leader_at(leader)
        else:  # PAUSED -> AUTONOMOUS
            self._park_leader()
            self._target = None
        self.phase = new_phase

    def _rate_limited(self, previous: dict[str, float], desired: dict[str, float]) -> dict[str, float]:
        return {
            k: previous[k] + float(np.clip(desired[k] - previous[k], -self._max_step, self._max_step))
            for k in previous
        }

    def _hold_leader_at(self, leader: dict[str, float]) -> None:
        self.teleop_device.send_feedback(leader)
        self.teleop_device.enable_torque()
        self._torque_enabled = True

    def _park_leader(self) -> None:
        """Move the leader to its home pose and release it there; without a home pose keep it holding."""
        if self.home_pose is None or not self._torque_enabled:
            return
        logger.info("Parking the leader at its home pose")
        teleop_smooth_move_to(self.teleop_device, self.home_pose, duration_s=LEADER_PARK_DURATION_S, fps=30)
        self.teleop_device.disable_torque()
        self._torque_enabled = False

    def reset(self) -> None:
        """Make the leader safe before the environment reset and set up the next episode.

        The leader is held if it was released. With `start_paused` it keeps holding and is moved onto the
        follower at the first step of the next episode; otherwise it is parked and the policy takes over.
        """
        if self.phase is not LeaderPhase.AUTONOMOUS:
            if self.phase is LeaderPhase.CORRECTING:
                self._hold_leader_at(self.teleop_device.get_action())
            if not self.config.start_paused:
                self._park_leader()
            self.phase = LeaderPhase.AUTONOMOUS
            self._target = None
        self._pause_pending = self.config.start_paused
        if self.episode_events is not None:
            self.episode_events.clear()

    def close(self) -> None:
        """Park the leader at its home pose, release its torque and stop the keyboard listener.

        Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True
        if self.episode_events is not None:
            self.episode_events.stop()
        if not getattr(self.teleop_device, "is_connected", True):
            return
        with contextlib.suppress(Exception):
            if self.home_pose is not None:
                logger.info("Moving the leader to its home pose")
                teleop_smooth_move_to(
                    self.teleop_device, self.home_pose, duration_s=LEADER_PARK_DURATION_S, fps=30
                )
                self._torque_enabled = True
        if self._torque_enabled:
            with contextlib.suppress(Exception):
                self.teleop_device.disable_torque()
            self._torque_enabled = False

    def get_config(self) -> dict[str, Any]:
        return {
            "motor_names": self.motor_names,
            "end_effector_step_sizes": self.end_effector_step_sizes,
            "use_gripper": self.use_gripper,
            "fps": self.fps,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("leader_joint_action_override")
@dataclass
class LeaderJointActionOverrideStep(ProcessorStep):
    """Sends the intervention joint targets to the follower during a leader-arm intervention.

    Placed after the IK steps: while an intervention is active the joint-target action they produced is
    replaced by the targets from :class:`LeaderArmInterventionStep`, `IK_solution` is set to them, and
    `steps_to_reset` (the stateful `EEReferenceAndDelta`, `EEBoundsAndSafety` and
    `InverseKinematicsRLStep`) are reset so that, once control returns to the policy, they start from the
    follower's new pose instead of the pose held before the intervention.
    """

    motor_names: list[str]
    steps_to_reset: list[ProcessorStep] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        target = complementary_data.get(INTERVENTION_JOINT_TARGET_KEY)
        if target is None:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        joints = [float(target[f"{m}.pos"]) for m in self.motor_names]
        if isinstance(action, torch.Tensor):
            new_transition[TransitionKey.ACTION] = torch.tensor(
                joints, dtype=action.dtype, device=action.device
            )
        else:
            new_transition[TransitionKey.ACTION] = torch.tensor(joints, dtype=torch.float32)

        new_complementary_data = dict(complementary_data)
        new_complementary_data[IK_SOLUTION_KEY] = np.asarray(joints, dtype=float)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data

        for step in self.steps_to_reset:
            step.reset()
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"motor_names": self.motor_names}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
