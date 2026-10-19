# !/usr/bin/env python

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

import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.envs import HILSerlRobotEnvConfig
from lerobot.envs.configs import LeaderInterventionConfig
from lerobot.model import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    GymHILAdapterProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    Numpy2TorchActionProcessorStep,
    ProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    RobotObservation,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
    identity_transition,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so_leader,  # noqa: F401
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.import_utils import require_package
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

from .joint_observations_processor import JointVelocityProcessorStep, MotorCurrentProcessorStep
from .leader_intervention import (
    LEADER_PHASE_KEY,
    LeaderArmInterventionStep,
    LeaderJointActionOverrideStep,
    LeaderPhase,
)

logging.basicConfig(level=logging.INFO)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def reset_follower_position(
    robot_arm: Robot, target_position: np.ndarray, max_joint_speed: float = 60.0
) -> None:
    """Reset robot arm to target position using smooth trajectory.

    The trajectory has at least 50 waypoints and enough of them to keep every joint under
    `max_joint_speed` (position units per second, degrees for most arms).
    """
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )
    max_distance = float(np.max(np.abs(np.asarray(target_position) - current_position)))
    num_steps = max(50, int(np.ceil(max_distance / (max_joint_speed * 0.015))))
    trajectory = torch.from_numpy(np.linspace(current_position, target_position, num_steps))
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        precise_sleep(0.015)


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
        home_pose: list[float] | None = None,
        release_wait_s: float = 0.0,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
            home_pose: Joint positions the robot moves to in `close`, before disconnecting.
            release_wait_s: Time the robot holds still after an episode, before moving to `reset_pose`.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s
        self.home_pose = home_pose
        self.release_wait_s = release_wait_s

        self.use_gripper = use_gripper

        self._joint_names = list(self.robot.bus.motors.keys())
        self._raw_joint_positions = None

        self._setup_spaces()

    def _get_observation(self) -> RobotObservation:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}

        return {"agent_pos": joint_positions, "pixels": images, **raw_joint_joint_position}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=0,
                high=10,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        # Reset the robot
        # self.robot.reset()
        if self.current_step > 0 and self.release_wait_s > 0:
            logging.info(f"Holding for {self.release_wait_s:.1f}s before the reset")
            precise_sleep(self.release_wait_s)
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            target = np.array(self.reset_pose, dtype=np.float32)
            motors = list(self.robot.bus.motors)
            if self.use_gripper and "gripper" in motors:
                # Move the arm with the gripper still at its commanded position, then open it at the reset
                # pose, so an object held at the end of the episode is put down there instead of thrown.
                arm_first = target.copy()
                gripper_goal = self.robot.bus.sync_read("Goal_Position")["gripper"]
                arm_first[motors.index("gripper")] = gripper_goal
                reset_follower_position(self.robot, arm_first)
            reset_follower_position(self.robot, target)
            log_say("Reset the environment done.", play_sounds=True)

        precise_sleep(max(self.reset_time_s - (time.perf_counter() - start_time), 0.0))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}

        self.robot.send_action(joint_targets_dict)

        obs = self._get_observation()

        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self) -> None:
        """Display robot camera feeds."""
        import cv2

        current_observation = self._get_observation()
        if current_observation is not None:
            image_keys = [key for key in current_observation if "image" in key]

            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def close(self) -> None:
        """Move the robot to `home_pose` (if set) and disconnect it."""
        if self.robot.is_connected:
            if self.home_pose is not None:
                logging.info("Moving the robot to its home pose")
                reset_follower_position(self.robot, np.array(self.home_pose))
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions


def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        require_package("gym-hil", extra="hilserl", import_name="gym_hil")
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_cfg = cfg.processor.reset
    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_cfg.fixed_reset_joint_positions if reset_cfg is not None else None,
        reset_time_s=reset_cfg.reset_time_s if reset_cfg is not None else 5.0,
        home_pose=reset_cfg.home_joint_positions if reset_cfg is not None else None,
        release_wait_s=reset_cfg.release_wait_s if reset_cfg is not None else 0.0,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            GymHILAdapterProcessorStep(),
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
        ]

        # Add time limit processor if reset config exists
        if cfg.processor.reset is not None:
            env_pipeline_steps.append(
                TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
            )

        env_pipeline_steps.extend(
            [
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=device),
            ]
        )

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    # Get robot and motor information for kinematics
    motor_names = list(env.robot.bus.motors.keys())

    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names,
        )

    env_pipeline_steps = [VanillaObservationProcessorStep()]

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))

    add_ee_pose = (
        cfg.processor.observation is not None and cfg.processor.observation.add_ee_pose_to_observation
    )
    if kinematics_solver is not None and add_ee_pose:
        env_pipeline_steps.append(
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            )
        )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    # Only add if max_gripper_pos is explicitly configured (required for normalization)
    if (
        cfg.processor.gripper is not None
        and cfg.processor.gripper.use_gripper
        and cfg.processor.max_gripper_pos is not None
    ):
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
    leader_mode = cfg.processor.control_mode == "leader"
    if leader_mode:
        if kinematics_solver is None or cfg.processor.inverse_kinematics is None:
            raise ValueError(
                'control_mode "leader" requires `processor.inverse_kinematics` (urdf_path, target_frame_name, '
                "end_effector_step_sizes, end_effector_bounds): leader moves are recorded as end-effector deltas."
            )
        if (
            cfg.processor.inverse_kinematics.end_effector_step_sizes is None
            or cfg.processor.inverse_kinematics.end_effector_bounds is None
        ):
            raise ValueError(
                'control_mode "leader" requires `inverse_kinematics.end_effector_step_sizes` and '
                "`inverse_kinematics.end_effector_bounds` (find them with `lerobot-find-joint-limits`)."
            )
        if teleop_device is None:
            raise ValueError('control_mode "leader" requires a `teleop` leader arm.')
        home_pose = cfg.processor.reset.home_joint_positions if cfg.processor.reset is not None else None
        teleop_events_step: ProcessorStep = LeaderArmInterventionStep(
            teleop_device=teleop_device,
            kinematics=kinematics_solver,
            motor_names=motor_names,
            end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
            use_gripper=use_gripper,
            fps=cfg.fps,
            config=cfg.processor.leader_intervention or LeaderInterventionConfig(),
            home_pose=dict(zip([f"{m}.pos" for m in motor_names], home_pose, strict=True))
            if home_pose is not None
            else None,
        )
    else:
        teleop_events_step = AddTeleopEventsAsInfoStep(teleop_device=teleop_device)

    action_pipeline_steps: list[ProcessorStep] = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        teleop_events_step,
        InterventionActionProcessorStep(
            use_gripper=use_gripper,
            terminate_on_success=terminate_on_success,
        ),
    ]

    # Replace InverseKinematicsProcessor with new kinematic processors
    if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
        # Add EE bounds and safety processor
        ee_reference_step = EEReferenceAndDelta(
            kinematics=kinematics_solver,
            end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
            motor_names=motor_names,
            use_latched_reference=False,
            use_ik_solution=True,
        )
        ee_bounds_step = EEBoundsAndSafety(
            end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
        )
        ik_step = InverseKinematicsRLStep(
            kinematics=kinematics_solver, motor_names=motor_names, initial_guess_current_joints=False
        )
        inverse_kinematics_steps = [
            MapTensorToDeltaActionDictStep(use_gripper=use_gripper),
            MapDeltaActionToRobotActionStep(),
            ee_reference_step,
            ee_bounds_step,
            GripperVelocityToJoint(
                clip_max=cfg.processor.max_gripper_pos,
                speed_factor=1.0,
                discrete_gripper=True,
            ),
            ik_step,
        ]
        action_pipeline_steps.extend(inverse_kinematics_steps)
        action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))
        if leader_mode:
            # During an intervention the follower copies the leader joint-to-joint instead of the IK target.
            action_pipeline_steps.append(
                LeaderJointActionOverrideStep(
                    motor_names=motor_names, steps_to_reset=[ee_reference_step, ee_bounds_step, ik_step]
                )
            )

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()

    if hasattr(env, "get_raw_joint_positions"):
        raw_joint_positions = env.get_raw_joint_positions()
        if raw_joint_positions is not None:
            complementary_data["raw_joint_positions"] = raw_joint_positions

    # Merge env and action-processor info: env wins for str keys, action-processor
    # wins for `TeleopEvents` enum keys and the leader intervention phase
    action_info = processed_action_transition[TransitionKey.INFO]
    new_info = info.copy()
    for key, value in action_info.items():
        if isinstance(key, TeleopEvents) or key == LEADER_PHASE_KEY:
            new_info[key] = value

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition


def reset_and_build_transition(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """Reset env + processors and return the first env-processed transition.

    Processors are reset first so that stateful steps holding hardware (e.g. a leader arm released during
    an intervention) are made safe before the environment moves the robot to its reset pose.
    """
    env_processor.reset()
    action_processor.reset()
    obs, info = env.reset()
    complementary_data: dict[str, Any] = {}
    if hasattr(env, "get_raw_joint_positions"):
        raw_joint_positions = env.get_raw_joint_positions()
        if raw_joint_positions is not None:
            complementary_data["raw_joint_positions"] = raw_joint_positions
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    return env_processor(data=transition)


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     env_processor: Environment processor
     action_processor: Action processor
     teleop_device: Teleoperator device
     cfg: gym_manipulator configuration
    """
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    if cfg.env.processor.control_mode == "leader":
        keys = cfg.env.processor.leader_intervention or LeaderInterventionConfig()
        if keys.follow_follower:
            print("- The leader follows the follower")
            print(f"- '{keys.pause_key}': with a hand on the leader, take over")
            print(f"- '{keys.pause_key}' again: the leader holds and the policy continues")
        else:
            print(f"- '{keys.pause_key}': pause (the leader moves onto the follower and holds)")
            print(f"- '{keys.correction_key}': with a hand on the leader, take over / hold again")
            print(f"- '{keys.pause_key}' again (after holding): resume the policy")
        print(
            f"- '{keys.success_key}' success, {'/'.join(keys.failure_keys)} failure, '{keys.rerecord_key}' re-record"
        )
        if keys.start_paused and not keys.follow_follower:
            print("- Each episode starts paused with the leader on the follower")
        if cfg.mode == "record":
            take_over_key = keys.pause_key if keys.follow_follower else keys.correction_key
            print(f"- Recording starts at the first '{take_over_key}' of each episode")
    else:
        print("- Use gamepad/teleop device for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    transition = reset_and_build_transition(env, env_processor, action_processor)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        if teleop_device and cfg.env.processor.control_mode != "leader":
            action_features = teleop_device.action_features
        else:
            # Leader-arm interventions are recorded in the policy's end-effector delta action space.
            names = ["delta_x", "delta_y", "delta_z"] + (["gripper"] if use_gripper else [])
            action_features = {
                "dtype": "float32",
                "shape": (len(names),),
                "names": names,
            }
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    # With a leader arm, an episode is recorded from the first take-over on, so the wait before the
    # demonstration starts is not stored as "do nothing" steps.
    record_from_takeover = cfg.env.processor.control_mode == "leader"
    episode_idx = 0
    episode_step = 0
    episode_frames = 0
    episode_start_time = time.perf_counter()

    try:
        while episode_idx < cfg.dataset.num_episodes_to_record:
            step_start_time = time.perf_counter()

            # Create a neutral action (no movement)
            neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            if use_gripper:
                neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

            observation = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)

            info = transition.get(TransitionKey.INFO) or {}
            recording = not record_from_takeover or episode_frames > 0
            recording = recording or info.get(LEADER_PHASE_KEY) == LeaderPhase.CORRECTING.value

            if cfg.mode == "record" and recording:
                action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                    "teleop_action", transition[TransitionKey.ACTION]
                )
                frame = {
                    **observation,
                    ACTION: action_to_record.cpu(),
                    REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                    DONE: np.array([terminated or truncated], dtype=bool),
                }
                if use_gripper:
                    discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                        "discrete_penalty", 0.0
                    )
                    frame["complementary_info.discrete_penalty"] = np.array(
                        [discrete_penalty], dtype=np.float32
                    )

                if dataset is not None:
                    frame["task"] = cfg.dataset.task
                    dataset.add_frame(frame)
                    episode_frames += 1

            episode_step += 1

            # Handle episode termination
            if terminated or truncated:
                episode_time = time.perf_counter() - episode_start_time
                logging.info(
                    f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
                )
                episode_step = 0
                episode_idx += 1

                if dataset is not None:
                    if episode_frames == 0:
                        logging.info(f"Nothing recorded (no take-over), re-recording episode {episode_idx}")
                        dataset.clear_episode_buffer()
                        episode_idx -= 1
                    elif transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                        logging.info(f"Re-recording episode {episode_idx}")
                        dataset.clear_episode_buffer()
                        episode_idx -= 1
                    else:
                        logging.info(f"Saving episode {episode_idx}")
                        dataset.save_episode()

                # Reset for new episode
                transition = reset_and_build_transition(env, env_processor, action_processor)
                episode_start_time = time.perf_counter()
                episode_frames = 0

            # Maintain fps timing
            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))
    finally:
        if dataset is not None and dataset.writer is not None and dataset.writer.image_writer is not None:
            logging.info("Waiting for image writer to finish...")
            dataset.writer.image_writer.stop()
        if dataset is not None:
            # Write the parquet footers, also when the loop is interrupted (Ctrl+C).
            dataset.finalize()

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def close_env_and_processors(env: gym.Env, *pipelines: DataProcessorPipeline) -> None:
    """Close the processor steps that hold hardware (e.g. an actuated leader arm), then the environment."""
    for pipeline in pipelines:
        for step in pipeline.steps:
            if callable(close := getattr(step, "close", None)):
                close()
    env.close()


def replay_trajectory(
    env: gym.Env, action_processor: DataProcessorPipeline, cfg: GymManipulatorConfig
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    actions = dataset.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        precise_sleep(max(1 / cfg.env.fps - (time.perf_counter() - start_time), 0.0))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    init_logging()
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    try:
        if cfg.mode == "replay":
            replay_trajectory(env, action_processor, cfg)
        else:
            control_loop(env, env_processor, action_processor, teleop_device, cfg)
    finally:
        close_env_and_processors(env, env_processor, action_processor)


if __name__ == "__main__":
    main()
