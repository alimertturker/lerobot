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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.envs.configs import LeaderInterventionConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class ExpoFTConfig:
    policy_path: str
    dataset_repo_id: str
    robot: RobotConfig
    teleop: TeleoperatorConfig
    reset_pose: list[float] | None
    home_pose: list[float] | None
    dataset_root: Path | None = None
    output_dir: Path = Path("outputs/expo_ft_so101")
    resume: Path | None = None
    check_only: bool = False  # Validate sampling and one update without connecting hardware or saving.
    real_time: bool = False  # Delayed VLA proposals, current-observation edits and latency-aware learning.
    real_time_delay_steps: int = 4  # Prefetch lead in control ticks; must fit in chunk_steps.
    real_time_filter_candidates: int = 8  # Noise seeds scored before denoising one backup candidate.
    eval_only: bool = False  # Run num_episodes with fixed checkpoint weights, without training.
    eval_record: bool = False  # Save accepted evaluation episodes as a separate LeRobot dataset.
    eval_dataset_repo_id: str = "local/expo_ft_evaluation"
    eval_dataset_root: Path | None = None  # Defaults to output_dir/evaluation_dataset.
    eval_dataset_push_to_hub: bool = False
    eval_dataset_private: bool = True
    debug_frozen_training: bool = False  # Training loop with zero updates and no checkpoint writes.
    debug_plan_seed: int | None = None  # Reset RNG to seed + plan index, after checkpoint restoration.
    capture_control_trace: bool = False  # Buffer per-tick command/timing records until episode end.
    capture_initial_plan: bool = False  # Save each episode's first planning input, RNG, and output.
    capture_all_plans: bool = False  # Capture every planning request when investigating a rollout.
    use_startup_poses: bool = False  # Capture the follower reset/home and leader home at connection time.
    rename_map: dict[str, str] = field(default_factory=dict)  # Empty uses the checkpoint's saved mapping.
    device: str = "cuda"
    cpu_threads: int = 4  # Avoid excessive thread overhead for per-frame CPU preprocessing.
    task: str = "Pick up the cube"
    dataset_use_degrees: bool = True  # Must match the units used when recording the base dataset.
    fps: int = 10
    episode_time_s: float = 60.0
    num_episodes: int = 100
    reset_time_s: float = 3.0
    release_wait_s: float = 3.0
    chunk_steps: int = 5
    num_candidates: int = 8
    candidate_batch_size: int = 2
    num_edits: int = 8
    edit_scale: float = 0.2
    discount: float = 0.99
    tau: float = 0.005
    num_critics: int = 10
    num_min_critics: int = 2
    hidden_dim: int = 256
    batch_size: int = 16
    vla_batch_size: int = 2
    train_vla: bool = True  # Disable VLA optimizer steps while continuing critic and edit learning.
    updates_per_episode: int = 50
    critic_updates_per_update: int = 1
    warmup_updates: int = 100
    critic_lr: float = 3e-4
    edit_lr: float = 3e-4
    vla_lr: float = 1e-5
    temperature: float = 0.01
    adjust_target_entropy: bool = False  # Account for edit_scale in the entropy target.
    grad_clip_norm: float = 10.0
    replay_capacity: int = 10000  # Frames, evicting entire episodes.
    storage_image_size: int = 0  # 0 preserves camera resolution for VLA inference and replay.
    critic_image_size: int = 96
    critic_encoder: str = "compact"  # "resnet" matches the released EXPO-FT encoder at 224x224.
    demo_episodes: int = 20
    seed: int = 1000
    joint_bounds: dict[str, list[float]] = field(default_factory=dict)
    save_every: int = 1
    shared_replay: bool = True  # Store immutable replay episodes once across checkpoints.
    checkpoint_keep_last: int = 0  # 0 keeps all; only checkpoints made by the new saver are managed.
    leader_intervention: LeaderInterventionConfig = field(
        default_factory=lambda: LeaderInterventionConfig(follow_follower=True)
    )

    def __post_init__(self) -> None:
        if self.eval_dataset_push_to_hub and (
            not self.eval_only or not self.eval_record or self.eval_dataset_repo_id.startswith("local/")
        ):
            raise ValueError("Evaluation upload requires eval_only=true, eval_record=true and a Hub repo ID")
        if self.debug_frozen_training and (self.resume is None or self.eval_only or self.check_only):
            raise ValueError("debug_frozen_training requires resume and eval_only=false, check_only=false")
        if self.debug_plan_seed is not None and (
            self.debug_plan_seed < 0 or not (self.eval_only or self.debug_frozen_training)
        ):
            raise ValueError("debug_plan_seed must be nonnegative and requires a frozen debug/evaluation run")
        if self.eval_only and (self.resume is None or self.check_only):
            raise ValueError(
                "eval_only requires a checkpoint in resume and cannot be combined with check_only"
            )
        if self.real_time and self.debug_plan_seed is not None:
            raise ValueError("debug_plan_seed currently supports normal EXPO-FT only")
        if self.real_time and not 0 < self.real_time_delay_steps <= self.chunk_steps:
            raise ValueError("real_time_delay_steps must be in [1, chunk_steps]")
        if self.real_time and self.real_time_filter_candidates < 1:
            raise ValueError("real_time_filter_candidates must be positive")
        positive = (
            "fps",
            "cpu_threads",
            "episode_time_s",
            "num_episodes",
            "reset_time_s",
            "chunk_steps",
            "num_candidates",
            "candidate_batch_size",
            "num_critics",
            "num_min_critics",
            "hidden_dim",
            "batch_size",
            "vla_batch_size",
            "critic_updates_per_update",
            "replay_capacity",
            "critic_image_size",
            "demo_episodes",
            "save_every",
            "edit_scale",
            "critic_lr",
            "edit_lr",
            "vla_lr",
            "temperature",
            "grad_clip_norm",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "num_edits",
            "updates_per_episode",
            "warmup_updates",
            "release_wait_s",
            "checkpoint_keep_last",
            "storage_image_size",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.num_edits > self.num_candidates or self.num_min_critics > self.num_critics:
            raise ValueError("num_edits/num_min_critics exceed the available candidates/critics")
        if not 0 < self.discount <= 1 or not 0 < self.tau <= 1:
            raise ValueError("discount and tau must be in (0, 1]")
        if self.leader_intervention.max_joint_speed <= 0:
            raise ValueError("max_joint_speed must be positive")
        if self.robot.type not in ("so100_follower", "so101_follower"):
            raise ValueError("This example supports one SO-100/SO-101 follower")
        if self.teleop.type not in ("so100_leader", "so101_leader"):
            raise ValueError("This example requires an SO-100/SO-101 leader")
        if self.robot.use_degrees != self.teleop.use_degrees:
            raise ValueError("Leader and follower must use the same joint units")
        if self.robot.use_degrees != self.dataset_use_degrees:
            raise ValueError("Robot joint units must match the base dataset's recording units")
        if self.critic_image_size < 32:
            raise ValueError("critic_image_size must be at least 32")
        if self.critic_encoder not in ("compact", "resnet"):
            raise ValueError("critic_encoder must be 'compact' or 'resnet'")
        if self.critic_encoder == "resnet" and self.critic_image_size != 224:
            raise ValueError("critic_encoder='resnet' requires critic_image_size=224")
        for pose in (self.reset_pose, self.home_pose):
            if pose is None:
                if not self.use_startup_poses:
                    raise ValueError("Set reset_pose/home_pose or enable use_startup_poses")
            elif len(pose) != 6:
                raise ValueError("reset_pose and home_pose require five arm joints plus the gripper")
