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

"""Run: python -m lerobot.rl.expo_ft.train --config_path examples/expo_ft_so101/configs/train.json"""

import argparse
import json
import logging
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import draccus
import torch
import torch.nn.functional as functional

from lerobot.cameras import opencv  # noqa: F401 (register camera config)
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import RenameObservationsProcessorStep, create_transition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.rl.leader_intervention import (
    INTERVENTION_JOINT_TARGET_KEY,
    LeaderArmInterventionStep,
    LeaderPhase,
)
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.teleoperators import make_teleoperator_from_config, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

from .checkpointing import CHECKPOINT_MARKER, prune_checkpoints
from .configuration import ExpoFTConfig
from .control import JointSafetyLimitError, compact_observation, limit_joint_target
from .learner import ExpoFTLearner
from .replay import Episode, EpisodeReplay

logger = logging.getLogger(__name__)


def feature_names(metadata: Any, key: str) -> list[str]:
    names = metadata.features[key].get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{key} requires an explicit ordered list of joint feature names")
    return names


def load_learner(cfg: ExpoFTConfig) -> tuple[ExpoFTLearner, Any, EpisodeReplay, int]:
    if not cfg.resume and not cfg.check_only:
        existing = cfg.output_dir / "checkpoints"
        if existing.exists() and any(existing.glob("episode_*/training_state.pt")):
            raise FileExistsError(
                f"{existing} already contains checkpoints. Use --resume=<checkpoint> "
                "to continue, or choose a new --output_dir for a fresh run."
            )
    if cfg.resume:
        saved_config = cfg.resume / "expo_ft_config.json"
        if saved_config.exists() and json.loads(saved_config.read_text())["chunk_steps"] != cfg.chunk_steps:
            raise ValueError(
                "chunk_steps differs from the saved critics/edit policy. Start a fresh run or "
                "restore the checkpoint's original chunk_steps."
            )
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select a working device")
    metadata = LeRobotDatasetMetadata(cfg.dataset_repo_id, root=cfg.dataset_root)
    if metadata.fps != cfg.fps:
        raise ValueError(f"Dataset FPS ({metadata.fps}) must match online FPS ({cfg.fps})")
    source = str(cfg.resume / "pretrained_model") if cfg.resume else cfg.policy_path
    logger.info("Loading VLA checkpoint: %s (CPU threads=%d)", source, torch.get_num_threads())
    policy_cfg = PreTrainedConfig.from_pretrained(source)
    if policy_cfg.type not in ("pi05", "smolvla"):
        raise ValueError("This example supports PI05 and SmolVLA checkpoints")
    if policy_cfg.output_features[ACTION].shape != (6,):
        raise ValueError("Base VLA must predict six joint positions, including the gripper")
    recorded_names = feature_names(metadata, ACTION)
    checkpoint_names = getattr(policy_cfg, "action_feature_names", None)
    if checkpoint_names and list(checkpoint_names) != recorded_names:
        raise ValueError("Checkpoint action order differs from the demonstration dataset")
    for key in (ACTION, OBS_STATE):
        if metadata.features[key]["shape"] != (6,) and metadata.features[key]["shape"] != [6]:
            raise ValueError(f"Dataset {key} must contain exactly six SO-arm joints")
    policy_cfg.pretrained_path = Path(source)
    policy_cfg.device = cfg.device
    if policy_cfg.type == "smolvla" and not policy_cfg.use_peft:
        # Full local weights already include the backbone; do not download its original weights again.
        policy_cfg.load_vlm_weights = False
    overrides = {"device_processor": {"device": cfg.device}}
    if cfg.rename_map:
        overrides["rename_observations_processor"] = {"rename_map": cfg.rename_map}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=source,
        preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    cfg.rename_map = next(
        (
            dict(step.rename_map)
            for step in preprocessor.steps
            if isinstance(step, RenameObservationsProcessorStep)
        ),
        {},
    )
    policy = make_policy(policy_cfg, ds_meta=metadata, rename_map=cfg.rename_map)
    image_keys = [
        key for key in metadata.camera_keys if cfg.rename_map.get(key, key) in policy_cfg.image_features
    ]
    learner = ExpoFTLearner(cfg, policy, preprocessor, postprocessor, image_keys=image_keys)
    logger.info("VLA ready; dataset cameras=%s, rename_map=%s", image_keys, cfg.rename_map)
    replay = EpisodeReplay(cfg.replay_capacity, cfg.seed)
    episodes = learner.restore(cfg.resume, replay) if cfg.resume else 0
    return learner, metadata, replay, episodes


def load_demonstrations(cfg: ExpoFTConfig, learner: ExpoFTLearner, replay: EpisodeReplay) -> None:
    metadata = LeRobotDatasetMetadata(cfg.dataset_repo_id, root=cfg.dataset_root)
    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        episodes=list(range(min(cfg.demo_episodes, metadata.total_episodes))),
        return_uint8=True,
    )
    total_episodes = min(cfg.demo_episodes, metadata.total_episodes)
    logger.info("Loading %d demonstration episodes (%d frames) into replay...", total_episodes, len(dataset))
    started = time.monotonic()
    completed = 0
    observations: list[dict] = []
    actions: list[torch.Tensor] = []
    episode_id = None

    def finish() -> None:
        nonlocal completed
        if not actions:
            return
        # Existing standard recordings do not contain a post-terminal observation. It is never bootstrapped.
        rewards = torch.zeros(len(actions))
        rewards[-1] = 1.0
        replay.add(
            Episode(
                observations + [observations[-1]],
                torch.stack(actions),
                rewards,
                torch.ones(len(actions), dtype=torch.bool),
                success=True,
                offline=True,
            )
        )
        completed += 1
        logger.info(
            "Loaded demonstration %d/%d: episode=%s, frames=%d, replay_frames=%d, elapsed=%.1fs",
            completed,
            total_episodes,
            episode_id,
            len(actions),
            replay.frames,
            time.monotonic() - started,
        )

    for frame in dataset:
        current_id = int(frame["episode_index"])
        if episode_id is not None and current_id != episode_id:
            finish()
            observations, actions = [], []
        episode_id = current_id
        observation = {OBS_STATE: frame[OBS_STATE].float().cpu()}
        for key in learner.image_keys:
            image = frame[key]
            if image.dtype != torch.uint8:
                image = (image * 255).round().clamp(0, 255).to(torch.uint8)
            height, width = image.shape[-2:]
            if max(height, width) > cfg.storage_image_size:
                shape = (
                    max(1, round(height * cfg.storage_image_size / max(height, width))),
                    max(1, round(width * cfg.storage_image_size / max(height, width))),
                )
                image = functional.interpolate(
                    image[None].float(), shape, mode="bilinear", align_corners=False
                )[0]
            observation[key] = image.round().clamp(0, 255).to(torch.uint8).cpu()
        observations.append(observation)
        actions.append(frame[ACTION].float().cpu())
    finish()
    if not replay.frames:
        raise ValueError("No demonstration frames loaded")
    if not any(len(e.actions) >= learner.horizon for e in replay.episodes):
        raise ValueError("At least one successful demonstration must span the VLA's full chunk_size")


def train_updates(learner: ExpoFTLearner, replay: EpisodeReplay, count: int) -> None:
    logger.info("Starting %d learning updates (replay_frames=%d)", count, replay.frames)
    started = time.monotonic()
    for index in range(count):
        metrics = learner.update(replay)
        if index % 10 == 0 or index == count - 1:
            logger.info(
                "Update %d (%d/%d), elapsed=%.1fs: %s",
                learner.updates,
                index + 1,
                count,
                time.monotonic() - started,
                metrics,
            )


def save_checkpoint(learner: ExpoFTLearner, replay: EpisodeReplay, episodes: int) -> Path:
    root = learner.cfg.output_dir / "checkpoints"
    path = root / f"episode_{episodes:06d}_update_{learner.updates:08d}"
    if path.exists():
        return path
    temporary = path.with_name(path.name + f".tmp_{time.time_ns()}")
    replay_dir = learner.cfg.output_dir / "replay" if learner.cfg.shared_replay else None
    try:
        learner.save(temporary, replay, episodes, replay_dir=replay_dir)
        (temporary / "expo_ft_config.json").write_text(json.dumps(draccus.encode(learner.cfg), indent=2))
        (temporary / CHECKPOINT_MARKER).write_text(json.dumps({"version": 1, "managed": True}))
        temporary.rename(path)
    except BaseException:
        # Only remove this invocation's unpublished temporary directory, never a completed checkpoint.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    logger.info("Saved %s", path)
    prune_checkpoints(learner.cfg.output_dir, learner.cfg.checkpoint_keep_last)
    return path


def move_follower(
    robot: Any, pose: list[float], names: list[str], duration: float, bounds: dict[str, list[float]]
) -> None:
    """Slow reset/parking; move the arm before opening the gripper, as in the HIL-SERL example."""
    initial = robot.get_observation()
    target = dict(zip(names, pose, strict=True))
    for key, value in target.items():
        if key in bounds and not bounds[key][0] <= value <= bounds[key][1]:
            raise ValueError(f"Reset/home target is outside configured joint_bounds: {key}")
        if key in bounds and not bounds[key][0] <= initial[key] <= bounds[key][1]:
            logger.warning(
                "Measured %s=%.3f is outside [%.3f, %.3f]; reset will move it toward the safe pose",
                key,
                initial[key],
                bounds[key][0],
                bounds[key][1],
            )
    steps = max(1, round(duration * 30))
    for index in range(1, steps + 1):
        fraction = index / steps
        command = {key: initial[key] + fraction * (target[key] - initial[key]) for key in names}
        for key in bounds.keys() & command.keys():
            command[key] = min(max(command[key], bounds[key][0]), bounds[key][1])
        command["gripper.pos"] = initial["gripper.pos"]
        robot.send_action(command)
        precise_sleep(1 / 30)
    # Finish the gripper motion gradually too; a hardware max_relative_target can clip a one-shot command.
    gripper_start = robot.get_observation()["gripper.pos"]
    for index in range(1, steps + 1):
        robot.send_action(
            {**target, "gripper.pos": gripper_start + index / steps * (target["gripper.pos"] - gripper_start)}
        )
        precise_sleep(1 / 30)


def run_episode(
    cfg: ExpoFTConfig,
    learner: ExpoFTLearner,
    worker: ThreadPoolExecutor,
    robot: Any,
    leader: Any,
    intervention: LeaderArmInterventionStep,
    action_names: list[str],
    state_names: list[str],
) -> Episode | None:
    observations: list[dict] = []
    actions: list[torch.Tensor] = []
    interventions: list[bool] = []
    pending: Future | None = None
    request_generation = 0
    generation = 0
    chunk: torch.Tensor | None = None
    chunk_index = 0
    success = False
    discard = False
    last_phase = intervention.phase
    started = time.monotonic()
    planning_wait_s = 0.0
    plan_count = 0
    completed_plans = 0
    planning_elapsed_s = 0.0
    safety_limit_reason: str | None = None
    request_started = started
    raw = robot.get_observation()
    try:
        while True:
            tick = time.monotonic()
            observation = compact_observation(raw, state_names, learner.image_keys, cfg.storage_image_size)
            transition = intervention(
                create_transition(
                    observation=raw,
                    complementary_data={TELEOP_ACTION_KEY: leader.get_action()},
                )
            )
            info = transition[TransitionKey.INFO]
            phase = intervention.phase
            if phase is not last_phase:
                generation += 1
                chunk = None
                last_phase = phase
            success = info[TeleopEvents.SUCCESS]
            discard = info[TeleopEvents.RERECORD_EPISODE]
            if success or info[TeleopEvents.TERMINATE_EPISODE] or tick - started >= cfg.episode_time_s:
                break

            if pending is not None and pending.done():
                result = pending.result()  # Propagate worker failures; never continue on a failed inference.
                planning_elapsed_s += time.monotonic() - request_started
                completed_plans += 1
                pending = None
                if request_generation == generation and phase is LeaderPhase.AUTONOMOUS:
                    chunk, chunk_index = result, 0

            if phase is LeaderPhase.AUTONOMOUS and chunk is None and pending is None:
                request_generation = generation
                request_started = time.monotonic()
                pending = worker.submit(learner.sample_action_chunk, observation)
                plan_count += 1
                # Do not waste a control tick when a cached/fast planner finishes immediately.
                if pending.done():
                    chunk, chunk_index = pending.result(), 0
                    planning_elapsed_s += time.monotonic() - request_started
                    completed_plans += 1
                    pending = None

            desired = None
            if phase is not LeaderPhase.AUTONOMOUS:
                desired = transition[TransitionKey.COMPLEMENTARY_DATA][INTERVENTION_JOINT_TARGET_KEY]
            elif chunk is not None and chunk_index < len(chunk):
                desired = dict(zip(action_names, chunk[chunk_index].tolist(), strict=True))
                chunk_index += 1
                if chunk_index == len(chunk):
                    chunk = None
            if desired is not None:
                desired = limit_joint_target(
                    desired, raw, cfg.leader_intervention.max_joint_speed / cfg.fps, cfg.joint_bounds
                )
                sent = robot.send_action(desired)
                if phase is not LeaderPhase.AUTONOMOUS:
                    # Keep the controller's reference at the limited target actually sent to the follower.
                    intervention._target = dict(sent)
                observations.append(observation)
                actions.append(torch.tensor([sent[key] for key in action_names], dtype=torch.float32))
                interventions.append(info[TeleopEvents.IS_INTERVENTION])
            # While inference is pending, hold the current goal and keep servicing leader and keys.
            precise_sleep(max(0, 1 / cfg.fps - (time.monotonic() - tick)))
            if desired is None and phase is LeaderPhase.AUTONOMOUS:
                planning_wait_s += time.monotonic() - tick
            raw = robot.get_observation()
    except JointSafetyLimitError as exc:
        safety_limit_reason = str(exc)
    finally:
        # Hold a released leader immediately, before waiting on any in-flight GPU call.
        intervention.reset()
        # Do not let an old inference race with the next episode's training/preprocessor state.
        if pending is not None:
            pending.cancel()
            if not pending.cancelled():
                pending.result()
    if safety_limit_reason is not None:
        logger.warning("Discarding episode and rerecording after joint safety limit: %s", safety_limit_reason)
        return None
    if not actions or discard:
        logger.info("Discarding episode (%d actions, rerecord=%s)", len(actions), discard)
        return None
    observations.append(compact_observation(raw, state_names, learner.image_keys, cfg.storage_image_size))
    rewards = torch.zeros(len(actions))
    rewards[-1] = float(success)
    elapsed = time.monotonic() - started
    timing = {
        "duration_s": elapsed,
        "planning_wait_s": planning_wait_s,
        "plans": float(plan_count),
        "completed_plans": float(completed_plans),
        "mean_plan_s": planning_elapsed_s / max(completed_plans, 1),
    }
    logger.info(
        "Episode ended: steps=%d success=%s interventions=%d duration=%.1fs planning_wait=%.1fs "
        "plans=%d mean_plan=%.3fs",
        len(actions),
        success,
        sum(interventions),
        elapsed,
        planning_wait_s,
        plan_count,
        timing["mean_plan_s"],
    )
    return Episode(
        observations, torch.stack(actions), rewards, torch.tensor(interventions), success, timing=timing
    )


def train(cfg: ExpoFTConfig) -> None:
    learner, metadata, replay, episodes = load_learner(cfg)
    robot = make_robot_from_config(cfg.robot)
    leader = make_teleoperator_from_config(cfg.teleop)
    action_names = feature_names(metadata, ACTION)
    state_names = feature_names(metadata, OBS_STATE)
    expected = set(robot.action_features)
    if len(action_names) != 6 or set(action_names) != expected or set(state_names) != expected:
        raise ValueError("Dataset action/state names must match the six SO-arm joint names")
    if action_names != state_names:
        raise ValueError("Dataset state/action joint order must agree for relative-action preprocessing")
    if set(leader.action_features) != expected:
        raise ValueError("Leader and follower joint names differ")
    if set(cfg.joint_bounds) - expected:
        raise ValueError("joint_bounds contains an unknown joint name")
    for key, limits in cfg.joint_bounds.items():
        if len(limits) != 2 or not limits[0] < limits[1]:
            raise ValueError(f"Invalid joint_bounds for {key}")
    camera_names = {key.removeprefix("observation.images.") for key in learner.image_keys}
    if not camera_names <= set(robot.cameras):
        raise ValueError("Robot cameras must use the same names as the pretrained VLA")
    if not cfg.resume:
        load_demonstrations(cfg, learner, replay)
    if cfg.check_only:
        first = replay.episodes[0].observations[0]
        chunk = learner.sample_action_chunk(first)
        metrics = learner.update(replay)
        logger.info("Hardware-free check passed: chunk_shape=%s metrics=%s", tuple(chunk.shape), metrics)
        return
    if not cfg.resume:
        train_updates(learner, replay, cfg.warmup_updates)
    logger.info("Warmup complete. Connecting arms and starting online episodes...")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    intervention = None
    safety_stop = False
    # A single worker serializes all GPU operations and stateful processor calls.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="expo-ft-gpu") as worker:
        try:
            robot.connect(calibrate=False)
            leader.connect(calibrate=False)
            if not robot.is_calibrated or not leader.is_calibrated:
                raise ValueError("Calibrate both arms with lerobot-calibrate before online training")
            leader_home = capture_home_poses(cfg, robot, leader, action_names)
            motor_names = [name.removesuffix(".pos") for name in action_names]
            intervention = LeaderArmInterventionStep(
                teleop_device=leader,
                kinematics=None,
                motor_names=motor_names,
                end_effector_step_sizes={},
                fps=cfg.fps,
                config=cfg.leader_intervention,
                action_representation="joint_positions",
                home_pose=leader_home,
            )
            assert cfg.reset_pose is not None and cfg.home_pose is not None
            while episodes < cfg.num_episodes:
                # Hold a released leader before any follower reset.
                intervention.reset()
                raw = robot.get_observation()
                reset_duration = max(
                    cfg.reset_time_s,
                    max(
                        abs(raw[key] - value) for key, value in zip(action_names, cfg.reset_pose, strict=True)
                    )
                    / cfg.leader_intervention.max_joint_speed,
                )
                move_follower(robot, cfg.reset_pose, action_names, reset_duration, cfg.joint_bounds)
                episode = run_episode(
                    cfg, learner, worker, robot, leader, intervention, action_names, state_names
                )
                intervention.reset()
                precise_sleep(cfg.release_wait_s)
                if episode is None:
                    continue
                replay.add(episode)
                episodes += 1
                # Training runs between episodes; the follower holds and the leader keeps following it.
                future = worker.submit(train_updates, learner, replay, cfg.updates_per_episode)
                while not future.done():
                    follower_pose = robot.get_observation()
                    if cfg.leader_intervention.follow_follower:
                        intervention._follow(
                            {key: follower_pose[key] for key in action_names}, leader.get_action()
                        )
                    precise_sleep(1 / cfg.fps)
                future.result()
                if episodes % cfg.save_every == 0:
                    worker.submit(save_checkpoint, learner, replay, episodes).result()
        except JointSafetyLimitError:
            safety_stop = True
            logger.exception("Joint safety bound crossed; stopping without moving the follower to home")
            raise
        finally:
            # Every callback runs even if another hardware cleanup fails. Saving is last and is
            # serialized behind any in-flight GPU update, including on Ctrl+C.
            with ExitStack() as cleanup:
                cleanup.callback(lambda: worker.submit(save_checkpoint, learner, replay, episodes).result())
                if robot.is_connected:
                    cleanup.callback(robot.disconnect)
                if leader.is_connected:
                    cleanup.callback(leader.disconnect)
                if not safety_stop and robot.is_connected and robot.is_calibrated and cfg.home_pose is not None:
                    raw = robot.get_observation()
                    duration = max(
                        cfg.reset_time_s,
                        max(
                            abs(raw[key] - value)
                            for key, value in zip(action_names, cfg.home_pose, strict=True)
                        )
                        / cfg.leader_intervention.max_joint_speed,
                    )
                    cleanup.callback(
                        move_follower, robot, cfg.home_pose, action_names, duration, cfg.joint_bounds
                    )
                if intervention is not None:
                    cleanup.callback(intervention.close)
                    cleanup.callback(intervention.reset)


def capture_home_poses(
    cfg: ExpoFTConfig, robot: Any, leader: Any, action_names: list[str]
) -> dict[str, float]:
    """Read poses only. The leader keeps its own home, independent of the follower's initial pose."""
    if cfg.use_startup_poses:
        follower = robot.get_observation()
        if cfg.reset_pose is None:
            cfg.reset_pose = [float(follower[key]) for key in action_names]
        if cfg.home_pose is None:
            cfg.home_pose = list(cfg.reset_pose)
        leader_pose = leader.get_action()
        home = {key: float(leader_pose[key]) for key in action_names}
        logger.info("Captured follower reset/home=%s and leader home=%s", cfg.home_pose, home)
        return home
    assert cfg.home_pose is not None
    return dict(zip(action_names, cfg.home_pose, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config_path", required=True)
    args, overrides = parser.parse_known_args()
    cfg = draccus.parse(ExpoFTConfig, config_path=args.config_path, args=overrides)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(cfg.output_dir / "train.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Writing training and episode metrics to %s", cfg.output_dir / "train.log")
    train(cfg)


if __name__ == "__main__":
    main()
