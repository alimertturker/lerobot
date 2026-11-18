"""Persist evaluation episodes between rollouts, without changing policy execution."""

import json
import logging

import numpy as np

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration import ExpoFTConfig
from .replay import Episode

logger = logging.getLogger(__name__)


class EvaluationRecorder:
    def __init__(self, cfg: ExpoFTConfig, action_names: list[str], state_names: list[str]):
        self.cfg = cfg
        self.enabled = cfg.eval_only and cfg.eval_record
        self.root = cfg.eval_dataset_root or cfg.output_dir / "evaluation_dataset"
        self.action_names, self.state_names = action_names, state_names
        self.dataset = None
        if self.enabled and self.root.exists():
            raise FileExistsError(
                f"Evaluation dataset already exists: {self.root}. Choose a new eval_dataset_root."
            )

    def __enter__(self):
        return self

    def add(self, episode: Episode) -> None:
        if not self.enabled:
            return
        if self.dataset is None:
            features = {
                OBS_STATE: {"dtype": "float32", "shape": (len(self.state_names),), "names": self.state_names},
                ACTION: {"dtype": "float32", "shape": (len(self.action_names),), "names": self.action_names},
                "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
                "next.done": {"dtype": "bool", "shape": (1,), "names": None},
                "next.success": {"dtype": "bool", "shape": (1,), "names": None},
                "complementary_info.is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
            }
            for key, image in episode.observations[0].items():
                if key.startswith("observation.images."):
                    features[key] = {
                        "dtype": "video",
                        "shape": tuple(image.permute(1, 2, 0).shape),
                        "names": ["height", "width", "channels"],
                    }
            self.dataset = LeRobotDataset.create(
                repo_id=self.cfg.eval_dataset_repo_id,
                root=self.root,
                fps=self.cfg.fps,
                robot_type=self.cfg.robot.type,
                features=features,
                use_videos=True,
                rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="fast"),
                encoder_threads=2,
                streaming_encoding=True,
                metadata_buffer_size=1,
            )
            (self.root / "evaluation_context.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(self.cfg.resume),
                        "real_time": self.cfg.real_time,
                        "chunk_steps": self.cfg.chunk_steps,
                        "task": self.cfg.task,
                        "use_degrees": self.cfg.dataset_use_degrees,
                        "sampling": "One frame per executed action; resets and inference-only hold ticks excluded.",
                        "terminal_observation": "The extra replay terminal observation is not a dataset action frame.",
                    },
                    indent=2,
                )
            )
        for index, action in enumerate(episode.actions):
            terminal = index == len(episode.actions) - 1
            frame = {
                key: value.permute(1, 2, 0).contiguous().numpy()
                if key.startswith("observation.images.")
                else value.numpy()
                for key, value in episode.observations[index].items()
            }
            frame.update(
                {
                    ACTION: action.numpy(),
                    "task": self.cfg.task,
                    "next.reward": np.array([episode.rewards[index].item()], dtype=np.float32),
                    "next.done": np.array([terminal], dtype=bool),
                    "next.success": np.array([terminal and episode.success], dtype=bool),
                    "complementary_info.is_intervention": np.array(
                        [episode.interventions[index].item()], dtype=bool
                    ),
                }
            )
            self.dataset.add_frame(frame)
        self.dataset.save_episode(parallel_encoding=False)
        with (self.root / "evaluation_episodes.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "episode_index": self.dataset.num_episodes - 1,
                        "success": episode.success,
                        "steps": len(episode.actions),
                        "intervention_steps": int(episode.interventions.sum()),
                        "timing": episode.timing,
                    }
                )
                + "\n"
            )
        logger.info("Saved evaluation episode %d to %s", self.dataset.num_episodes, self.root)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.dataset is not None:
            self.dataset.finalize()
            logger.info("Evaluation dataset finalized: %s", self.root)
            if exc_type is None and self.cfg.eval_dataset_push_to_hub:
                self.dataset.push_to_hub(private=self.cfg.eval_dataset_private)
