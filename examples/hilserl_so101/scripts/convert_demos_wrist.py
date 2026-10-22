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

"""Turn xyz-only demos ([dx, dy, dz, gripper]) into wrist demos ([dx, dy, dz, d_wrist_flex, d_wrist_roll, gripper]).

The xyz part is kept: it is the gripper-tip move computed with all joints, the same as in the wrist action. The
wrist part is the measured wrist_flex / wrist_roll change to the next frame divided by the step size from
configs/train_hilserl_leader.json (`joint_step_sizes`), clipped to [-1, 1]; 0 on the last frame of an episode.
Measured positions lag the commanded ones by up to a step, which matters little for the wrist (it barely sags).

Copies the source dataset (images and videos unchanged) and rewrites the actions, `meta/info.json` and the
action statistics. The source dataset is not modified. Nothing moves.

  scripts/convert_demos_wrist.sh    # outputs/datasets/so101_leader_demos -> outputs/datasets/so101_leader_demos_wrist
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
# observation.state starts with the joint positions in motor order
WRIST_STATE_INDEX = {"wrist_flex": 3, "wrist_roll": 4}


def wrist_actions(episode: pd.DataFrame, step_sizes: dict[str, float]) -> np.ndarray:
    """Return the episode's actions with the wrist deltas inserted before the gripper."""
    actions = np.stack(episode["action"].to_numpy()).astype(np.float32)
    if actions.shape[1] != 4:
        raise ValueError(f"Expected [dx, dy, dz, gripper] actions, got shape {actions.shape}")
    state = np.stack(episode["observation.state"].to_numpy()).astype(np.float64)
    columns = []
    for joint, step in step_sizes.items():
        position = state[:, WRIST_STATE_INDEX[joint]]
        delta = np.zeros(len(position))
        delta[:-1] = np.diff(position) / step
        columns.append(np.clip(delta, -1.0, 1.0))
    return np.concatenate([actions[:, :3], np.stack(columns, axis=1), actions[:, 3:]], axis=1).astype(
        np.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=PROJECT_DIR / "outputs/datasets/so101_leader_demos")
    parser.add_argument("--dst", type=Path, default=PROJECT_DIR / "outputs/datasets/so101_leader_demos_wrist")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/train_hilserl_leader.json")
    args = parser.parse_args()

    step_sizes = json.loads(args.config.read_text())["env"]["processor"]["inverse_kinematics"][
        "joint_step_sizes"
    ]
    if list(step_sizes) != list(WRIST_STATE_INDEX):
        raise ValueError(f"Expected joint_step_sizes for {list(WRIST_STATE_INDEX)}, got {step_sizes}")
    if args.dst.exists():
        raise SystemExit(f"{args.dst} already exists; move it away first.")
    info = json.loads((args.src / "meta/info.json").read_text())
    if info["features"]["action"]["shape"] != [4]:
        raise SystemExit(f"{args.src} does not hold [dx, dy, dz, gripper] actions.")

    shutil.copytree(args.src, args.dst)
    try:
        episode_stats = {}
        for path in sorted((args.dst / "data").rglob("*.parquet")):
            frames = pd.read_parquet(path)
            frames = frames.sort_values("index")
            new_actions = [None] * len(frames)
            for episode_index, episode in frames.groupby("episode_index"):
                episode = episode.sort_values("frame_index")
                actions = wrist_actions(episode, step_sizes)
                for row, action in zip(episode.index, actions, strict=True):
                    new_actions[frames.index.get_loc(row)] = action
                episode_stats[episode_index] = get_feature_stats(actions, axis=0, keepdims=False)
            frames["action"] = new_actions
            frames.to_parquet(path, index=False)

        for path in sorted((args.dst / "meta/episodes").rglob("*.parquet")):
            episodes = pd.read_parquet(path)
            for stat in episode_stats[episodes["episode_index"].iloc[0]]:
                column = f"stats/action/{stat}"
                dtype = episodes[column].iloc[0].dtype  # keep the stored dtype (parquet can't mix them)
                episodes[column] = [episode_stats[i][stat].astype(dtype) for i in episodes["episode_index"]]
            episodes.to_parquet(path, index=False)

        names = ["delta_x", "delta_y", "delta_z"] + [f"delta_{joint}" for joint in step_sizes] + ["gripper"]
        info["features"]["action"] = {"dtype": "float32", "shape": [len(names)], "names": names}
        (args.dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")

        stats = json.loads((args.dst / "meta/stats.json").read_text())
        action_stats = aggregate_stats([{"action": s} for s in episode_stats.values()])["action"]
        stats["action"] = {k: np.asarray(v).tolist() for k, v in action_stats.items()}
        (args.dst / "meta/stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    except Exception:
        shutil.rmtree(args.dst)
        raise

    print(f"Wrote {args.dst}: {len(episode_stats)} episodes, actions {names}")
    for i, joint in enumerate(step_sizes, start=3):
        low, high = action_stats["min"][i], action_stats["max"][i]
        print(f"  delta_{joint}: min {low:.2f}, max {high:.2f}, std {action_stats['std'][i]:.3f}")


if __name__ == "__main__":
    main()
