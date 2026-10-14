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

"""Push the project's datasets or trained policy to the Hugging Face Hub (private repos).

  scripts/push_hub.sh demos     # outputs/datasets/so101_leader_demos -> $HF_USER/so101_leader_demos
  scripts/push_hub.sh policy    # latest training checkpoint          -> $HF_USER/hilserl_so101_leader

HF_USER comes from local.env (via push_hub.sh). Needs `hf auth login` once. Add --public to make the repo
public.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset

USER = os.environ.get("HF_USER", "")
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS = {
    "demos": "so101_leader_demos",
}
POLICY_REPO = "hilserl_so101_leader"
CHECKPOINTS = PROJECT_DIR / "outputs/train/hilserl_so101_leader/checkpoints"


def push_dataset(name: str, private: bool) -> None:
    root = PROJECT_DIR / "outputs/datasets" / name
    if not (root / "meta/info.json").exists():
        raise SystemExit(f"No dataset at {root}")
    dataset = LeRobotDataset(f"{USER}/{name}", root=root)
    print(f"Pushing {dataset.num_episodes} episodes from {root} to {USER}/{name}")
    dataset.push_to_hub(private=private, tags=["LeRobot", "hil-serl", "so101"])
    print(f"Done: https://huggingface.co/datasets/{USER}/{name}")


def push_policy(private: bool) -> None:
    last = CHECKPOINTS / "last/pretrained_model"
    if not last.exists():
        raise SystemExit(f"No checkpoint at {last}")
    step = (CHECKPOINTS / "last").resolve().name
    repo_id = f"{USER}/{POLICY_REPO}"
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    print(f"Pushing checkpoint {step} to {repo_id}")
    api.upload_folder(repo_id=repo_id, folder_path=last, commit_message=f"Checkpoint {step}")
    print(f"Done: https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("what", choices=[*DATASETS, "policy"])
    parser.add_argument("--public", action="store_true", help="Create a public repo (default: private)")
    args = parser.parse_args()
    if not USER:
        raise SystemExit("HF_USER is not set (set it in local.env and run scripts/push_hub.sh)")
    if args.what == "policy":
        push_policy(private=not args.public)
    else:
        push_dataset(DATASETS[args.what], private=not args.public)


if __name__ == "__main__":
    main()
