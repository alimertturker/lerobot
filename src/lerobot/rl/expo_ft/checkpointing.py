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

"""Retention for completed EXPO checkpoints and their shared immutable replay episodes."""

import json
import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)
CHECKPOINT_MARKER = "checkpoint_storage.json"


def prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    root = output_dir / "checkpoints"
    managed = []
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir() or not re.fullmatch(r"episode_\d+_update_\d+", path.name):
            continue
        marker = path / CHECKPOINT_MARKER
        if marker.is_file():
            try:
                if json.loads(marker.read_text()) == {"version": 1, "managed": True}:
                    managed.append(path)
            except (OSError, ValueError):
                logger.warning("Leaving unreadable checkpoint marker untouched: %s", marker)
    managed.sort(key=lambda path: path.name)
    if keep_last > 0:
        for path in managed[:-keep_last]:
            shutil.rmtree(path)
            logger.info("Removed older managed checkpoint: %s", path)

    replay_dir = output_dir / "replay"
    if not replay_dir.is_dir():
        return
    referenced: set[str] = set()
    for manifest_path in root.glob("*/replay_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            resolved = (manifest_path.parent / manifest["directory"]).resolve()
            if manifest["format"] != "shared_episodes_v1" or resolved != replay_dir.resolve():
                logger.warning("Skipping replay cleanup: unexpected replay manifest %s", manifest_path)
                return
            referenced.update(manifest["episode_ids"])
        except (KeyError, TypeError, ValueError, OSError):
            logger.warning("Skipping replay cleanup: cannot read %s", manifest_path)
            return
    if not referenced:
        return
    for path in replay_dir.glob("*.pt"):
        if not path.is_symlink() and re.fullmatch(r"[0-9a-f]{32}", path.stem) and path.stem not in referenced:
            path.unlink()
