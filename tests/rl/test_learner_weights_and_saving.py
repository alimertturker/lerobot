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

"""The actor waits for the learner's first weights; the learner swaps saved buffers in only when complete."""

import threading
import time
from pathlib import Path
from queue import Queue

import pytest
import torch

pytest.importorskip("grpc")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rl.actor import wait_for_policy_parameters  # noqa: E402
from lerobot.rl.learner import _save_buffer_atomically  # noqa: E402
from lerobot.transport.utils import state_to_bytes  # noqa: E402


class FakeAlgorithm:
    def __init__(self):
        self.loaded = []

    def load_weights(self, state_dicts, device=None):
        self.loaded.append(state_dicts)


def test_actor_waits_for_the_first_weights():
    queue: Queue = Queue()
    algorithm = FakeAlgorithm()
    threading.Timer(0.3, lambda: queue.put(state_to_bytes({"policy": {"w": torch.ones(2)}}))).start()
    start = time.perf_counter()
    assert wait_for_policy_parameters(algorithm, queue, "cpu", threading.Event())
    assert time.perf_counter() - start >= 0.25
    assert len(algorithm.loaded) == 1
    torch.testing.assert_close(algorithm.loaded[0]["policy"]["w"], torch.ones(2))


def test_actor_loads_the_latest_queued_weights():
    queue: Queue = Queue()
    queue.put(state_to_bytes({"policy": {"w": torch.zeros(2)}}))
    queue.put(state_to_bytes({"policy": {"w": torch.ones(2)}}))
    algorithm = FakeAlgorithm()
    assert wait_for_policy_parameters(algorithm, queue, "cpu", threading.Event())
    assert len(algorithm.loaded) == 1
    torch.testing.assert_close(algorithm.loaded[0]["policy"]["w"], torch.ones(2))


def test_actor_stops_waiting_on_shutdown():
    shutdown = threading.Event()
    threading.Timer(0.2, shutdown.set).start()
    algorithm = FakeAlgorithm()
    assert not wait_for_policy_parameters(algorithm, Queue(), "cpu", shutdown)
    assert algorithm.loaded == []


class FakeBuffer:
    def __init__(self, content: str, fail: bool = False):
        self.content = content
        self.fail = fail

    def to_lerobot_dataset(self, repo_id, fps, root):
        Path(root).mkdir(parents=True)
        (Path(root) / "data.txt").write_text(self.content)
        if self.fail:
            raise KeyboardInterrupt  # e.g. a second Ctrl+C in the middle of the save


def test_buffer_save_replaces_the_previous_copy(tmp_path):
    dataset_dir = str(tmp_path / "dataset")
    _save_buffer_atomically(FakeBuffer("old"), "repo", 10, dataset_dir)
    _save_buffer_atomically(FakeBuffer("new"), "repo", 10, dataset_dir)
    assert (tmp_path / "dataset" / "data.txt").read_text() == "new"
    assert not (tmp_path / "dataset.tmp").exists()


def test_interrupted_buffer_save_keeps_the_previous_copy(tmp_path):
    dataset_dir = str(tmp_path / "dataset")
    _save_buffer_atomically(FakeBuffer("old"), "repo", 10, dataset_dir)
    with pytest.raises(KeyboardInterrupt):
        _save_buffer_atomically(FakeBuffer("partial", fail=True), "repo", 10, dataset_dir)
    assert (tmp_path / "dataset" / "data.txt").read_text() == "old"
    # the next save cleans up the partial copy
    _save_buffer_atomically(FakeBuffer("new"), "repo", 10, dataset_dir)
    assert (tmp_path / "dataset" / "data.txt").read_text() == "new"
