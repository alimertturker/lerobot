"""Capture the exact first planning input and output for comparing robot runs."""

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from torch import Tensor

if TYPE_CHECKING:
    from .learner import ExpoFTLearner

logger = logging.getLogger(__name__)
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expo-ft-diagnostics")
_pending: list[Future] = []


def flush_captures() -> None:
    """Finish queued disk writes after robot cleanup, and surface recording errors."""
    pending = list(_pending)
    _pending.clear()
    for future in pending:
        future.result()


def capture_initial_plan(
    learner: "ExpoFTLearner", observation: dict[str, Tensor], context: dict | None = None
) -> Tensor:
    # Called in the existing GPU worker, so neither sampling order nor processor ownership changes.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else []
    details = {}
    requested_ns = time.time_ns()
    chunk = learner.sample_action_chunk(observation, diagnostics=details)
    completed_ns = time.time_ns()
    directory = (
        learner.cfg.output_dir
        / ("policy_plans" if learner.cfg.capture_all_plans else "initial_plans")
        / str(requested_ns)
    )
    metadata = {
        "requested_time_ns": requested_ns,
        "completed_time_ns": completed_ns,
        **(context or {}),
        "eval_only": learner.cfg.eval_only,
        "updates": learner.updates,
        "resume": str(learner.cfg.resume) if learner.cfg.resume else None,
        "state": observation["observation.state"].tolist(),
        "task": learner.cfg.task,
        "image_keys": learner.image_keys,
        "num_candidates": learner.cfg.num_candidates,
        "num_edits": learner.cfg.num_edits,
        "edit_scale": learner.cfg.edit_scale,
        **details,
    }
    # Observations already own their CPU image storage; the chunk is a fresh CPU result.
    # Only immutable snapshots reach the writer, which never calls policy/processors or consumes RNG.
    payload = {"observation": observation, "action_chunk": chunk, "rng": cpu_rng, "cuda_rng": cuda_rng}
    _pending.append(_writer.submit(_write_capture, directory, payload, metadata, tuple(learner.image_keys)))
    logger.info("Captured policy plan: %s (updates=%d)", directory, learner.updates)
    return chunk


def sample_debug_plan(learner: "ExpoFTLearner", observation: dict[str, Tensor], index: int, episode_id: str):
    """Runs in the sole GPU worker; pair stochastic draws across fresh-process debug trials."""
    seed = learner.cfg.debug_plan_seed
    if seed is not None:
        torch.manual_seed(seed + index)
    capture = learner.cfg.capture_all_plans or (learner.cfg.capture_initial_plan and index == 0)
    if capture:
        return capture_initial_plan(
            learner,
            observation,
            {
                "episode_id": episode_id,
                "plan_index": index,
                "debug_seed": None if seed is None else seed + index,
            },
        )
    return learner.sample_action_chunk(observation)


class ControlTrace:
    """Collect small CPU records; write only after the episode's control loop finishes."""

    def __init__(self, cfg, names: list[str]):
        self.enabled = cfg.capture_control_trace
        self.episode_id = str(time.time_ns())
        self.directory = cfg.output_dir / "control_traces"
        self.records = []
        self.add(
            "start",
            action_names=names,
            eval_only=cfg.eval_only,
            debug_frozen_training=cfg.debug_frozen_training,
            debug_plan_seed=cfg.debug_plan_seed,
        )

    def add(self, event: str, **values):
        if self.enabled:
            self.records.append(
                {"event": event, "monotonic_ns": time.monotonic_ns(), "time_ns": time.time_ns(), **values}
            )

    def finish(self):
        if self.enabled:
            _pending.append(_writer.submit(self._write))

    def _write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.episode_id}.jsonl"
        with path.open("x") as stream:
            for record in self.records:
                stream.write(json.dumps(record) + "\n")
        logger.info("Saved control trace: %s", path)


def _write_capture(directory: Path, payload: dict, metadata: dict, image_keys: tuple[str, ...]) -> None:
    directory.mkdir(parents=True)
    torch.save(payload, directory / "plan.pt")
    for key in image_keys:
        pixels = payload["observation"][key].detach().cpu().permute(1, 2, 0).numpy()
        Image.fromarray(pixels).save(directory / f"{key.removeprefix('observation.images.')}.png")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
