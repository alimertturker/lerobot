from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.rl.expo_ft.evaluation import EvaluationRecorder
from tests.rl.test_expo_ft import cfg as cfg, episode


def test_video_dataset_roundtrip_and_interrupted_finalize(cfg):
    cfg = replace(cfg, eval_only=True, eval_record=True, resume=cfg.output_dir / "checkpoint")
    names = [f"joint{i}.pos" for i in range(6)]
    first, second = episode(4), episode(3, success=False)
    first.interventions[1] = True
    for item in (first, second):
        for obs in item.observations:
            obs["observation.images.top"] = obs["observation.images.wrist"].clone()
    with pytest.raises(KeyboardInterrupt), EvaluationRecorder(cfg, names, names) as recorder:
        recorder.add(first)
        recorder.add(second)
        raise KeyboardInterrupt
    root = cfg.output_dir / "evaluation_dataset"
    dataset = LeRobotDataset(cfg.eval_dataset_repo_id, root=root, video_backend="pyav")
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 7
    assert len(list(root.rglob("*.mp4"))) == 2  # One consolidated file per camera.
    for key in ["observation.images.wrist", "observation.images.top"]:
        assert dataset[0][key].shape == (3, 32, 32)
    data = dataset.hf_dataset
    np.testing.assert_allclose(np.asarray(data["action"]), torch.cat([first.actions, second.actions]).numpy())
    assert list(data["next.done"]) == [False, False, False, True, False, False, True]
    assert list(data["next.success"]) == [False, False, False, True, False, False, False]
    assert list(data["complementary_info.is_intervention"]) == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert (root / "evaluation_context.json").is_file()
    assert len((root / "evaluation_episodes.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError, match="already exists"):
        EvaluationRecorder(cfg, names, names)


@pytest.mark.parametrize("evaluate,record", [(False, False), (False, True), (True, False)])
def test_recording_disabled_does_not_create_dataset(cfg, evaluate, record):
    cfg = replace(cfg, eval_only=evaluate, eval_record=record, resume=cfg.output_dir / "checkpoint")
    with EvaluationRecorder(cfg, [], []) as recorder:
        recorder.add(episode())
    assert not (cfg.output_dir / "evaluation_dataset").exists()


def test_upload_is_opt_in_and_requires_hub_namespace(cfg):
    with pytest.raises(ValueError, match="Hub repo ID"):
        replace(cfg, eval_only=True, eval_record=True, eval_dataset_push_to_hub=True, resume=cfg.output_dir)


@pytest.mark.parametrize("upload,interrupted", [(False, False), (True, False), (True, True)])
def test_upload_only_after_successful_finalization(cfg, upload, interrupted):
    cfg = replace(
        cfg,
        eval_only=True,
        eval_record=True,
        resume=cfg.output_dir,
        eval_dataset_repo_id="test-user/evaluation",
        eval_dataset_push_to_hub=upload,
    )
    events = []
    dataset = SimpleNamespace(
        finalize=Mock(side_effect=lambda: events.append("finalized")),
        push_to_hub=Mock(side_effect=lambda **kwargs: events.append("uploaded")),
    )
    recorder = EvaluationRecorder(cfg, [], [])
    recorder.dataset = dataset
    recorder.__exit__(KeyboardInterrupt if interrupted else None, None, None)
    assert events == (["finalized", "uploaded"] if upload and not interrupted else ["finalized"])
    if upload and not interrupted:
        dataset.push_to_hub.assert_called_once_with(private=True)
