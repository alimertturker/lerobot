"""Reference encoder integration, padding alignment, and checkpoint compatibility."""

import json

import pytest
import torch

from lerobot.rl.expo_ft.networks import ReferenceObservationEncoder, _same_pad
from lerobot.rl.expo_ft.replay import EpisodeReplay
from lerobot.rl.expo_ft.train import load_learner
from lerobot.utils.constants import OBS_STATE
from tests.rl.test_expo_ft import cfg as cfg, episode, learner, observation


def test_same_padding_preserves_flax_stride_alignment():
    # JAX SAME puts the extra pixel on the bottom/right for an even input.
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    pooled = torch.nn.functional.max_pool2d(_same_pad(x, 3, 2, value=-float("inf")), 3, 2)
    torch.testing.assert_close(pooled, torch.tensor([[[[10.0, 11.0], [14.0, 15.0]]]]))
    padded = _same_pad(torch.ones(1, 1, 5, 5), 3, 2)
    assert padded.shape == (1, 1, 7, 7)
    assert padded[..., 0, :].sum() == padded[..., -1, :].sum() == 0


def test_reference_encoder_camera_order_features_and_gradients():
    torch.set_num_threads(1)
    keys = ["observation.images.wrist", "observation.images.top"]
    model = ReferenceObservationEncoder(keys, 6, 224)
    # Covers both cameras, the spatial image projection, and separate state embedding.
    assert sum(p.numel() for p in model.parameters()) == 34_139_456
    obs = {k: torch.rand(1, 3, 64, 80) for k in keys}
    obs[OBS_STATE] = torch.randn(1, 6)
    seen = []
    hook = model.stem.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    features = model(obs)
    hook.remove()
    assert features.shape == (1, 576)
    expected_top = torch.nn.functional.interpolate(
        obs[keys[1]], (224, 224), mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(seen[0][:, :3], expected_top)
    features.square().mean().backward()
    for p in (model.stem.weight, model.image_projection[0].weight, model.state_projection[0].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_reference_encoder_runs_candidate_selection_and_learner_update(cfg):
    torch.set_num_threads(1)
    cfg.critic_encoder = "resnet"
    cfg.critic_image_size = 224
    cfg.batch_size = cfg.vla_batch_size = 1
    agent = learner(cfg)
    replay = EpisodeReplay(100)
    replay.add(episode())
    before = agent.critics.encoder.stem.weight.detach().clone()
    action = agent.sample_action_chunk(observation())
    assert action.shape == (cfg.chunk_steps, 6) and torch.isfinite(action).all()
    metrics = agent.update(replay)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert not torch.equal(before, agent.critics.encoder.stem.weight)
    assert "vla_loss" in metrics


def test_resume_rejects_legacy_encoder_before_loading_policy(cfg, tmp_path):
    cfg.resume = tmp_path
    cfg.critic_encoder = "resnet"
    cfg.critic_image_size = 224
    (tmp_path / "expo_ft_config.json").write_text(json.dumps({"chunk_steps": cfg.chunk_steps}))
    with pytest.raises(ValueError, match="checkpoint architecture"):
        load_learner(cfg)
