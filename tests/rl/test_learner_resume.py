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

"""Learner resume restores the trained policy and keeps the given run settings; the weight push to the actor
leaves out a frozen pretrained vision backbone."""

import os

import pytest
import torch

pytest.importorskip("grpc")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.envs.configs import HILSerlRobotEnvConfig  # noqa: E402
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig  # noqa: E402
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy  # noqa: E402
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig  # noqa: E402
from lerobot.rl.learner import handle_resume_logic, make_learner_policy  # noqa: E402
from lerobot.rl.train_rl import TrainRLServerPipelineConfig  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OBS_IMAGE,
    OBS_STATE,
    PRETRAINED_MODEL_DIR,
)

STATE_DIM = 4
ACTION_DIM = 3


def make_policy_config(with_images: bool = False) -> GaussianActorConfig:
    config = GaussianActorConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * STATE_DIM, "max": [1.0] * STATE_DIM},
            ACTION: {"min": [-1.0] * ACTION_DIM, "max": [1.0] * ACTION_DIM},
        },
        num_discrete_actions=3,
        device="cpu",
    )
    if with_images:
        config.input_features[OBS_IMAGE] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))
        config.dataset_stats[OBS_IMAGE] = {"mean": [[[0.5]]] * 3, "std": [[[0.2]]] * 3}
        config.latent_dim = 16
        config.state_encoder_hidden_dim = 16
    config.validate_features()
    return config


def make_train_config(output_dir) -> TrainRLServerPipelineConfig:
    cfg = TrainRLServerPipelineConfig()
    cfg.policy = make_policy_config()
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    cfg.env = HILSerlRobotEnvConfig(features=features, features_map={k: k for k in features})
    cfg.output_dir = output_dir
    return cfg


def last_checkpoint_dir(output_dir) -> str:
    return os.path.join(output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK, PRETRAINED_MODEL_DIR)


def test_resume_restores_the_trained_policy(tmp_path):
    cfg = make_train_config(tmp_path)
    trained = make_learner_policy(cfg)
    with torch.no_grad():  # stand-in for training: every trainable weight moves away from its init
        for param in trained.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param))
    step_dir = tmp_path / CHECKPOINTS_DIR / "000100"
    trained.save_pretrained(step_dir / PRETRAINED_MODEL_DIR)
    os.symlink("000100", tmp_path / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK)

    cfg.resume = True
    resumed = make_learner_policy(cfg)

    expected = trained.state_dict()
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=f"{name} was not restored")
    # the policy's own config still names where it started from, not the checkpoint
    assert cfg.policy.pretrained_path is None


def test_resume_keeps_the_given_run_settings_and_the_checkpoint_training_settings(tmp_path):
    checkpoint_cfg = make_train_config(tmp_path)
    checkpoint_cfg.batch_size = 64
    checkpoint_cfg.save_freq = 1000
    checkpoint_cfg.policy.actor_learner_config.policy_parameters_push_frequency = 50
    config_dir = tmp_path / CHECKPOINTS_DIR / "000100" / PRETRAINED_MODEL_DIR
    config_dir.mkdir(parents=True)
    checkpoint_cfg._save_pretrained(config_dir)
    os.symlink("000100", tmp_path / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK)

    given = make_train_config(tmp_path)
    given.resume = True
    given.batch_size = 256
    given.save_freq = 5000
    given.buffer_save_freq = 20000
    given.policy.actor_learner_config.policy_parameters_push_frequency = 10

    cfg = handle_resume_logic(given)

    assert cfg.resume
    assert cfg.batch_size == 64  # training settings stay as in the checkpoint
    assert cfg.save_freq == 5000
    assert cfg.buffer_save_freq == 20000
    assert cfg.policy.actor_learner_config.policy_parameters_push_frequency == 10


def make_algorithm(pretrained_backbone: bool) -> tuple[SACAlgorithm, GaussianActorPolicy]:
    config = make_policy_config(with_images=True)
    policy = GaussianActorPolicy(config=config)
    if pretrained_backbone:
        # Built with the default (frozen) encoder, then marked as pretrained, which only changes what is sent.
        config.vision_encoder_name = "some/pretrained-backbone"
    algorithm = SACAlgorithm(policy=policy, config=SACAlgorithmConfig.from_policy_config(config))
    return algorithm, policy


def frozen_names(module: torch.nn.Module) -> set[str]:
    return {name for name, param in module.named_parameters() if not param.requires_grad}


def test_weight_push_leaves_out_a_frozen_pretrained_backbone():
    learner, learner_policy = make_algorithm(pretrained_backbone=True)
    actor, actor_policy = make_algorithm(pretrained_backbone=True)
    # both sides load the same pretrained backbone
    actor_policy.actor.encoder.image_encoder.load_state_dict(
        learner_policy.actor.encoder.image_encoder.state_dict()
    )

    weights = learner.get_weights()
    frozen = frozen_names(learner_policy.actor)
    assert frozen and not frozen & set(weights["policy"])
    assert set(weights["policy"]) == set(learner_policy.actor.state_dict()) - frozen

    actor.load_weights(weights)
    expected = learner_policy.actor.state_dict()
    for name, value in actor_policy.actor.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    expected = learner_policy.discrete_critic.state_dict()
    for name, value in actor_policy.discrete_critic.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_weight_push_keeps_a_frozen_backbone_that_is_not_pretrained():
    learner, learner_policy = make_algorithm(pretrained_backbone=False)
    weights = learner.get_weights()
    assert set(weights["policy"]) == set(learner_policy.actor.state_dict())


def test_loading_weights_with_a_missing_trainable_tensor_fails():
    learner, learner_policy = make_algorithm(pretrained_backbone=True)
    actor, _ = make_algorithm(pretrained_backbone=True)
    weights = learner.get_weights()
    trainable = next(n for n, p in learner_policy.actor.named_parameters() if p.requires_grad)
    del weights["policy"][trainable]
    with pytest.raises(RuntimeError, match="do not match"):
        actor.load_weights(weights)
