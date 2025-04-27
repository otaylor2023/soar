#!/usr/bin/env python3
# Training script using PPO with Intrinsic Curiosity Model (ICM)
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from register_env import env_creator
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
import os

# Register environment and models
register_env("FlagFrenzyEnv-v0", env_creator)
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
ray.init(ignore_reinit_error=True)

class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):
        # Track engagement actions
        info = episode.last_info_for()
        if info:
            if info.get("engage_action_taken"):
                episode.custom_metrics["engage_action_count"] = 1
            else:
                episode.custom_metrics["engage_action_count"] = 0

# Config for PPO with curiosity-driven exploration
config = (
    PPOConfig()
    .environment(env="FlagFrenzyEnv-v0", env_config={})
    .framework("torch")
    .rollouts(
        num_rollout_workers=24,
        num_envs_per_worker=1,
        rollout_fragment_length="auto",
    )
    .training(
        model={
            "custom_model": "flag_frenzy_model",
            "custom_action_dist": "hybrid_action_dist",
            "vf_share_layers": False,
            # ICM configuration
            "custom_model_config": {
                "use_curiosity": True,
                "feature_dim": 288,
                "forward_net_hiddens": [256, 256],
                "inverse_net_hiddens": [256, 256],
                "curiosity_lr": 0.0005,
                "curiosity_weight": 0.05,  # Weight for intrinsic rewards
                "forward_loss_weight": 0.2,  # Weight between forward and inverse dynamics losses
            }
        },
        gamma=0.995,
        lambda_=0.95,
        kl_coeff=0.2,
        clip_param=0.3,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        lr=3e-5,
    )
    .resources(num_gpus=1)
    .experimental(_enable_new_api_stack=False)
    .debugging(log_level="ERROR")
    .callbacks(CustomMetricsCallback)
)

# Set horizon
config["horizon"] = 900

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={
            "training_iteration": 150,
            "timesteps_total": 5000000,
            "episode_reward_mean": 200
        },
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=15,
            num_to_keep=5),
        name="ppo_curiosity_2",
        log_to_file=True),
).fit()
