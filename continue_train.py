#!/usr/bin/env python3

# Training script for continuing from checkpoint
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
import os

# Register model and environment
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
register_env("FlagFrenzyEnv-v0", env_creator)

# Initialize Ray
ray.init(ignore_reinit_error=True)

from ray.rllib.algorithms.callbacks import DefaultCallbacks

class FlagFrenzyCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for()
        if info:
            if info.get("engage_action_taken"):
                # Log entity/target ID as custom metrics
                entity_id = info.get("engage_entity_id", -1)
                target_id = info.get("engage_target_id", -1)

                # Aggregate counts
                episode.custom_metrics["engage_action_count"] = 1
                episode.custom_metrics["engage_entity_id"] = entity_id
                episode.custom_metrics["engage_target_id"] = target_id
            else:
                episode.custom_metrics["engage_action_count"] = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        it = result["training_iteration"]
        if it <= 200:  # linear decay 0.01 -> 0.003
            new_coeff = 0.01 - 0.007 * (it / 200)
            algorithm.config["entropy_coeff"] = new_coeff

# Path to the latest checkpoint
latest_checkpoint_dir = "/home/hackathon/doc/soar/ray_results/flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_26b0e_00000_0_2025-04-19_09-13-57/checkpoint_000029"

# Load checkpoint and get its config
print(f"Restoring from checkpoint: {latest_checkpoint_dir}")
temp_algo = PPO.from_checkpoint(latest_checkpoint_dir)
config = temp_algo.config.to_dict()

# Update config with our parameters
config.update({
    "num_rollout_workers": 24,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 20,
    "horizon": 900,
    "num_gpus": 1,
    "callbacks": FlagFrenzyCallbacks,
})

# Stop the temporary algorithm
temp_algo.stop()

# Create new algorithm with updated config and restore checkpoint
algo = PPO(config=config)
algo.restore(latest_checkpoint_dir)

# Train for additional iterations
for i in range(150):  # Train for 150 more iterations
    result = algo.train()
    if i % 15 == 0:  # Save checkpoint every 15 iterations
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved at iteration {i} to {checkpoint_dir}")
    
    # Print metrics
    print(f"Iteration {i}")
    print(f"  Mean Reward: {result['episode_reward_mean']}")
    print(f"  Mean Episode Length: {result['episode_len_mean']}")

# Final save
algo.save()
print("Training completed!")
