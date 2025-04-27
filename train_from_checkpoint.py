import argparse
import ray
from ray import air, tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
import os

# Register model
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

register_env("FlagFrenzyEnv-v0", env_creator)

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

def get_ppo_config(num_workers=24, num_gpus=0):
    """Get the PPO configuration."""
    config = (
        PPOConfig()
        .environment(env="FlagFrenzyEnv-v0", env_config={})
        .framework("torch")
        .rollouts(
            num_rollout_workers=num_workers,
        )
        .training(
            model={
                "custom_model": "flag_frenzy_model",
                "custom_action_dist": "hybrid_action_dist",
            },
            gamma=0.995,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=10.0,
            grad_clip=0.5,
            lr=3e-5,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=20,
        )
        .resources(num_gpus=num_gpus)
        .callbacks(FlagFrenzyCallbacks)
        .to_dict()
    )
    config["horizon"] = 900
    return config

def main():
    parser = argparse.ArgumentParser(description='Train FlagFrenzy model with optional checkpoint loading')
    parser.add_argument('--checkpoint-path', type=str, help='Path to checkpoint to load from', default=None)
    parser.add_argument('--num-workers', type=int, help='Number of rollout workers', default=1)
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use', default=0)
    parser.add_argument('--num-iterations', type=int, help='Number of training iterations', default=1)
    parser.add_argument('--experiment-name', type=str, help='Name for the experiment', default="flag_frenzy_ppo")
    args = parser.parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Restoring from checkpoint: {args.checkpoint_path}")
        # Load checkpoint config
        algo = PPO.from_checkpoint(args.checkpoint_path)
        # Get the config from the loaded algorithm
        config = algo.config.to_dict()
        # Update the config with new parameters
        config["num_rollout_workers"] = args.num_workers
        config["num_gpus"] = args.num_gpus
        # Stop the temporary algorithm
        algo.stop()
        # Create new algorithm with updated config
        algo = PPO(config=config)
    else:
        if args.checkpoint_path:
            print(f"Warning: Checkpoint path {args.checkpoint_path} does not exist")
        # Create new PPO instance with fresh config
        config = get_ppo_config(num_workers=args.num_workers, num_gpus=args.num_gpus)
        algo = PPO(config=config)

    # Train for specified number of iterations
    for i in range(args.num_iterations):
        result = algo.train()
        print(f"\nIteration {i}: Episode reward mean = {result['episode_reward_mean']}")
        
        # Get the policy and its model
        policy = algo.get_policy()
        model = policy.model
        
        # Get and print attributions from the last batch
        if hasattr(model, 'get_last_attribution'):
            attributions = model.get_last_attribution()
            if attributions:
                print("\nSample decisions from this iteration:")
                # Print first 3 attributions from the batch
                for j, attribution in enumerate(attributions[:3]):
                    print(f"\nDecision {j+1}:")
                    explanation = model.format_explanation(attribution)
                    print(explanation)
        
        # Save checkpoint every 15 iterations
        if i % 15 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved at {checkpoint_dir}")

    # Save final checkpoint
    checkpoint_dir = algo.save()
    print(f"Final checkpoint saved at {checkpoint_dir}")
    
    # Cleanup
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main() 