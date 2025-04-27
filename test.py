#!/usr/bin/env python3
# Script to run multiple episodes with a trained agent and record actions and probabilities

import argparse
import json
import numpy as np
import os
import torch
from datetime import datetime
from pathlib import Path

# Import Ray and RLlib components
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Import custom models and env
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
from env.flag_frenzy_env import FlagFrenzyEnv
from register_env import env_creator


def load_config(checkpoint_path):
    """Load the configuration from a checkpoint."""
    config = {
        "env": "FlagFrenzyEnv-v0",
        "framework": "torch",
        "model": {
            "custom_model": "flag_frenzy_model",
            "custom_action_dist": "hybrid_action_dist",
        }
    }
    return config


def load_algorithm(checkpoint_path):
    """Load algorithm from a checkpoint."""
    # Register custom components
    ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
    ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
    register_env("FlagFrenzyEnv-v0", env_creator)
    
    # Load the algorithm with the checkpoint
    algo = PPO(config=load_config(checkpoint_path))
    algo.restore(checkpoint_path)
    
    return algo


def get_action_type_name(action_type):
    """Convert action type index to name."""
    action_types = ["NO_OP", "MOVE", "RETURN_TO_BASE", "ENGAGE_TARGET"]
    if 0 <= action_type < len(action_types):
        return action_types[action_type]
    return f"UNKNOWN_{action_type}"


def run_episodes(checkpoint_path, num_episodes=5, output_dir=None, render=False):
    """Run multiple episodes with the trained agent and record actions and probabilities."""
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Running {num_episodes} episodes")
    
    # Load the algorithm from checkpoint
    algo = load_algorithm(checkpoint_path)
    
    # Create output directory if not exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_dir = f"agent_runs_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    # Track episode results
    all_episodes = []
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"Running episode {episode+1}/{num_episodes}")
        
        # Initialize environment
        env = FlagFrenzyEnv(save_replay=render)
        obs, info = env.reset()
        
        episode_data = {
            "episode_id": episode,
            "steps": [],
            "reward": 0,
            "length": 0,
            "win": False
        }
        
        done = False
        step = 0
        
        # Run episode until done
        while not done:
            # Get action from policy
            action, policy_info = algo.compute_single_action(
                observation=obs,
                explore=True,
                policy_id="default_policy"
            )
            
            # Extract action probabilities from policy info
            action_dist = policy_info.get("action_dist_inputs", None)
            if action_dist is not None:
                # First 4 values are action logits, convert to probabilities
                action_logits = action_dist[:4]
                action_probs = torch.softmax(torch.tensor(action_logits), dim=0).numpy()
            else:
                action_probs = None
            
            # Execute action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record step data
            step_data = {
                "step": step,
                "action": {
                    "action_type": int(action["action_type"]),
                    "action_name": get_action_type_name(action["action_type"]),
                    "params": action["params"].tolist() if isinstance(action["params"], np.ndarray) else action["params"]
                },
                "reward": float(reward),
                "action_probabilities": action_probs.tolist() if action_probs is not None else None,
                "observation": {
                    "mission": obs["mission"].tolist() if isinstance(obs["mission"], np.ndarray) else obs["mission"],
                    # Only include parts of observation that are useful for analysis
                    # Full observation can be very large
                }
            }
            
            # Add any relevant info from the environment
            if "engage_action_taken" in info:
                step_data["engage_info"] = {
                    "entity_id": info.get("engage_entity_id", -1),
                    "target_id": info.get("engage_target_id", -1)
                }
            
            episode_data["steps"].append(step_data)
            episode_data["reward"] += reward
            
            # Update for next step
            obs = next_obs
            step += 1
            
            if step % 100 == 0:
                print(f"  Step {step}, cumulative reward: {episode_data['reward']:.2f}")
        
        # Record final episode stats
        episode_data["length"] = step
        episode_data["win"] = bool(info.get("win", False))
        all_episodes.append(episode_data)
        
        print(f"Episode {episode+1} complete:")
        print(f"  Total reward: {episode_data['reward']:.2f}")
        print(f"  Length: {episode_data['length']}")
        print(f"  Win: {episode_data['win']}")
        
        # Save this episode's data
        episode_filename = os.path.join(output_dir, f"episode_{episode}.json")
        with open(episode_filename, 'w') as f:
            json.dump(episode_data, f, indent=2)
        print(f"  Saved to {episode_filename}")
    
    # Save summary of all episodes
    summary_filename = os.path.join(output_dir, "summary.json")
    summary_data = {
        "checkpoint": checkpoint_path,
        "num_episodes": num_episodes,
        "episodes": [
            {
                "episode_id": ep["episode_id"],
                "reward": ep["reward"],
                "length": ep["length"],
                "win": ep["win"]
            } for ep in all_episodes
        ]
    }
    
    with open(summary_filename, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"All episodes complete. Summary saved to {summary_filename}")
    
    # Calculate and print stats
    rewards = [ep["reward"] for ep in all_episodes]
    wins = [ep["win"] for ep in all_episodes]
    
    print("\nEpisode Statistics:")
    print(f"  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Win rate: {np.mean(wins):.2%}")
    
    return all_episodes


def parse_args():
    parser = argparse.ArgumentParser(description="Run episodes with a trained agent")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory or file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--output", default=None, help="Directory to save episode data")
    parser.add_argument("--render", action="store_true", help="Save replays of the episodes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_episodes(args.checkpoint, args.episodes, args.output, args.render)