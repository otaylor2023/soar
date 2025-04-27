#!/usr/bin/env python3
"""
Script to run multiple episodes with a trained agent and record actions and probabilities.

This script loads a trained agent from a checkpoint, runs it through multiple episodes,
and records detailed information about each step, including:
- Actions taken and their parameters
- Action probabilities from the policy
- Rewards received
- Model state information (value estimates, logits)
- Mission state

Results are saved both as individual episode files and a summary file.
"""

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


def run_episodes(checkpoint_path, num_episodes=5, output_dir=None, render=False, explore=True):
    """Run multiple episodes with the trained agent and record actions and probabilities."""
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Running {num_episodes} episodes")
    print(f"Exploration enabled: {explore}")
    if not explore:
        print("Running in deterministic mode (no exploration)")
    
    # Load the algorithm from checkpoint
    algo = load_algorithm(checkpoint_path)
    
    # Get the policy object directly for probability calculations
    policy = algo.get_policy()
    
    # Create output directory if not exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_dir = f"agent_runs_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
            # Get action and compute probabilities
            action_probs = None
            try:
                # Use the prepare_observation_for_model function from eval_replay
                formatted_obs = {}
                try:
                    # Prepare observation tensors
                    formatted_obs = {
                        "entities": torch.FloatTensor(obs["entities"]).unsqueeze(0).to(device),
                        "visibility": {
                            "legacy": torch.FloatTensor(obs["visibility"]["legacy"]).unsqueeze(0).to(device),
                            "dynasty": torch.FloatTensor(obs["visibility"]["dynasty"]).unsqueeze(0).to(device)
                        },
                        "mission": torch.FloatTensor(obs["mission"]).unsqueeze(0).to(device),
                        "controllable_entities": torch.FloatTensor(obs["controllable_entities"]).unsqueeze(0).to(device),
                        "entity_id_list": obs["entity_id_list"]
                    }
                    
                    if "valid_engage_mask" in obs:
                        formatted_obs["valid_engage_mask"] = torch.FloatTensor(obs["valid_engage_mask"]).unsqueeze(0).to(device)
                except Exception as e:
                    print(f"  Warning: Error formatting observation: {e}")
                
                # Create input dict for the policy model
                input_dict = {"obs": formatted_obs}
                
                # Compute action through the algorithm
                action = algo.compute_single_action(
                    observation=obs,
                    explore=explore,  # Use the explore parameter
                    policy_id="default_policy"
                )
                
                # Separately compute probabilities and model state
                with torch.no_grad():
                    try:
                        # Forward pass through the model to get logits
                        model_output, state_values = policy.model(input_dict)
                        
                        # Extract action logits (first 4 values) and convert to probabilities
                        action_logits = model_output[0, :4].cpu().numpy()
                        action_probs = torch.softmax(torch.tensor(action_logits), dim=0).numpy()
                        
                        # Get value function estimate
                        value_estimate = policy.model.value_function().cpu().numpy()
                        
                        # Save model state values
                        model_state = {
                            "value_estimate": float(value_estimate[0]) if len(value_estimate.shape) > 0 else float(value_estimate),
                            "action_logits": action_logits.tolist()
                        }
                        
                        # If parameters are available, save them too
                        if model_output.shape[1] > 4:
                            param_values = model_output[0, 4:].cpu().numpy()
                            model_state["param_values"] = param_values.tolist()
                    except Exception as e:
                        print(f"  Warning: Error computing model outputs: {e}")
                        model_state = {"error": str(e)}
                        # Fall back to simpler computation
                        pass
            except Exception as e:
                print(f"  Warning: Error in action computation: {e}")
                # If we couldn't compute action through policy, use a simple no-op action
                if isinstance(action, dict):
                    print("  Action already computed")
                else:
                    action = {"action_type": 0, "params": np.zeros(10)}
            
            # Execute action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record step data
            try:
                # Handle action data based on its structure
                if isinstance(action, dict):
                    action_data = {
                        "action_type": int(action["action_type"]),
                        "action_name": get_action_type_name(action["action_type"]),
                        "params": action["params"].tolist() if isinstance(action["params"], np.ndarray) else action["params"]
                    }
                else:
                    # If action is not a dict, log what we can
                    print(f"  Warning: action is of type {type(action)}, not dict")
                    action_data = {
                        "raw_action": str(action)
                    }
                
                # Create step data
                step_data = {
                    "step": step,
                    "action": action_data,
                    "reward": float(reward),
                    "action_probabilities": action_probs.tolist() if action_probs is not None else None,
                }
                
                # Add model state information if available
                if 'model_state' in locals() and model_state is not None:
                    step_data["model_state"] = model_state
                
                # Add observation data if available
                if isinstance(obs, dict) and "mission" in obs:
                    step_data["observation"] = {
                        "mission": obs["mission"].tolist() if isinstance(obs["mission"], np.ndarray) else obs["mission"],
                        # Only include parts of observation that are useful for analysis
                        # Full observation can be very large
                    }
            except Exception as e:
                print(f"  Warning: Error recording step data: {e}")
                step_data = {
                    "step": step,
                    "error": str(e),
                    "reward": float(reward)
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
    parser = argparse.ArgumentParser(
        description="Run episodes with a trained agent and record action probabilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", 
        required=True, 
        help="Path to checkpoint directory or file"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=5, 
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--output", 
        default=None, 
        help="Directory to save episode data (default: agent_runs_TIMESTAMP)"
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Save replays of the episodes"
    )
    parser.add_argument(
        "--explore", 
        action="store_true",
        help="Enable exploration (stochastic policy)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Run episodes with the agent
    run_episodes(
        checkpoint_path=args.checkpoint, 
        num_episodes=args.episodes, 
        output_dir=args.output, 
        render=args.render,
        explore=args.explore
    )