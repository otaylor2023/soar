#!/usr/bin/env python3
import ray
from ray.rllib.algorithms.appo import APPO
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
import json
from datetime import datetime
import os
import numpy as np
import argparse

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def format_action_params(action_type, params):
    if action_type == 0:  # No-op
        return "No parameters"
    elif action_type == 1:  # Move
        entity_id = params[0]
        spline_points = []
        for i in range(3):
            x = params[1 + i*2]
            y = params[2 + i*2]
            spline_points.append(f"({x:.2f}, {y:.2f})")
        return f"Entity {entity_id:.2f}, Spline points: {' → '.join(spline_points)}"
    elif action_type == 2:  # RTB
        return f"Entity {params[0]:.2f}"
    elif action_type == 3:  # Engage
        engagement_levels = ["Defensive", "Balanced", "Aggressive", "Very Aggressive"]
        weapons_usage = ["Tight", "Standard", "Free Fire"]
        eng_level = min(int(params[2] * 4), 3)  # Scale to 0-3
        weap_usage = min(int(params[3] * 3), 2)  # Scale to 0-2
        return f"Source: {params[0]:.2f}, Target: {params[1]:.2f}, Level: {engagement_levels[eng_level]}, Weapons: {weapons_usage[weap_usage]}"
    return str(params)

def get_action_type_string(action):
    action_type = action["action_type"]
    action_types = {
        0: "NO-OP",
        1: "MOVE",
        2: "RTB",
        3: "ENGAGE"
    }
    return action_types.get(action_type, "UNKNOWN")

def evaluate_model(checkpoint_path, model_type="ppo", num_episodes=5):
    # Register environment and models
    register_env("FlagFrenzyEnv-v0", env_creator)
    ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
    ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

    ray.init(ignore_reinit_error=True)

    # Restore the trained algorithm based on model type
    if model_type.lower() == "appo":
        algo = APPO.from_checkpoint(checkpoint_path)
    else:  # PPO or PPO with curiosity
        algo = PPO.from_checkpoint(checkpoint_path)

    # Create a directory for evaluation results if it doesn't exist
    eval_dir = "replay_evaluations_"
    os.makedirs(eval_dir, exist_ok=True)

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    episode_infos = []
    wins = 0
    
    env = algo.env_creator(algo.config.env_config)

    for i in range(num_episodes):
        print(f"\nEpisode {i + 1}/{num_episodes}")
        episode_reward = 0
        episode_info = []
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        mission_success = False
        
        while not (done or truncated):
            action = algo.compute_single_action(obs)
            action_type = action["action_type"]
            action_name = get_action_type_string(action)
            param_desc = format_action_params(action_type, action["params"])
            print(f"Action: {action_name} | {param_desc}")
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Check for mission success (flagship destroyed)
            if info.get("flagship_destroyed", False):
                mission_success = True
            
            # Store step information
            step_info = {
                "step": step,
                "action": convert_to_json_serializable(action),
                "reward": float(reward),
                "info": {k: convert_to_json_serializable(v) for k, v in info.items() if k != 'valid_engage_mask'}
            }
            episode_info.append(step_info)
            
            print(f"Step {step}: Reward = {reward:.2f}")
        
        if mission_success:
            wins += 1
            print(f"Episode {i + 1} WON - Flagship destroyed!")
        else:
            print(f"Episode {i + 1} LOST - Flagship not destroyed")
            
        episode_rewards.append(float(episode_reward))
        episode_lengths.append(int(step))
        episode_infos.append(episode_info)
        print(f"Episode {i + 1} finished with total reward: {episode_reward:.2f}, length: {step}")

    # Calculate statistics
    avg_reward = float(sum(episode_rewards) / len(episode_rewards))
    avg_length = float(sum(episode_lengths) / len(episode_lengths))
    win_rate = float(wins / num_episodes)

    # Save evaluation results
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    results = {
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "num_episodes": num_episodes,
        "average_reward": avg_reward,
        "average_episode_length": avg_length,
        "win_rate": win_rate,
        "total_wins": wins,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_infos": episode_infos
    }

    results_file = os.path.join(eval_dir, f"{model_type}_{timestamp}_eval.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation Results:")
    print(f"Model Type: {model_type}")
    print(f"Win Rate: {win_rate:.2%} ({wins}/{num_episodes} episodes)")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Results saved to: {results_file}")

    # Clean up
    env.close()
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--model-type", type=str, default="ppo", choices=["ppo", "appo", "curious"], 
                      help="Type of model to evaluate (ppo, appo, or curious)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.model_type, args.episodes)
