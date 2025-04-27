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
from utils import format_explanation
from llm_interpreter import LLMInterpreter

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

def evaluate_model(checkpoint_path, model_type="ppo", num_episodes=1, interpret=False):
    """Evaluate a trained model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_type: Type of model to evaluate (ppo, appo, or curious)
        num_episodes: Number of episodes to evaluate
        interpret: Whether to use LLM interpretation of actions
    """
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

    # Enable attribution
    algo.get_policy().model.set_attribution_enabled(True)

    # Initialize LLM interpreter if enabled
    if interpret:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        interpreter = LLMInterpreter(api_key=api_key)
    else:
        interpreter = None

    # Create directories for evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = f"eval_results_{timestamp}"
    interpretations_dir = os.path.join(eval_dir, "interpretations")
    os.makedirs(interpretations_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    episode_infos = []
    
    env = algo.env_creator(algo.config.env_config)
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        if interpret:
            interpreter.start_new_episode()  # Start tracking new episode
        
        episode_reward = 0
        step = 0
        obs, info = env.reset()
        done = truncated = False
        
        while not (done or truncated):
            action = algo.compute_single_action(obs)
            policy = algo.get_policy()
            model = policy.model
            action_type = action["action_type"]
            action_name = get_action_type_string(action)
            param_desc = format_action_params(action_type, action["params"])
            attribution_dict = model.get_last_attribution()
            print(f"Action: {action_name} | {param_desc}")
            print(f"Attribution: {attribution_dict}")

            # Get LLM interpretation if enabled
            if interpret:
                interpretation = interpreter.interpret_flag_frenzy_action(env, action, attribution_dict, obs)
                print(f"\nTactical Analysis:\n{interpretation}\n")
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            print(f"Step {step}: Reward = {reward:.2f}")

        print(f"Episode ended. Total Reward: {episode_reward:.4f}, Length: {step}")
        
        # Save episode interpretations if enabled
        if interpret:
            interpretation_file = interpreter.save_episode_interpretations(interpretations_dir)
            print(f"Saved interpretations to: {interpretation_file}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        episode_infos.append({
            "episode": episode + 1,
            "reward": float(episode_reward),
            "length": step,
            "interpretation_file": interpretation_file if interpret else None
        })

    env.close()
    
    # Save overall evaluation results
    results = {
        "timestamp": timestamp,
        "checkpoint_path": checkpoint_path,
        "model_type": model_type,
        "num_episodes": num_episodes,
        "avg_reward": float(sum(episode_rewards) / len(episode_rewards)),
        "avg_length": float(sum(episode_lengths) / len(episode_lengths)),
        "episodes": episode_infos
    }
    
    results_file = os.path.join(eval_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to: {eval_dir}")
    ray.shutdown()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--model-type", type=str, default="ppo", choices=["ppo", "appo", "curious"], 
                      help="Type of model to evaluate (ppo, appo, or curious)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--interpret", action="store_true", help="Enable LLM interpretation of actions")
    
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.model_type, args.episodes, args.interpret)
