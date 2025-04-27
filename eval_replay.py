#!/usr/bin/env python3
# Script to evaluate a replay and output action probabilities

import argparse
import json
import numpy as np
import os
import torch
from pathlib import Path

# Import model and action distribution
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import register_env

# Import custom models and env
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
from env.flag_frenzy_env import FlagFrenzyEnv
from register_env import env_creator

# Import simulation interfaces
import env.SimulationInterface as SimulationInterface
from env.SimulationInterface import EntitySpawned, Victory, AdversaryContact


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


def load_policy(checkpoint_path):
    """Load policy from a checkpoint."""
    # Register custom components
    ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
    ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
    register_env("FlagFrenzyEnv-v0", env_creator)
    
    # Load the algorithm with the checkpoint
    algo = PPO(config=load_config(checkpoint_path))
    algo.restore(checkpoint_path)
    
    return algo.get_policy()


def get_action_type_name(action_type):
    """Convert action type index to name."""
    action_types = ["NO_OP", "MOVE", "RETURN_TO_BASE", "ENGAGE_TARGET"]
    if 0 <= action_type < len(action_types):
        return action_types[action_type]
    return f"UNKNOWN_{action_type}"


def prepare_observation_for_model(obs):
    """Properly prepare the observation for model input.
    
    This ensures the tensors are correctly formatted to avoid the flatten() error.
    """
    # Convert numpy arrays to torch tensors with correct dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process entities tensor
    entities = torch.FloatTensor(obs["entities"]).to(device)
    if entities.dim() == 2:  # Add batch dimension if missing
        entities = entities.unsqueeze(0)
    
    # Process visibility
    legacy = torch.FloatTensor(obs["visibility"]["legacy"]).to(device)
    dynasty = torch.FloatTensor(obs["visibility"]["dynasty"]).to(device)
    if legacy.dim() == 1:
        legacy = legacy.unsqueeze(0)
    if dynasty.dim() == 1:
        dynasty = dynasty.unsqueeze(0)
    
    # Process mission
    mission = torch.FloatTensor(obs["mission"]).to(device)
    if mission.dim() == 1:
        mission = mission.unsqueeze(0)
    
    # Process controllable entities
    controllable_entities = torch.FloatTensor(obs["controllable_entities"]).to(device)
    if controllable_entities.dim() == 1:
        controllable_entities = controllable_entities.unsqueeze(0)
    
    # Process engage mask if present
    if "valid_engage_mask" in obs:
        engage_mask = torch.FloatTensor(obs["valid_engage_mask"]).to(device)
        if engage_mask.dim() == 2:
            engage_mask = engage_mask.unsqueeze(0)
    else:
        engage_mask = None
    
    # Process entity_id_list
    entity_id_list = obs["entity_id_list"]
    
    # Reconstruct observation with proper tensor dimensions
    formatted_obs = {
        "entities": entities,
        "visibility": {
            "legacy": legacy,
            "dynasty": dynasty
        },
        "mission": mission,
        "controllable_entities": controllable_entities,
        "entity_id_list": entity_id_list
    }
    
    if engage_mask is not None:
        formatted_obs["valid_engage_mask"] = engage_mask
    
    return formatted_obs


def evaluate_replay(replay_path, checkpoint_path, output_path=None):
    """Evaluate a replay file against a trained policy."""
    print(f"Evaluating replay: {replay_path}")
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load the replay
    with open(replay_path, 'r') as f:
        replay_data = json.load(f)
    
    print(f"Loaded replay: {replay_data.get('Name', 'Unknown')}")
    print(f"Random Seed: {replay_data.get('RandomSeed', 'Unknown')}")
    
    # Initialize environment
    env = FlagFrenzyEnv(save_replay=False)
    env.reset()
    
    # Load the policy
    policy = load_policy(checkpoint_path)
    
    # Process mission events to rebuild state
    results = []
    events = replay_data.get("MissionEvents", [])
    player_events = replay_data.get("PlayerEvents", [])
    
    print(f"Processing {len(events)} mission events and {len(player_events)} player events")
    
    # Sort all events by frame index
    all_events = events + player_events
    all_events.sort(key=lambda x: x.get("FrameIndex", 0))
    
    # Create a mapping of frame indices to player actions
    frame_to_action = {}
    for event in player_events:
        frame_idx = event.get("FrameIndex", 0)
        action_data = {
            "event_name": event.get("EventName", "Unknown"),
            "entity_id": event.get("EntityId", -1),
        }
        
        # Add additional data based on event type
        if event.get("EventName") == "SimPlayerEvent_Move":
            action_data["action_type"] = 1  # MOVE
            action_data["spline"] = event.get("Spline", [])
        elif event.get("EventName") == "SimPlayerEvent_RTB":
            action_data["action_type"] = 2  # RETURN_TO_BASE
        elif event.get("EventName") == "SimPlayerEvent_Commit":
            action_data["action_type"] = 3  # ENGAGE_TARGET
            action_data["target_id"] = event.get("TargetGroupId", -1)
            action_data["engagement_level"] = event.get("EngagementLevel", 0)
            action_data["weapon_usage"] = event.get("WeaponUsage", 0)
        else:
            action_data["action_type"] = 0  # NO_OP
            
        frame_to_action[frame_idx] = action_data
    
    # Process events frame by frame
    prev_frame = -1
    for event in all_events:
        frame_idx = event.get("FrameIndex", 0)
        
        # Only evaluate at frames where player actions occurred
        if frame_idx != prev_frame and frame_idx in frame_to_action:
            # Get current observation
            observation = env._get_observations()
            
            try:
                # Prepare observation for model
                formatted_obs = prepare_observation_for_model(observation)
                
                # Create input dictionary
                input_dict = {SampleBatch.OBS: formatted_obs}
                
                # Get action distribution from policy
                with torch.no_grad():
                    # Forward pass through the model
                    model_output, _ = policy.model(input_dict)
                    
                    # Extract outputs - first 4 values are action logits, rest are parameters
                    action_logits = model_output[:, :4].cpu().numpy()
                    action_probs = torch.softmax(model_output[:, :4], dim=-1).cpu().numpy()
                    action_params = model_output[:, 4:].cpu().numpy()
                    
                    # Get value estimate
                    value = policy.model.value_function().cpu().numpy()
                    
                    # Record the player action
                    player_action = frame_to_action.get(frame_idx, {"action_type": -1})
                    
                    # Store results
                    step_result = {
                        "frame": frame_idx,
                        "action_logits": action_logits.tolist()[0] if len(action_logits.shape) > 1 else action_logits.tolist(),
                        "action_probabilities": action_probs.tolist()[0] if len(action_probs.shape) > 1 else action_probs.tolist(),
                        "action_parameters": action_params.tolist()[0] if len(action_params.shape) > 1 else action_params.tolist(),
                        "value_estimate": value.tolist()[0] if isinstance(value, np.ndarray) and len(value.shape) > 0 else value.tolist(),
                        "player_action": {
                            "action_type": player_action.get("action_type", -1),
                            "action_name": get_action_type_name(player_action.get("action_type", -1)),
                            "details": player_action
                        }
                    }
                    
                    results.append(step_result)
                    print(f"Processed frame {frame_idx} with action {get_action_type_name(player_action.get('action_type', -1))}")
                    
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
            
            prev_frame = frame_idx
        
        # Update environment state based on event type
        if event in events:
            try:
                # Simulate mission event in the environment
                if hasattr(env, "process_simulation_events"):
                    env.process_simulation_events([event])
            except Exception as e:
                print(f"Error processing mission event: {str(e)}")
        
    # Save or return results
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    print(f"Evaluation complete. Processed {len(results)} action frames.")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a replay against a trained policy")
    parser.add_argument("--replay", required=True, help="Path to replay file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory or file")
    parser.add_argument("--output", default=None, help="Path to save output results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_replay(args.replay, args.checkpoint, args.output)
