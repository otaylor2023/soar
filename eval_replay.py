import json
import os
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
from env.flag_frenzy_env import FlagFrenzyEnv
import datetime

# Register model and action distribution
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

def load_checkpoint(checkpoint_path):
    """Load trained model from checkpoint."""
    algo = PPO(
        config={
            "env": "FlagFrenzyEnv-v0",
            "framework": "torch",
            "model": {
                "custom_model": "flag_frenzy_model",
                "custom_action_dist": "hybrid_action_dist",
            }
        }
    )
    algo.restore(checkpoint_path)
    return algo

def process_replay_state(replay_data, frame_index):
    """Convert replay state to model observation format."""
    # Initialize observation components
    max_entities = 100
    entity_feat_dim = 26
    mission_dim = 7
    
    entities = np.zeros((max_entities, entity_feat_dim), dtype=np.float32)
    mission = np.zeros(mission_dim, dtype=np.float32)
    visibility = {
        "legacy": np.zeros(max_entities, dtype=np.float32),
        "dynasty": np.zeros(max_entities, dtype=np.float32)
    }
    valid_engage_mask = np.zeros((max_entities, max_entities), dtype=np.float32)
    entity_id_list = []
    
    # Process entities and their states
    for event in replay_data["MissionEvents"]:
        if event["FrameIndex"] == frame_index:
            if event["EventName"] == "SimMissionEvent_SpawnUnit":
                entity_idx = len(entity_id_list)
                if entity_idx < max_entities:
                    entity_id_list.append(event["EntityId"])
                    
                    # Fill entity features (position, velocity, etc.)
                    entities[entity_idx] = create_entity_features(event)
                    
                    # Set visibility based on faction
                    if event["Faction"] == 0:  # Legacy
                        visibility["legacy"][entity_idx] = 1.0
                    else:  # Dynasty
                        visibility["dynasty"][entity_idx] = 1.0
    
    return {
        "entities": entities,
        "mission": mission,
        "visibility": visibility,
        "valid_engage_mask": valid_engage_mask.flatten(),
        "entity_id_list": np.array(entity_id_list, dtype=np.float32)
    }

def create_entity_features(event):
    """Create feature vector for an entity."""
    features = np.zeros(26, dtype=np.float32)
    
    # Basic features from event data
    features[0] = event["EntityId"]
    features[1] = event["Faction"]
    
    # Position
    if "Location" in event:
        features[2] = event["Location"]["X"]
        features[3] = event["Location"]["Y"]
        features[4] = event["Location"]["Z"]
    
    # Velocity
    if "Vel" in event:
        features[5] = event["Vel"]["X"]
        features[6] = event["Vel"]["Y"]
        features[7] = event["Vel"]["Z"]
    
    # Entity type features from DISEntityType
    if "DISEntityType" in event:
        dis_type = event["DISEntityType"]
        features[8] = dis_type["Kind"]
        features[9] = dis_type["Domain"]
        features[10] = dis_type["Country"]
        features[11] = dis_type["Category"]
        features[12] = dis_type["SubCategory"]
        features[13] = dis_type["Specific"]
        features[14] = dis_type["Extra"]
    
    return features

def evaluate_replay(replay_path, checkpoint_path, log_dir="evaluation_logs"):
    """Evaluate model decisions on a replay file."""
    # Load replay data
    with open(replay_path, 'r') as f:
        replay_data = json.load(f)
    
    # Load model
    algo = load_checkpoint(checkpoint_path)
    
    # Process each frame
    max_frames = max(event["FrameIndex"] for event in replay_data["MissionEvents"])
    
    print(f"Evaluating replay: {os.path.basename(replay_path)}")
    print(f"Total frames: {max_frames}")
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    replay_name = os.path.splitext(os.path.basename(replay_path))[0]
    log_file = os.path.join(log_dir, f"{replay_name}_eval_{timestamp}.log")
    
    with open(log_file, 'w') as log:
        log.write(f"Evaluation of {replay_path}\n")
        log.write(f"Using model checkpoint: {checkpoint_path}\n")
        log.write(f"Total frames: {max_frames}\n")
        log.write(f"Timestamp: {timestamp}\n")
        log.write("-" * 80 + "\n\n")
        
        for frame in range(max_frames + 1):
            # Convert replay state to observation
            obs = process_replay_state(replay_data, frame)
            
            # Get model's action
            action = algo.compute_single_action(
                observation=obs,
                explore=False  # Deterministic evaluation
            )
            
            # Parse action
            action_type = action[0]  # Discrete action type
            params = action[1:]      # Continuous parameters
            
            # Map action type to name for better readability
            action_names = {
                0: "NoOp",
                1: "Move",
                2: "MoveFormation",
                3: "Engage",
                4: "FollowPath"
            }
            action_name = action_names.get(action_type, f"Unknown_{action_type}")
            
            # Log frame, action type and parameters
            log_entry = f"Frame {frame}: Action={action_name} "
            
            # Add specific details based on action type
            if action_type == 3:  # Engage action
                target_idx = np.argmax(params[1:100])  # First param is source
                entity_id = -1
                target_id = -1
                if len(obs['entity_id_list']) > 0:
                    entity_id = obs['entity_id_list'][0]  # First entity is source
                if len(obs['entity_id_list']) > target_idx:
                    target_id = obs['entity_id_list'][target_idx]
                log_entry += f"Source={entity_id} Target={target_id}"
                print(f"Frame {frame}: Model would engage entity {target_id}")
            elif action_type == 1:  # Move action
                dest_x, dest_y = params[0], params[1]
                log_entry += f"Destination=({dest_x:.2f}, {dest_y:.2f})"
            
            # Write to log file
            log.write(log_entry + "\n")
            
            # Find and log events that occurred in this frame
            frame_events = [event for event in replay_data["MissionEvents"] if event["FrameIndex"] == frame]
            if frame_events:
                log.write(f"  Events in frame {frame}:\n")
                for event in frame_events:
                    event_type = event.get("EventName", "Unknown")
                    entity_id = event.get("EntityId", -1)
                    log.write(f"  - {event_type} (EntityId: {entity_id})\n")
            
            # Add separator between frames for readability
            if frame_events:
                log.write("\n")
    
    print(f"Evaluation log saved to: {log_file}")
    return log_file

if __name__ == "__main__":
    # Example usage
    replay_path = "Replays/22-04-2025 14-30-18.json"
    checkpoint_path = "ray_results/rewardshape_flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_74231_00000_0_2025-04-22_13-51-44/checkpoint_000009"
    evaluate_replay(replay_path, checkpoint_path)
