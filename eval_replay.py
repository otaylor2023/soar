import json
import os
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
from env.flag_frenzy_env import FlagFrenzyEnv

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

def evaluate_replay(replay_path, checkpoint_path):
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
        
        # Print interesting actions (e.g., engagements)
        if action_type == 3:  # Engage action
            target_idx = np.argmax(params[1:100])  # First param is source
            print(f"Frame {frame}: Model would engage entity {obs['entity_id_list'][target_idx]}")

if __name__ == "__main__":
    # Example usage
    replay_path = "Replays/18-04-2025 20-23-49.json"
    checkpoint_path = "ray_results/rewardshape_flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_74231_00000_0_2025-04-22_13-51-44/checkpoint_000009"
    evaluate_replay(replay_path, checkpoint_path)
