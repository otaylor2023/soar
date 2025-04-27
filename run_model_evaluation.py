"""
Run a trained model in the FlagFrenzy environment for evaluation.

This script loads a trained model from a checkpoint and runs it in the environment.
"""

import os
import json
import torch
import numpy as np
import glob
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from models.model_oter import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
from ray.rllib.models import ModelCatalog
from register_env import env_creator
import datetime
import argparse

# Register model components (needed for loading checkpoints)
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

# Register the environment
register_env("FlagFrenzyEnv-v0", env_creator)

def run_evaluation(checkpoint_path, replay_path, max_frames=None):
    """
    Run a trained model on a replay file for evaluation.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        replay_path: Path to the replay file
        max_frames: Maximum number of frames to process (None=all)
    """
    print(f"\nLoading model from: {checkpoint_path}")
    # Initialize algorithm with the checkpoint
    algo = PPO(
        config={
            "env": "FlagFrenzyEnv-v0",
            "framework": "torch",
            "model": {
                "custom_model": "flag_frenzy_model",
                "custom_action_dist": "hybrid_action_dist",
            },
            "explore": False,  # Deterministic evaluation
        }
    )
    
    try:
        algo.restore(checkpoint_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load the replay file
    print(f"\nLoading replay from: {replay_path}")
    try:
        with open(replay_path, 'r') as f:
            replay_data = json.load(f)
        print(f"Replay file loaded successfully")
    except Exception as e:
        print(f"Error loading replay file: {e}")
        return None
    
    # Check if we have mission events
    if "MissionEvents" not in replay_data:
        print(f"Error: Replay file doesn't contain MissionEvents")
        return None
    
    # Extract all frame indices from the events in the replay
    frame_indices = [event.get("FrameIndex", 0) for event in replay_data.get("MissionEvents", [])]
    if not frame_indices:
        print(f"Error: No frames found in replay file")
        return None
    
    # Get unique frame indices and count events per frame    
    unique_frames = sorted(set(frame_indices))
    events_per_frame = {frame: len([e for e in replay_data["MissionEvents"] if e.get("FrameIndex") == frame]) 
                       for frame in unique_frames}
    
    max_frame = max(unique_frames) if unique_frames else 0
    total_events = len(replay_data.get("MissionEvents", []))
    
    print(f"Replay contains {len(unique_frames)} unique frames, with {total_events} total events")
    print(f"Frame indices range from {min(unique_frames)} to {max_frame}")
    
    # The environment runs at 600 FPS, so let's sample key frames
    # We want frames where significant events happened (more events in that frame)
    
    # Sort frames by number of events (most events first)
    significant_frames = sorted(events_per_frame.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top frames with the most events
    selected_frames = [frame for frame, count in significant_frames[:10] if count > 1]
    
    # If we don't have enough significant frames, add some regularly spaced frames
    if len(selected_frames) < 5 and len(unique_frames) > 5:
        print("Not enough significant frames found, adding some regularly spaced frames")
        # Add frames at regular intervals
        interval = max(1, len(unique_frames) // 5)
        for i in range(0, len(unique_frames), interval):
            if unique_frames[i] not in selected_frames:
                selected_frames.append(unique_frames[i])
                if len(selected_frames) >= 10:
                    break
    
    # Make sure frames are in chronological order
    selected_frames = sorted(selected_frames)
    print(f"Selected {len(selected_frames)} frames for analysis")
    
    # Print info about selected frames
    for i, frame in enumerate(selected_frames):
        events = [e for e in replay_data["MissionEvents"] if e.get("FrameIndex") == frame]
        print(f"  Frame {i+1}: FrameIndex={frame}, Events={len(events)}")
    
    unique_frames = selected_frames
    
    # Limit the number of frames if specified
    if max_frames is not None:
        print(f"Limiting to {max_frames} frames")
        unique_frames = unique_frames[:max_frames]
    
    # Process each frame
    results = []
    frame_actions = []
    
    print("\nEvaluating model on replay frames:")
    
    # Create a temporary environment to help convert replay frames to observations
    temp_env = env_creator({})
    
    # Process each frame sequentially
    for frame_idx, frame in enumerate(unique_frames):
        print(f"Processing frame {frame_idx+1}/{len(unique_frames)} (FrameIndex: {frame})...")
        
        # Build observation from the replay frame
        try:
            # Get events for this frame
            frame_events = [event for event in replay_data["MissionEvents"] 
                           if event.get("FrameIndex") == frame]
            
            # Convert to observation format
            obs = process_replay_frame(replay_data, frame, temp_env)
            
            if obs is None:
                print(f"  Warning: Could not create observation for frame {frame}")
                continue
                
            # Get model's action
            action = algo.compute_single_action(obs, explore=False)
            
            # Handle either dictionary or array format for actions
            if isinstance(action, dict):
                # Dictionary format
                action_type = action["action_type"]
                action_params = action["params"]
            else:
                # Array format (first element is action type, rest are params)
                action_type = action[0]
                action_params = action[1:]
            
            # Map action type to a readable name
            action_names = {0: "NoOp", 1: "Move", 2: "ReturnToBase", 3: "Engage"}
            action_name = action_names.get(action_type, f"Unknown_{action_type}")
            
            # Print the action
            print(f"  Frame {frame}: Model would take action: {action_name}")
            
            # Track the action
            action_info = {
                "frame": frame,
                "action_type": int(action_type),
                "action_name": action_name,
                "params": action_params.tolist() if hasattr(action_params, "tolist") else action_params,
                "events_in_frame": len(frame_events)
            }
            frame_actions.append(action_info)
            
        except Exception as e:
            print(f"  Error processing frame {frame}: {e}")
    
    # Clean up
    temp_env.close()
    
    # Record the results
    results = {
        "replay_path": replay_path,
        "frames_evaluated": len(frame_actions),
        "actions": frame_actions
    }
    
    # Save the results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Create replay_analysis directory if it doesn't exist
    os.makedirs("replay_analysis", exist_ok=True)
    
    output_file = f"replay_analysis/evaluation_results_{checkpoint_name}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "replay_path": replay_path,
            "frames_analyzed": max_frames,
            "results": results
        }, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to: {output_file}")
    return output_file

def process_replay_frame(replay_data, frame_index, env):
    """Process a replay frame into an observation format the model can use."""
    print(f"Getting a sample observation from the environment...")
    try:
        # Reset the environment to get a fresh observation
        sample_obs, _ = env.reset()
        obs_keys = sample_obs.keys()
        print(f"Observation contains keys: {obs_keys}")
        
        # Create a copy of this observation structure to fill with our data
        obs = {}
        for key in obs_keys:
            if isinstance(sample_obs[key], dict):
                obs[key] = {}
                for subkey in sample_obs[key]:
                    # Use zeros with the same shape
                    obs[key][subkey] = np.zeros_like(sample_obs[key][subkey])
            else:
                # Use zeros with the same shape
                obs[key] = np.zeros_like(sample_obs[key])
        
        # Get all events for this specific frame
        frame_events = [e for e in replay_data["MissionEvents"] if e.get("FrameIndex") == frame_index]
        print(f"Found {len(frame_events)} events for frame {frame_index}")
        
        # Extract entity information from events
        entities = []
        entity_ids = []
        entity_factions = {}
        entity_positions = {}
        
        # First pass: collect entity information
        for event in frame_events:
            if event.get("EventName", "").startswith("SimMissionEvent_Spawn"):
                entity_id = event.get("EntityId")
                if entity_id is not None:
                    entity_ids.append(entity_id)
                    entity_factions[entity_id] = event.get("Faction", 0)
                    
                    if "Location" in event:
                        loc = event["Location"]
                        entity_positions[entity_id] = (loc.get("X", 0), loc.get("Y", 0), loc.get("Z", 0))
        
        # Fill entity data
        if "entities" in obs and entity_ids:
            # We'll only populate data for entities we found
            max_entities = min(len(entity_ids), obs["entities"].shape[0])
            
            # Keep track of indices for entities with faction 0 and 1
            faction0_indices = []
            faction1_indices = []
            
            # Fill entity features with basic information
            for i in range(max_entities):
                entity_id = entity_ids[i]
                faction = entity_factions.get(entity_id, 0)
                
                # Track indices by faction
                if faction == 0:
                    faction0_indices.append(i)
                else:
                    faction1_indices.append(i)
                
                # Basic entity data - normalize values
                obs["entities"][i, 0] = entity_id / 1000.0  # Normalize ID
                obs["entities"][i, 1] = faction  # Faction value
                
                # Position if available
                if entity_id in entity_positions:
                    x, y, z = entity_positions[entity_id]
                    obs["entities"][i, 2] = x / 1000000.0  # Normalize position
                    obs["entities"][i, 3] = y / 1000000.0
                    obs["entities"][i, 4] = z / 1000000.0
                
                # Set alive status to 1 (assuming all entities in frame are alive)
                alive_index = 13  # Approximate index based on env code
                if obs["entities"].shape[1] > alive_index:
                    obs["entities"][i, alive_index] = 1.0
            
            # Set visibility based on faction
            if "visibility" in obs and isinstance(obs["visibility"], dict):
                if "legacy" in obs["visibility"]:
                    # Faction 0 (Legacy) can see faction 1 (Dynasty) entities
                    for idx in faction1_indices:
                        if idx < len(obs["visibility"]["legacy"]):
                            obs["visibility"]["legacy"][idx] = 1
                
                if "dynasty" in obs["visibility"]:
                    # Faction 1 (Dynasty) can see faction 0 (Legacy) entities
                    for idx in faction0_indices:
                        if idx < len(obs["visibility"]["dynasty"]):
                            obs["visibility"]["dynasty"][idx] = 1
            
            # Set controllable entities (assume entities with IDs 11 and 35 are controllable)
            if "controllable_entities" in obs:
                for i, entity_id in enumerate(entity_ids):
                    if i < len(obs["controllable_entities"]) and entity_id in [11, 35]:
                        obs["controllable_entities"][i] = 1
        
        # Set mission status vector - time is the most reliable value
        if "mission" in obs:
            # Default values for mission vector
            obs["mission"] = np.zeros_like(obs["mission"])
            
            # Set time value (normalized)
            if obs["mission"].shape[0] > 6:
                obs["mission"][6] = frame_index / 9000.0  # Normalize by max game time
            
            # Set flag alive status (assume flag is alive unless we find it's destroyed)
            # This is just a placeholder since we don't have that information
            if obs["mission"].shape[0] > 0:
                obs["mission"][0] = 1.0  # Flag is alive by default
        
        # If entity_id_list is required
        if "entity_id_list" in obs and entity_ids:
            # Create a numpy array of entity IDs
            entity_id_array = np.array(entity_ids, dtype=np.float32)
            
            # Ensure it matches the expected shape
            if len(entity_id_array) <= len(obs["entity_id_list"]):
                obs["entity_id_list"][:len(entity_id_array)] = entity_id_array
            else:
                obs["entity_id_list"] = entity_id_array[:len(obs["entity_id_list"])]
        
        print(f"Created observation with {len(entity_ids)} entities")
                
        return obs
    
    except Exception as e:
        print(f"Error creating sample observation: {e}")
        return None

if __name__ == "__main__":
    # Configuration parameters (edit these directly)
    CHECKPOINT_PATH = "ray_replay/flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_26b0e_00000_0_2025-04-19_09-13-57/checkpoint_000029"
    
    # Get the most recent replay file
    replay_files = sorted(glob.glob("Replays/*.json"), key=os.path.getmtime, reverse=True)
    REPLAY_PATH = replay_files[0] if replay_files else "Replays/27-04-2025 04-04-17.json"
    print(f"Using most recent replay file: {REPLAY_PATH}")
    
    # Process up to 10 significant frames from the replay
    MAX_FRAMES = 10
    
    # Run the evaluation
    run_evaluation(CHECKPOINT_PATH, REPLAY_PATH, MAX_FRAMES)
