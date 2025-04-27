import json
import os
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from models.model_oter import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
import datetime
import argparse

# Register model and action distribution
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)


def load_model_weights(checkpoint_path, export_path=None):
    """Load trained model weights from checkpoint and optionally export the frozen model.
    
    This function extracts just the PyTorch model weights without requiring the full RLlib environment.
    
    Args:
        checkpoint_path: Path to the checkpoint to load
        export_path: If provided, export the frozen model to this path
        
    Returns:
        The loaded CustomModel instance in eval mode
    """
    print(f"Loading model weights from: {checkpoint_path}")
    
    # Handle different types of checkpoint paths
    if os.path.isdir(checkpoint_path):
        # If a directory is provided, try to find the policy_state.pkl file
        if os.path.exists(os.path.join(checkpoint_path, "policies", "default_policy", "policy_state.pkl")):
            state_file = os.path.join(checkpoint_path, "policies", "default_policy", "policy_state.pkl")
            print(f"Found policy state file in directory: {state_file}")
        else:
            # Look for common checkpoint filenames
            options = ["policy_state.pkl", "checkpoint.pt", "model.pt", "params.pkl"]
            found = False
            for option in options:
                potential_path = os.path.join(checkpoint_path, option)
                if os.path.exists(potential_path):
                    state_file = potential_path
                    found = True
                    break
            
            if not found:
                print(f"Warning: No model weights found in {checkpoint_path}")
                state_file = checkpoint_path
    elif os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".json"):
        # If a json checkpoint file is given, try to parse it to find the real weights
        try:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            # Check if this is the main RLlib checkpoint or policy checkpoint
            if "default_policy" in checkpoint_dir:
                # It's the policy checkpoint, look for policy_state.pkl in the same directory
                state_file = os.path.join(checkpoint_dir, "policy_state.pkl")
            else:
                # It's the main checkpoint, look for policy directory
                state_file = os.path.join(checkpoint_dir, "policies", "default_policy", "policy_state.pkl")
            
            if not os.path.exists(state_file):
                print(f"Warning: Expected policy state file not found: {state_file}")
                state_file = checkpoint_path
        except Exception as e:
            print(f"Error finding policy state from JSON: {e}")
            state_file = checkpoint_path
    else:
        # Use the provided path directly
        state_file = checkpoint_path
        
    print(f"Loading state from: {state_file}")
    
    # Create a blank model variable to ensure it's always defined
    model = None
    
    try:
        # Set up a blank model - try using gymnasium first, fall back to gym if needed
        try:
            from gymnasium.spaces import Box, Dict, Discrete
            print("Using gymnasium for spaces")
        except ImportError:
            try:
                from gym.spaces import Box, Dict, Discrete
                print("Using gym for spaces")
            except ImportError:
                print("Neither gymnasium nor gym found. Creating simplified spaces.")
                # Define minimal classes to avoid import errors
                class Box:
                    def __init__(self, low, high, shape): self.shape = shape
                class Dict(dict): pass
                class Discrete:
                    def __init__(self, n): self.n = n
        
        import numpy as np
        
        # Create dummy observation and action spaces
        obs_space = Dict({
            "entities": Box(low=-10, high=10, shape=(100, 26)),
            "mission": Box(low=-10, high=10, shape=(7,)),
            "visibility": Dict({
                "legacy": Box(low=0, high=1, shape=(100,)),
                "dynasty": Box(low=0, high=1, shape=(100,))
            }),
            "controllable_entities": Box(low=0, high=1, shape=(100,)),
            "valid_engage_mask": Box(low=0, high=1, shape=(100, 100)),
            "entity_id_list": Box(low=0, high=100, shape=(100,))
        })
        
        action_space = Dict({
            "action_type": Discrete(4),
            "params": Box(low=-1.0, high=1.0, shape=(10,))
        })
        
        try:
            model = CustomModel(obs_space, action_space, 14, {}, "flag_frenzy_model")
        except Exception as e:
            print(f"Error creating model: {e}")
        
        # Try to load state directly as PyTorch state
        try:
            # Check if this is a JSON file
            if state_file.endswith('.json'):
                print(f"Detected JSON checkpoint file, trying to parse it")
                try:
                    with open(state_file, 'r') as f:
                        checkpoint_info = json.load(f)
                        
                    # Look for the actual weights file
                    checkpoint_dir = os.path.dirname(state_file)
                    if 'checkpoint' in checkpoint_info:
                        weights_path = os.path.join(checkpoint_dir, checkpoint_info['checkpoint']['value'])
                        if os.path.exists(weights_path):
                            print(f"Found weights file: {weights_path}")
                            state_file = weights_path
                        else:
                            print(f"Referenced weights file not found: {weights_path}")
                except Exception as e:
                    print(f"Error parsing JSON checkpoint: {e}")
            
            # Now try to load the weights file
            try:
                # Use weights_only=False to handle the PyTorch 2.6 change
                state_dict = torch.load(state_file, weights_only=False)
            except Exception as e:
                print(f"Error loading weights with weights_only=False: {e}")
                # Try again with weights_only=True as last resort
                try:
                    state_dict = torch.load(state_file, weights_only=True)
                    print("Successfully loaded weights with weights_only=True")
                except Exception as e2:
                    print(f"Also failed with weights_only=True: {e2}")
                    raise
                    
            if "worker" in state_dict:
                # Handle RLLib nested structure
                for key in ["worker", "state", "policy_states", "default_policy"]:
                    if key in state_dict:
                        state_dict = state_dict[key]
            
            # The weights might be in a state_dict format or at the model field
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            # If we have a model at this point, load directly
            if isinstance(state_dict, dict) and "_features" not in state_dict:
                model.load_state_dict(state_dict)
            elif hasattr(state_dict, "model"):
                model.load_state_dict(state_dict.model.state_dict())
            else:
                print("Could not extract model weights directly. Using safe fallback method.")
                
                # Create a temporary PPO algorithm to load the checkpoint
                from ray.rllib.algorithms.ppo import PPO
                print("Creating temporary PPO algorithm to extract model...")
                algo = PPO(
                    config={
                        "framework": "torch",
                        "model": {
                            "custom_model": "flag_frenzy_model",
                            "custom_action_dist": "hybrid_action_dist",
                        },
                        "explore": False,
                        "_disable_preprocessor_api": True,
                    }
                )
                
                # Just load the policy and model
                try:
                    algo.restore(checkpoint_path)
                    policy = algo.get_policy()
                    model = policy.model
                except Exception as e:
                    print(f"Error loading with RLlib: {e}")
                    print("Warning: Model weights may not be loaded correctly")
                    
        except Exception as e:
            print(f"Error loading PyTorch state: {e}")
            print("Warning: Model weights may not be loaded correctly")
    
    except Exception as e:
        print(f"Error setting up model: {e}")
        print("Continuing with blank model")
        
    # Set the model to evaluation mode if we have a valid model
    if model is not None and hasattr(model, "eval"):
        model.eval()
        print("Model set to evaluation mode (frozen weights)")
    else:
        print("Warning: No valid model was created, will not be able to run inference")
    
    # Export the frozen model if a path is provided and we have a valid model
    if export_path and model is not None and hasattr(model, "state_dict"):
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            print(f"Exporting frozen model to: {export_path}")
            torch.save(model.state_dict(), export_path)
            
            # Also export model metadata
            metadata = {
                "source_checkpoint": checkpoint_path,
                "export_date": datetime.datetime.now().isoformat(),
                "frozen": True,
                "inference_only": True,
                "input_shape": {
                    "entities": [100, 26],
                    "mission": [7],
                    "visibility": {
                        "legacy": [100],
                        "dynasty": [100]
                    },
                    "controllable_entities": [100]
                },
                "output_shape": {
                    "action_type": 4,
                    "params": 10
                }
            }
            with open(f"{export_path}.meta.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error exporting model: {e}")
    
    return model


def process_replay_frame(replay_data, frame_index):
    """Convert replay state at specific frame to observation format."""
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
    controllable_entities = np.zeros(max_entities, dtype=np.float32)
    valid_engage_mask = np.zeros((max_entities, max_entities), dtype=np.float32)
    entity_id_list = []
    
    # Process entities and their states at this frame
    events_at_frame = [event for event in replay_data["MissionEvents"] if event["FrameIndex"] == frame_index]
    
    for event in events_at_frame:
        if event["EventName"] in ["SimMissionEvent_SpawnUnit", "SimMissionEvent_SpawnEntity"]:
            entity_idx = len(entity_id_list)
            if entity_idx < max_entities:
                entity_id = event["EntityId"]
                entity_id_list.append(entity_id)
                
                # Fill entity features
                entities[entity_idx] = create_entity_features(event)
                
                # Set visibility based on faction
                if event["Faction"] == 0:  # Legacy
                    visibility["legacy"][entity_idx] = 1.0
                else:  # Dynasty
                    visibility["dynasty"][entity_idx] = 1.0
                
                # Mark controllable entities (assuming IDs 11 and 35 are controllable)
                if entity_id in [11, 35]:
                    controllable_entities[entity_idx] = 1.0
    
    # Build full observation dictionary
    return {
        "entities": entities,
        "mission": mission,
        "visibility": visibility,
        "controllable_entities": controllable_entities,
        "valid_engage_mask": valid_engage_mask,
        "entity_id_list": np.array(entity_id_list, dtype=np.int32)
    }


def create_entity_features(event):
    """Create feature vector for an entity."""
    features = np.zeros(26, dtype=np.float32)
    
    # Basic features from event data
    features[0] = event["EntityId"] / 100.0  # Normalize ID
    features[1] = event["Faction"]
    
    # Position
    if "Location" in event:
        # Normalize coordinates based on map bounds (approximately -1300000 to 1300000)
        map_bounds = 1300000.0
        features[2] = event["Location"]["X"] / map_bounds
        features[3] = event["Location"]["Y"] / map_bounds
        features[4] = event["Location"]["Z"] / map_bounds
    
    # Velocity
    if "Vel" in event:
        # Normalize velocity (assuming max velocity around 1000 m/s)
        max_vel = 1000.0
        features[5] = event["Vel"]["X"] / max_vel if "X" in event["Vel"] else 0.0
        features[6] = event["Vel"]["Y"] / max_vel if "Y" in event["Vel"] else 0.0
        features[7] = event["Vel"]["Z"] / max_vel if "Z" in event["Vel"] else 0.0
    
    # Entity type features from DISEntityType
    if "DISEntityType" in event:
        dis_type = event["DISEntityType"]
        features[8] = dis_type.get("Kind", 0) / 10.0
        features[9] = dis_type.get("Domain", 0) / 10.0
        features[10] = dis_type.get("Country", 0) / 255.0
        features[11] = dis_type.get("Category", 0) / 10.0
        features[12] = dis_type.get("SubCategory", 0) / 10.0
        features[13] = dis_type.get("Specific", 0) / 10.0
        features[14] = dis_type.get("Extra", 0) / 100.0
    
    return features


def run_replay(replay_path, checkpoint_path, output_dir="replay_analysis", export_path=None):
    """Run a trained model on a replay file and capture decisions."""
    # Verify replay path exists
    if not os.path.exists(replay_path):
        print(f"Error: Replay file not found: {replay_path}")
        print("Available replay files:")
        replay_dir = os.path.dirname(replay_path) or "Replays"
        if os.path.exists(replay_dir):
            replay_files = [f for f in os.listdir(replay_dir) if f.endswith(".json")]
            for i, f in enumerate(replay_files[:10]):
                print(f"  {i+1}. {f}")
            if len(replay_files) > 10:
                print(f"  ... and {len(replay_files) - 10} more")
            print(f"\nPlease update REPLAY_PATH in the script to use one of these files.")
        return None
    
    # Load replay data
    try:
        with open(replay_path, 'r') as f:
            replay_data = json.load(f)
        print(f"Loaded replay: {os.path.basename(replay_path)}")
    except Exception as e:
        print(f"Error loading replay file: {e}")
        return None
    
    # Load model weights (and optionally export them)
    model = load_model_weights(checkpoint_path, export_path)
    if model is None:
        print("Failed to load model weights, cannot continue with inference")
        return None
    
    # Get max frame from replay
    try:
        if "MissionEvents" in replay_data and replay_data["MissionEvents"]:
            max_frame = max(event.get("FrameIndex", 0) for event in replay_data["MissionEvents"])
            print(f"Replay {os.path.basename(replay_path)} has {max_frame} frames")
        else:
            print(f"Warning: No mission events found in replay {os.path.basename(replay_path)}")
            max_frame = 0
    except Exception as e:
        print(f"Error determining max frame: {e}")
        print(f"Replay file keys: {list(replay_data.keys())}")
        print(f"First few entries: {replay_data.get('MissionEvents', [])[:3]}")
        max_frame = 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_name = os.path.splitext(os.path.basename(replay_path))[0]
    output_file = os.path.join(output_dir, f"{replay_name}_model_decisions_{timestamp}.json")
    
    # Process all frames and capture model decisions
    model_decisions = []
    
    print(f"Running model on replay frames...")
    # Process frames using the configured step size
    for frame in range(0, max_frame + 1, FRAME_STEP if 'FRAME_STEP' in globals() else 10):
        print(f"Processing frame {frame}/{max_frame}...", end="\r")
        
        # Build observation from replay frame
        obs = process_replay_frame(replay_data, frame)
        
        # Get model's action
        try:
            # If we have a full PPO algorithm object
            if hasattr(model, 'compute_single_action'):
                action = model.compute_single_action(
                    observation=obs,
                    explore=False  # Deterministic evaluation
                )
                # Map action to readable format
                action_type = int(action[0])
                params = action[1:].tolist()
            else:
                # If we have a direct PyTorch model
                with torch.no_grad():
                    # The error is occurring in processing tensor inputs
                    # Let's prepare the input properly
                    
                    # Convert numpy arrays to tensors
                    input_obs = {}
                    for key, value in obs.items():
                        if key == "visibility":
                            input_obs[key] = {
                                "legacy": torch.tensor(value["legacy"], dtype=torch.float32).unsqueeze(0),
                                "dynasty": torch.tensor(value["dynasty"], dtype=torch.float32).unsqueeze(0)
                            }
                        elif key == "entity_id_list":
                            # Handle this differently - keep as numpy for now
                            input_obs[key] = value
                        else:
                            # For other arrays, convert to tensor and add batch dimension
                            try:
                                input_obs[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
                            except Exception as e:
                                print(f"Error converting {key} to tensor: {e}")
                                # Use a dummy tensor as fallback
                                if key == "entities":
                                    input_obs[key] = torch.zeros((1, 100, 26), dtype=torch.float32)
                                elif key == "mission":
                                    input_obs[key] = torch.zeros((1, 7), dtype=torch.float32)
                                elif key == "controllable_entities":
                                    input_obs[key] = torch.zeros((1, 100), dtype=torch.float32)
                                elif key == "valid_engage_mask":
                                    input_obs[key] = torch.zeros((1, 100, 100), dtype=torch.float32)
                    
                    # Create input_dict
                    input_dict = {"obs": input_obs}
                    
                    try:
                        output, _ = model(input_dict, [], None)
                        
                        # Parse the output
                        logits = output[:4]  # First 4 outputs are action logits
                        params = output[4:].tolist()  # Remaining are the continuous params
                        
                        # Get the most likely action
                        action_type = torch.argmax(logits).item()
                        action = [action_type] + params
                    except Exception as e:
                        print(f"Error during model inference: {e}")
                        # Default to NoOp action as fallback
                        action_type = 0
                        action = [action_type] + [0.0] * 10
            
            action_names = {
                0: "NoOp",
                1: "Move",
                2: "ReturnToBase",
                3: "Engage"
            }
            action_name = action_names.get(action_type, f"Unknown_{action_type}")
            
            # Capture decision
            decision = {
                "frame": frame,
                "action_type": action_type,
                "action_name": action_name,
                "parameters": params,
                "entity_ids_present": obs["entity_id_list"].tolist()
            }
            
            # Add more context for certain action types
            if action_type == 3:  # Engage
                source_param = params[0]
                target_param = params[1]
                
                # Convert parameter values to entity indices
                entity_list = obs["entity_id_list"].tolist()
                if entity_list:
                    entity_count = len(entity_list)
                    src_idx = min(max(int(source_param * (entity_count - 1)), 0), entity_count - 1)
                    tgt_idx = min(max(int(target_param * (entity_count - 1)), 0), entity_count - 1)
                    
                    decision["source_entity_id"] = entity_list[src_idx]
                    decision["target_entity_id"] = entity_list[tgt_idx]
            
            model_decisions.append(decision)
            
        except Exception as e:
            print(f"Error processing frame {frame}: {e}")
    
    print("\nSaving model decisions...")
    
    # Save model decisions
    with open(output_file, 'w') as f:
        json.dump({
            "replay_path": replay_path,
            "model_checkpoint": checkpoint_path,
            "timestamp": timestamp,
            "decisions": model_decisions
        }, f, indent=2)
    
    print(f"Analysis saved to: {output_file}")
    return output_file


# ============================================================================
# CONFIGURATION PARAMETERS - Edit these values directly instead of using CLI args
# ============================================================================

# Path to the replay file you want to analyze (try different files if one doesn't have frames)
REPLAY_PATH = "Replays/22-04-2025 14-37-21.json"  # Try this one with more frames

# Path to the model checkpoint - this should point to the policy_state.pkl file which contains the actual model weights
CHECKPOINT_PATH = "ray_replay/flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_26b0e_00000_0_2025-04-19_09-13-57/checkpoint_000029/policies/default_policy/policy_state.pkl"

# Try these replay files that might have more frames
# REPLAY_PATH = "Replays/22-04-2025 14-37-21.json" 

# Output directory for analysis results
OUTPUT_DIR = "replay_analysis"

# Path to export the frozen model weights (None = don't export)
# This creates a separate inference-only model that isn't tied to RLlib
# Setting to None disables export, uncomment the path to enable export
EXPORT_PATH = "exported_models/flag_frenzy_frozen_model.pt"
# EXPORT_PATH = None  # Uncomment this line to disable export

# How many frames to skip (higher = faster processing, lower = more detailed analysis)
FRAME_STEP = 10

# ============================================================================

def find_latest_checkpoint():
    """Find the latest checkpoint in ray_results directory"""
    base_dir = "ray_results"
    if not os.path.exists(base_dir):
        print(f"Warning: {base_dir} directory not found")
        return None
        
    # Find all checkpoint directories
    checkpoints = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.startswith("checkpoint_"):
                checkpoint_path = os.path.join(root, d)
                # Get the checkpoint number
                try:
                    checkpoint_num = int(d.split("_")[1])
                    checkpoints.append((checkpoint_path, checkpoint_num))
                except (ValueError, IndexError):
                    pass
    
    if not checkpoints:
        print(f"No checkpoints found in {base_dir}")
        return None
        
    # Sort by checkpoint number (descending) and return the latest
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints[0][0]

if __name__ == "__main__":
    print(f"Starting model replay analysis...\n")
    
    # Check if the checkpoint path exists, try to find a valid one if not
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Warning: Checkpoint path not found: {CHECKPOINT_PATH}")
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            print(f"Found latest checkpoint: {latest_checkpoint}")
            CHECKPOINT_PATH = latest_checkpoint
        else:
            print("No valid checkpoint found. Please check the CHECKPOINT_PATH setting.")
    
    print(f"\nUsing replay: {REPLAY_PATH}")
    print(f"Using model checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Export path: {EXPORT_PATH or 'None (export disabled)'}")
    print(f"Frame step: {FRAME_STEP}\n")
    
    output_file = run_replay(REPLAY_PATH, CHECKPOINT_PATH, OUTPUT_DIR, EXPORT_PATH)
    
    if output_file:
        print(f"\nAnalysis complete! Results saved to: {output_file}")
    else:
        print("\nAnalysis failed. Please check the errors above.")
