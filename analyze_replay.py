import json
import numpy as np
import torch
from pathlib import Path
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import os

from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO
from register_env import env_creator

# Register custom model and action distribution
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

class ReplayAnalyzer:
    def __init__(self, replay_path: str, checkpoint_path: str):
        self.replay_path = replay_path
        self.checkpoint_path = checkpoint_path
        
        # Load replay data
        with open(replay_path, 'r') as f:
            self.replay_data = json.load(f)
            
        # Group events by frame
        self.events_by_frame = defaultdict(list)
        self.frame_indices = set()
        
        for event in self.replay_data.get('MissionEvents', []):
            frame_idx = event.get('FrameIndex', 0)
            self.events_by_frame[frame_idx].append(event)
            self.frame_indices.add(frame_idx)
            
        self.frame_indices = sorted(list(self.frame_indices))
        
        # Track entity state
        self.entity_state: Dict[int, Dict] = {}  # EntityId -> current state
        self.entity_history: Dict[int, List[Dict]] = defaultdict(list)  # EntityId -> list of states
        
        # Create environment for observation structure
        self.env = env_creator({})
        
        # Initialize algorithm with the checkpoint
        self.algo = PPO(
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
            self.algo.restore(checkpoint_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def process_frame_events(self, frame_idx: int) -> Tuple[Dict[str, Any], List[str]]:
        """Process all events in a frame and return the observation and event descriptions."""
        events = self.events_by_frame[frame_idx]
        descriptions = []
        
        # Update entity states based on events
        for event in events:
            event_name = event.get('EventName', '')
            entity_id = event.get('EntityId')
            
            if 'SpawnUnit' in event_name:
                # Track new entity
                self.entity_state[entity_id] = {
                    'id': entity_id,
                    'faction': event.get('Faction'),
                    'type': event.get('Identifier'),
                    'position': event.get('Pos'),
                    'velocity': event.get('Vel'),
                    'controllable': event.get('Controllable', False),
                    'weapons': event.get('Loadout', {}).get('Weapons', {}),
                    'specifications': event.get('Specifications', {})
                }
                descriptions.append(f"Spawned {event['Identifier']} (ID: {entity_id}) for faction {event['Faction']}")
                
            elif 'Move' in event_name:
                if entity_id in self.entity_state:
                    self.entity_state[entity_id]['position'] = event.get('Position')
                    self.entity_state[entity_id]['velocity'] = event.get('Velocity')
                    descriptions.append(f"Entity {entity_id} moved to {event.get('Position')}")
                    
            elif 'Engage' in event_name or 'Combat' in event_name:
                descriptions.append(f"Combat event: {event_name} involving entity {entity_id}")
                
            elif 'Destroy' in event_name:
                if entity_id in self.entity_state:
                    del self.entity_state[entity_id]
                descriptions.append(f"Entity {entity_id} was destroyed")
                
            # Archive state for history
            if entity_id in self.entity_state:
                self.entity_history[entity_id].append(self.entity_state[entity_id].copy())
        
        # Create observation for model using environment's format
        observation = self.create_observation(frame_idx)
        
        return observation, descriptions

    def create_observation(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Create observation dictionary for the model from current state."""
        # Get sample observation structure from environment
        sample_obs, _ = self.env.reset()
        obs = {}
        
        # Initialize with same structure as environment
        for key in sample_obs.keys():
            if isinstance(sample_obs[key], dict):
                obs[key] = {}
                for subkey in sample_obs[key]:
                    obs[key][subkey] = np.zeros_like(sample_obs[key][subkey])
            else:
                obs[key] = np.zeros_like(sample_obs[key])
        
        # Fill entity features
        for idx, (entity_id, state) in enumerate(self.entity_state.items()):
            if idx >= 100:  # Max entities limit
                break
                
            # Convert entity state to feature vector
            entity_features = self.entity_to_features(state)
            obs["entities"][idx] = entity_features
            
            # Set controllable flag
            obs["controllable_entities"][idx] = 1 if state['controllable'] else 0
            
            # Set visibility based on faction
            if state['faction'] == 0:  # Legacy
                obs["visibility"]["legacy"][idx] = 1
            else:  # Dynasty
                obs["visibility"]["dynasty"][idx] = 1
                
            # Set valid engagement mask (simplified - can engage other faction)
            for other_idx, (other_id, other_state) in enumerate(self.entity_state.items()):
                if state['faction'] != other_state['faction']:
                    obs["valid_engage_mask"][idx * 100 + other_idx] = 1
                    
            obs["entity_id_list"][idx] = entity_id
        
        return obs

    def entity_to_features(self, state: Dict) -> np.ndarray:
        """Convert entity state to feature vector."""
        features = np.zeros(26)
        
        # Basic features
        features[0] = state['id']
        features[1] = state['faction']
        features[2] = 1 if state['controllable'] else 0
        
        # Position
        if state['position']:
            features[3] = state['position']['X']
            features[4] = state['position']['Y']
            features[5] = state['position']['Z']
            
        # Velocity
        if state['velocity']:
            features[6] = state['velocity']['X']
            features[7] = state['velocity']['Y']
            features[8] = state['velocity']['Z']
            
        # Specifications
        specs = state['specifications']
        if specs:
            features[9] = specs.get('CruisingSpeed', 0)
            features[10] = specs.get('MaxSpeed', 0)
            features[11] = specs.get('RadarCrossSection', 0)
            
        # Weapons (count and basic stats)
        weapons = state['weapons']
        if weapons:
            features[12] = len(weapons)  # Number of weapon systems
            
            # Aggregate weapon stats
            total_range = 0
            total_damage = 0
            total_ammo = 0
            for weapon in weapons.values():
                if 'Projectile' in weapon:
                    total_range += weapon['Projectile'].get('Range', 0)
                    total_damage += weapon['Projectile'].get('Damage', 0)
                total_ammo += weapon.get('Ammo', 0)
                
            features[13] = total_range
            features[14] = total_damage
            features[15] = total_ammo
            
        return features

    def analyze_model_decision(self, observation: Dict[str, np.ndarray]) -> Tuple[Dict, str]:
        """Get model's decision for the current observation and return explanation."""
        # Get model's action using algo.compute_single_action
        actions_by_entity = {}
        
        # Get actions for each controllable entity
        for idx, is_controllable in enumerate(observation["controllable_entities"]):
            if is_controllable:
                entity_id = int(observation["entity_id_list"][idx])
                # Create entity-specific observation by masking others
                entity_obs = {k: v.copy() if not isinstance(v, dict) else 
                            {sk: sv.copy() for sk, sv in v.items()}
                            for k, v in observation.items()}
                entity_obs["current_entity_index"] = idx
                
                # Get action for this entity
                action = self.algo.compute_single_action(entity_obs, explore=False)
                
                # Store action
                if isinstance(action, dict):
                    action_type = action["action_type"]
                    params = action["params"]
                else:
                    action_type = action[0]
                    params = action[1:]
                    
                actions_by_entity[entity_id] = {
                    "action_type": action_type,
                    "params": params
                }
        
        # Create explanation
        action_types = ["Move", "Return to Base", "Combat Air Patrol", "Engage Target"]
        explanation = "Model decisions by entity:\n"
        
        for entity_id, action in actions_by_entity.items():
            action_type = action["action_type"]
            params = action["params"]
            
            explanation += f"Entity {entity_id}:\n"
            explanation += f"  - Action: {action_types[action_type]}\n"
            
            if action_type == 0:  # Move
                explanation += f"  - Movement parameters: {params[:6]}\n"  # First 6 params are spline points
            elif action_type == 3:  # Engage
                target_idx = int(params[0] * 100)  # Scale param to entity index
                explanation += f"  - Target entity index: {target_idx}\n"
            
        return actions_by_entity, explanation

    def analyze_replay(self):
        """Analyze the entire replay file frame by frame."""
        print(f"Analyzing replay: {self.replay_path}")
        print(f"Total frames: {len(self.frame_indices)}")
        
        for frame_idx in self.frame_indices:
            print(f"\nFrame {frame_idx}:")
            
            # Process frame events
            observation, event_descriptions = self.process_frame_events(frame_idx)
            
            # Print events
            if event_descriptions:
                print("Events:")
                for desc in event_descriptions:
                    print(f"  - {desc}")
            
            # Get model's decision
            actions_by_entity, explanation = self.analyze_model_decision(observation)
            
            print("Model decisions:")
            print(explanation)
            
            # Print current state summary
            print(f"Active entities: {len(self.entity_state)}")
            
def main():
    parser = argparse.ArgumentParser(description='Analyze a replay file with a trained model')
    parser.add_argument('replay_path', type=str, help='Path to the replay file')
    parser.add_argument('model_path', type=str, help='Path to the trained model weights')
    
    args = parser.parse_args()
    
    analyzer = ReplayAnalyzer(args.replay_path, args.model_path)
    analyzer.analyze_replay()

if __name__ == "__main__":
    main() 