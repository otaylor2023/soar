import json
import numpy as np
from register_env import env_creator
from pathlib import Path
import logging
import os
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from models.model_oter import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
from utils import format_explanation


class ReplayEnvironment:
    def __init__(self, replay_file_path: str, checkpoint_path: str = None):
        """Initialize replay environment.
        
        Args:
            replay_file_path: Path to the replay JSON file
            checkpoint_path: Optional path to a PPO checkpoint for running agent simulation
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize Ray if checkpoint is provided
        self.enable_simulation = False
        if checkpoint_path:
            # Disable terminal color support to avoid isatty errors
            os.environ["COLORFUL_DISABLE"] = "1"
            # Initialize Ray with minimal logging
            if not ray.is_initialized():
                ray.init(logging_level=logging.ERROR)
            
            # Register custom components
            ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
            ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
            register_env("FlagFrenzyEnv-v0", env_creator)
            
            # Load algorithm from checkpoint
            self.algo = PPO.from_checkpoint(checkpoint_path)
            self.enable_simulation = True
            self.algo.get_policy().model.set_attribution_enabled(self.enable_simulation)

        # Load replay data
        self.replay_file_path = Path(replay_file_path)
        with open(self.replay_file_path, "r") as f:
            self.replay_data = json.load(f)

        # Extract events
        self.mission_events = self.replay_data.get("MissionEvents", [])
        self.player_events = self.replay_data.get("PlayerEvents", [])
        self.current_event_idx = 0

        self.random_seed = self.replay_data.get("RandomSeed", None)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            import torch
            torch.manual_seed(self.random_seed)
            # if self.enable_simulation:
            #     self.algo.workers.foreach_worker(
            #         lambda w: w.set_seed(self.random_seed)
            #     )

        def seed_worker(worker):
            env = worker.env
            if hasattr(env, "seed"):
                env.seed(self.random_seed)
            np.random.seed(self.random_seed)
            try:
                import torch
                torch.manual_seed(self.random_seed)
            except ImportError:
                pass

        self.algo.workers.foreach_worker(seed_worker)
        # Create environment
        self.env = env_creator({"save_replay": False, "seed": self.random_seed})
        
        self.logger.info(f"Loaded replay file: {replay_file_path}")
        self.logger.info(f"Found {len(self.mission_events)} mission events and {len(self.player_events)} player events")
        
        if self.enable_simulation:
            self.logger.info(f"Agent simulation enabled with checkpoint: {checkpoint_path}")


    def reset(self):
        """Reset environment and apply initial mission events."""
        self.current_event_idx = 0
        
        # Reset env
        obs, info = self.env.reset()
        
        # Apply mission events
        self._apply_mission_events()
        
        return obs, info

    def _apply_mission_events(self):
        """Apply all mission events to set up initial state."""
        for event in self.mission_events:
            event_type = event.get("EventType")
            if event_type == "EntitySpawned":
                self.env._entity_spawned(event)
            elif event_type == "TargetGroupSpawned":
                self.env._target_group_spawned(event)
            elif event_type == "VictoryConditionsSet":
                self.env._set_victory_conditions(event)
            elif event_type == "FlagSpawned":
                self.env._flag_spawned(event)

    def step(self):
        """Step through one player event in the replay."""
        if self.current_event_idx >= len(self.player_events):
            return None, None, True, True, {"message": "Replay complete"}

        event = self.player_events[self.current_event_idx]
        self.current_event_idx += 1

        # Convert event to action
        player_action = self._event_to_action(event)
        
        # Get observation before applying action
        current_obs = self.env._get_observations()
        
        if self.enable_simulation:
            agent_action = self.algo.compute_single_action(current_obs, explore=False)
            policy = self.algo.get_policy()
            model = policy.model
            attribution_dict = model.get_last_attribution()
            print(f"player_action: {player_action}")
            print(f"player attribution from agent: {format_explanation(attribution_dict[player_action['action_type']][0], self.env)}")
            print(f"agent_action: {agent_action}")
            print(f"agent attribution from agent: {format_explanation(attribution_dict[agent_action['action_type']][0], self.env)}")
        
        # Execute action in environment
        obs, reward, done, truncated, info = self.env.step(player_action)
        
        return obs, reward, done, truncated, info

    def _event_to_action(self, event):
        """Convert a player event to an environment action."""
        event_type = event.get("EventType")
        action = {"action_type": 0, "params": np.zeros(self.env.param_vector_size)}
        # TODO: fix all of these assumptions i didnt make them
        if event_type == "Move":
            action["action_type"] = 1
            # Extract spline points and convert to normalized params
            spline_points = event.get("SplinePoints", [])
            if spline_points:
                params = []
                for point in spline_points[:5]:  # Take up to 5 points
                    x = (point["X"] - self.env.area_min[0]) / (self.env.area_max[0] - self.env.area_min[0])
                    y = (point["Y"] - self.env.area_min[1]) / (self.env.area_max[1] - self.env.area_min[1])
                    params.extend([x, y])
                action["params"][:len(params)] = params

        elif event_type == "RTB":
            action["action_type"] = 2

        elif event_type == "Engage":
            action["action_type"] = 3
            # Extract target info and engagement parameters
            target_id = event.get("TargetId", 0)
            engagement_level = event.get("EngagementLevel", 0)
            weapons_mode = event.get("WeaponsMode", 0)
            
            # Normalize parameters
            action["params"][0] = target_id
            action["params"][1] = engagement_level
            action["params"][2] = weapons_mode

        return action

    def close(self):
        """Close the environment."""
        self.env.close()

def main():
    """Example usage of replay environment."""
    import argparse
    
    # Disable terminal color support to avoid isatty errors
    os.environ["COLORFUL_DISABLE"] = "1"
    
    parser = argparse.ArgumentParser(description='Run replay of a FlagFrenzy simulation')
    parser.add_argument('--replay-path', type=str, required=True, help='Path to replay file')
    parser.add_argument('--checkpoint-path', type=str, required=False, help='Path to PPO checkpoint')
    args = parser.parse_args()

    # Create and initialize replay environment
    replay_env = ReplayEnvironment(args.replay_path, args.checkpoint_path)
    
    try:
        # Reset environment
        obs, info = replay_env.reset()
        
        # Step through replay
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            obs, reward, done, truncated, info = replay_env.step()
            if obs is None:  # Replay complete
                break
                
            total_reward += reward if reward else 0
            step += 1
            
            if done or truncated:
                break
        
        print(f"Replay complete after {step} steps with total reward: {total_reward}")
    
    finally:
        replay_env.close()
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()
