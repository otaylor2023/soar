import argparse
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution
import torch

# Register custom components
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
register_env("FlagFrenzyEnv-v0", env_creator)

def main():
    parser = argparse.ArgumentParser(description='Run inference with FlagFrenzy PPO model')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='Path to PPO checkpoint')
    parser.add_argument('--num-episodes', type=int, default=5, help='Number of episodes to run')
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    # Load PPO from checkpoint
    print(f"Loading PPO agent from checkpoint: {args.checkpoint_path}")
    algo = PPO.from_checkpoint(args.checkpoint_path)

    # Create environment manually
    env = env_creator({})

    for episode in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            # RLlib expects dict obs, convert if needed
            action = algo.compute_single_action(obs)
            policy = algo.get_policy()
            model = policy.model
            print(f"action: {action}")
            # print(f"feature importance: {algo.get_last_attribution()}")
            attribution_dict = model.get_last_attribution()
            # print(f"attribution_dict: {list(attribution_dict.keys())}")
            # print(f"action['action_type']: {action['action_type']}")
            selected_attribution_list = attribution_dict[action["action_type"]]
            # print(f"selected_attribution_list: {selected_attribution_list}")
            print(f"printing {len(selected_attribution_list)} attributions for actions taken")
            for i in range(len(selected_attribution_list)):
                print(model.format_explanation(selected_attribution_list[i]))

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if done or truncated:
                break

        print(f"Episode {episode+1} finished in {step} steps with total reward: {total_reward}")

    env.close()
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
