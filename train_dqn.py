#!/usr/bin/env python3
# Training script using DQN instead of PPO
import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.model import CustomModel
from models.hybrid_action_dist import HybridActionDistribution

# Register model
ModelCatalog.register_custom_model("flag_frenzy_model", CustomModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

register_env("FlagFrenzyEnv-v0", env_creator)
ray.init(ignore_reinit_error=True)


from ray.rllib.algorithms.callbacks import DefaultCallbacks

class FlagFrenzyCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for()
        if info:
            if info.get("engage_action_taken"):
                # Log entity/target ID as custom metrics
                entity_id = info.get("engage_entity_id", -1)
                target_id = info.get("engage_target_id", -1)

                # Aggregate counts
                episode.custom_metrics["engage_action_count"] = 1
                episode.custom_metrics["engage_entity_id"] = entity_id
                episode.custom_metrics["engage_target_id"] = target_id
            else:
                episode.custom_metrics["engage_action_count"] = 0

# Config for DQN
config = (
    DQNConfig()
    .environment(env="FlagFrenzyEnv-v0", env_config={})
    .framework("torch")
    .rollouts(
        num_rollout_workers=24,
        rollout_fragment_length=32,
    )
    .training(
        model={
            "custom_model": "flag_frenzy_model",
            "custom_action_dist": "hybrid_action_dist",
        },
        # DQN specific parameters
        gamma=0.995,
        lr=1e-4,
        train_batch_size=512,
        target_network_update_freq=10000,
        learning_starts=10000,
        buffer_size=500000,
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 1000000,  # Decay epsilon over 1M steps
        },
        # Prioritized replay buffer
        replay_buffer_config={
            "type": "PrioritizedReplayBuffer",
            "capacity": 500000,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            "prioritized_replay_beta_annealing_timesteps": 1000000,
        },
        grad_clip=10.0,
        # Dueling DQN
        dueling=True,
        # Double DQN
        double_q=True,
    )
    .resources(num_gpus=1)
    .callbacks(FlagFrenzyCallbacks)
    .to_dict()
)
config["horizon"] = 900

results = tune.Tuner(
    "DQN",
    param_space=config,
    run_config=air.RunConfig(
        stop={"training_iteration": 200},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=15,
            num_to_keep=5),
        name="dqn_flag_frenzy",
        log_to_file=True),
).fit()