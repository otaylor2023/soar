#!/usr/bin/env python3
# Training script using APPO instead of DQN
import ray
from ray import air, tune
from ray.rllib.algorithms.appo import APPOConfig
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

# Config for APPO
config = (
    APPOConfig()
    .environment(env="FlagFrenzyEnv-v0", env_config={})
    .framework("torch")
    .rollouts(
        num_rollout_workers=24,
        rollout_fragment_length=50,
        num_envs_per_worker=1,
    )
    .training(
        train_batch_size=4000,
        lr=3e-5,
        gamma=0.995,
        lambda_=0.95,
        clip_param=0.02,
        vf_loss_coeff=1.0,
        entropy_coeff=0.01,
        model={
            "custom_model": "flag_frenzy_model",
            "custom_model_config": {
                "enable_attribution": True,  # Enable gradient attribution
                "attribution_config": {
                    "num_samples": 50,  # Number of samples for integrated gradients
                    "internal_batch_size": 1  # Batch size for attribution computation
                }
            },
            "custom_action_dist": "hybrid_action_dist",
        },
    )
    .experimental(_enable_new_api_stack=False)  # Use the stable API
    .resources(
        num_gpus=1,
        num_learner_workers=1,  
        num_gpus_per_learner_worker=1,  
    )
    .debugging(log_level="INFO")
    .callbacks(FlagFrenzyCallbacks)
    .to_dict()
)

# Set horizon
config["horizon"] = 900

results = tune.Tuner(
    "APPO",
    param_space=config,
    run_config=air.RunConfig(
        stop={"training_iteration": 200},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=15,
            num_to_keep=5),
        name="appo_flag_frenzy_3",
        log_to_file=True),
).fit()