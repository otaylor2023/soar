# Training script
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
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

class FlagFrenzyCallbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        it = result["training_iteration"]
        if it <= 200:  # linear decay 0.01 -> 0.003
            new_coeff = 0.01 - 0.007 * (it / 200)
            algorithm.config["entropy_coeff"] = new_coeff

# Config for PPO
config = (
    PPOConfig()
    .environment(env="FlagFrenzyEnv-v0", env_config={})
    .framework("torch")
    .training(
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
        clip_param=0.02,
        gamma=0.995,
        lambda_=0.95,
        kl_coeff=0.2,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=30,
        lr=3e-5,
        entropy_coeff=0.01,
    )
    .rollouts(
        num_rollout_workers=24,
        rollout_fragment_length=50,
        num_envs_per_worker=1
    )
    .resources(num_gpus=1)
    .debugging(log_level="INFO")
    .callbacks(FlagFrenzyCallbacks)
    .to_dict()
)
config["horizon"] = 900

results = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=air.RunConfig(
        stop={"training_iteration": 150},     # ≈ 19 M steps
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=15,
            num_to_keep=5),
        name="rewardshape_flag_frenzy_ppo",
        log_to_file=True),
).fit()
