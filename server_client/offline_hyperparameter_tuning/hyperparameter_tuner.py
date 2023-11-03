import random

import ray
from ray import tune, train, air
from ray.tune.registry import register_env
from teastore_env import Teastore
from ray.rllib.algorithms.ppo import PPOConfig


ray.init(ignore_reinit_error=True)
register_env("teastore", lambda config: Teastore())


config = (PPOConfig()
          .training(lr=tune.grid_search([0.01, 0.001, 0.0001]),
                    train_batch_size = tune.grid_search([32, 64, 128]),
                    sgd_minibatch_size = tune.grid_search([8, 16, 32]),
                    clip_param=tune.grid_search([0.1, 0.2, 0.3]),
                    lambda_=tune.uniform(0.9,1.0),
                    entropy_coeff=tune.uniform(0, 0.01))
          .environment(env="teastore"))


# Our ray version is old so we should upgrade it. In the new version, run_conig will be train.RunConfig
tuner = tune.Tuner("PPO",
                   param_space=config,
                   run_config=air.RunConfig(stop={"training_iteration":100}))

results = tuner.fit()

best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
best_checkpoint = best_result.checkpoint

