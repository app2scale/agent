import random

import ray
from ray import tune, train, air
from ray.tune.registry import register_env
from env_v3 import Teastore
from ray.rllib.algorithms.ppo import PPOConfig
import pprint
import pandas as pd


ray.init(ignore_reinit_error=True)
register_env("teastore", lambda config: Teastore())


config = (PPOConfig()
          .training(lr=tune.grid_search([0.001, 0.0001]),
                    train_batch_size = tune.grid_search([512, 1024]),
                    sgd_minibatch_size = tune.grid_search([64, 128]),
                    clip_param=tune.grid_search([0.1, 0.2, 0.3]),
                    lambda_=tune.grid_search([0.9, 0.95, 1]), # 0.9, 0.95, 1
                    entropy_coeff=tune.grid_search([0, 0.005, 0.01])) # 0, 0.005, 0.01
          .environment(env="teastore"))


# Our ray version is old so we should upgrade it. In the new version, run_conig will be train.RunConfig
tuner = tune.Tuner("PPO",
                   param_space=config,
                   run_config=air.RunConfig(stop={"training_iteration":250}))

results = tuner.fit()

best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
best_checkpoint = best_result.checkpoint
try:
    with open("best_result.out", "w") as output_file:
        output_file.write(f"En iyi sonuç: {best_result}\n")
        output_file.write(f"En iyi sonuçun metriği: {best_result.metric}\n")
        output_file.write(f"En iyi sonuçun modu: {best_result.mode}\n")
        output_file.write(f"En iyi sonuçun checkpoint'i: {best_checkpoint}\n")

    df_results = results.get_dataframe()
    df_results.to_csv('results.csv', index=False)
except Exception as e:
    print(e)
