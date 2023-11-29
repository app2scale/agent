from teastore import Teastore
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo, sac
import ray
import gym
import shutil
import os
from ray.tune.logger import pretty_print
from ray import tune, air
import tensorboard
import string
import random
import datetime


if __name__ == "__main__":
    
    num_iterations = 3000
    number_of_rollout_workers = 0
    evaluating_interval = 3000
    number_of_gpus = 0
    save_interval = 1

    ray.init(ignore_reinit_error=True)
    register_env("teastore", lambda config: Teastore())

    config_ppo = (ppo.PPOConfig()
        .rollouts(num_rollout_workers=number_of_rollout_workers, enable_connectors=False)
        .resources(num_gpus=number_of_gpus)
        .environment(env="teastore")
        .training(train_batch_size=32,sgd_minibatch_size=16,
                  model={"fcnet_hiddens": [64,64]},
                  lr=0.001)
        # .evaluation(evaluation_interval=evaluating_interval, evaluation_duration = evaluation_duration)
        # .callbacks(MetricCallbacks)
        )
    

    algo = config_ppo.build()
    logdir = algo.logdir

    for i in range(num_iterations):
        print("------------- Iteration", i+1, "-------------")
        result = algo.train()
        if ((i+1) % save_interval) == 0:
            path_to_checkpoint = algo.save(checkpoint_dir = logdir) 
            print("----- Checkpoint -----")
            print(f"An Algorithm checkpoint has been created inside directory: {path_to_checkpoint}.")
        print(pretty_print(result))

        print("Episode Reward Mean: ", result["episode_reward_mean"])
        print("Episode Reward Min: ", result["episode_reward_min"])
        print("Episode Reward Max: ", result["episode_reward_max"])
        
    algo.stop()
    ray.shutdown()