from teastore import Teastore
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
import ray
import gym
import shutil
import os
from ray.tune.logger import pretty_print
from ray import tune, air
import tensorboard
from metric_callbacks import MetricCallbacks
import string
import random
import datetime


def generate_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))

def generate_checkpoint_folder_name():
    letters_and_digits = string.ascii_letters + string.digits
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    random_string = ''.join(random.choice(letters_and_digits) for _ in range(8))
    foldername = f"policy_checkpoints/{current_datetime}_{random_string}"
    return foldername



if __name__ == "__main__":
    
    num_iterations = 2500
    number_of_rollout_workers = 0
    evaluating_interval = 2
    number_of_gpus = 0
    save_interval = 2500

    checkpoint_dir = generate_checkpoint_folder_name()

    ray.init(ignore_reinit_error=True)
    register_env("teastore", lambda config: Teastore())


    config = (ppo.PPOConfig()
              .rollouts(num_rollout_workers=number_of_rollout_workers)
              .resources(num_gpus=number_of_gpus)
              .environment(env="teastore")
              .evaluation(evaluation_interval=evaluating_interval)
              .callbacks(MetricCallbacks)
            )
    algo = config.build()

    for i in range(num_iterations):
        print("------------- Iteration", i+1, "-------------")
        result = algo.train()
        if ((i+1) % save_interval) == 0:
            path_to_checkpoint = algo.save(checkpoint_dir)
            print("----- Checkpoint -----")
            print(f"An Algorithm checkpoint has been created inside directory: {path_to_checkpoint}.")
        print(pretty_print(result))
        if "evaluation" in result.keys():
            print(result["evaluation"]["custom_metrics"])
    
    algo.stop()
    ray.shutdown()


    
   


  



