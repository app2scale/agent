from teastore import Teastore
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray
import numpy as np
import time
import pandas as pd


ray.init(ignore_reinit_error=True)
register_env("teastore", lambda config: Teastore())
checkpoint_dir = "/Users/hasan.nayir/ray_results/PPO_teastore_2023-11-28_16-26-46gq61hp37/"
policy_name = "checkpoint_003000"
path_to_checkpoint = checkpoint_dir + policy_name
algo = Algorithm.from_checkpoint(path_to_checkpoint)
columns = ["replica", "cpu", "heap", "previous_tps", "instant_tps", "reward", "sum_reward"]
output = pd.DataFrame(columns=columns)
env = Teastore()
obs, info = env.reset()
print("Initial State: ", obs)


done = False
truncated = False
sum_reward = 0

step = 1
while not done:
    action = algo.compute_single_action(obs)
    next_obs, reward, done, truncated, info = env.step(action)
    sum_reward += reward
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    temp = [next_obs["replica"], next_obs["cpu"], next_obs["heap"], next_obs["previous_tps"][0],
            next_obs["instant_tps"][0], reward, sum_reward]
    output.loc[step,:] = temp
    print(output)
    output.to_csv("../results/test_results.csv", index=False)
    obs = next_obs
    step += 1
    