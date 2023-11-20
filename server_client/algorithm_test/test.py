from teastore_v5_env import Teastore
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray
import numpy as np
import time
import pandas as pd


def action_to_string(action):
    action_dict = {"DO_NOTHING": 0, "INCREASE_REPLICA": 1, "DECREASE_REPLICA": 2, "INCREASE_CPU": 3, "DECREASE_CPU": 4, "INCREASE_HEAP": 5, "DECREASE_HEAP": 6}
    for key, value in action_dict.items():
        if value == action:
            return key



ray.init(ignore_reinit_error=True)
register_env("teastore", lambda config: Teastore())
checkpoint_dir = "/Users/hasan.nayir/ray_results/PPO_teastore_2023-11-14_17-42-02i1w2vuha/"
policy_name = "checkpoint_010000"
path_to_checkpoint = checkpoint_dir + policy_name
algo = Algorithm.from_checkpoint(path_to_checkpoint)
columns = ["action", "action_n","replica", "cpu", "heap", "previous_tps", "instant_Tps", "reward", "sum_reward"]
output = pd.DataFrame(columns=columns)
env = Teastore()
obs, info = env.reset()


done = False
sum_reward = 0

step = 1
while not done:
    time.sleep(2)
    action = algo.compute_single_action(obs)
    # action = 3
    next_obs, reward, done, truncated, info = env.step(action)
    sum_reward += reward
    # reversed_next_obs = reverse_state_status(next_obs)
    str_action = action_to_string(action)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    temp = [str_action, action, next_obs["replica"], next_obs["cpu"], next_obs["heap"], next_obs["previous_tps"],
            next_obs["instant_tps"], reward, sum_reward]
    output.loc[step,:] = temp
    print(output)
    output.to_csv("test_results.csv", index=False)
    obs = next_obs
    step += 1
    