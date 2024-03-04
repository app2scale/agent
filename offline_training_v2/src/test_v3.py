from env_v3 import Teastore
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray
import numpy as np
import time
import pandas as pd
from itertools import product

# ray.init(ignore_reinit_error=True)
register_env("teastore", lambda config: Teastore())
policy_name = "/Users/hasan.nayir/ray_results/PPO_teastore_2024-03-04_11-24-3056dvrf_m/checkpoint_001250"
algo = Algorithm.from_checkpoint(policy_name)
env = Teastore()
replica = [1, 2, 3]
cpu = [4, 5, 6, 7, 8, 9]
possible_state = np.array(list(product(replica, cpu)))
action_map = {0:"do nothing",
              1: "increase replica",
              2: "decrease replica",
              3: "increase cpu",
              4:"decrease cpu"}


obs, info = env.reset()
obs = np.array([3,9,24])
# obs = np.array([3,4,128])

done = False
truncated = False
sum_reward = 0
columns = ["replica", "cpu", "expected_tps", "reward"]
load = np.linspace(24, 160, 18)

def find_next_state(state, action):
    if action == 0:
        next_state = state[:2] + np.array([0, 0])
    elif action == 1: # increase_replica
        next_state = state[:2] + np.array([1,0])
    elif action == 2: # decrease_replica
        next_state = state[:2] + np.array([-1,0])
    elif action == 3: # increase_cpu
        next_state = state[:2] + np.array([0, 1])
    else: # decrease_cpu
        next_state = state[:2] + np.array([0,-1])
    return next_state

output = pd.DataFrame(columns=columns)
if __name__ == "__main__":
    step = 0

    for _ in range(0,40):
        action = algo.compute_single_action(obs)
        next_state = find_next_state(obs, action)
        next_state = np.concatenate([next_state, [24]])
        
        print(f"Prev state: {obs} ---- Action: {action_map[action]} ---- Next state: {next_state}")
        obs = next_state

        # next_obs, reward, done, truncated, info= env.step(action)
        # sum_reward += reward
        # temp = [next_obs[0], next_obs[1], next_obs[2], next_obs[3], reward]


        # output.loc[step,:] = temp
        # print(output)
        # # output.to_csv("../results/output_pv.csv", index=False)

        # obs = next_obs
        # step += 1