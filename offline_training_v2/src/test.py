from env import Teastore
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray
import numpy as np
import time
import pandas as pd



ray.init(ignore_reinit_error=True)
register_env("teastore", lambda config: Teastore())
policy_name = "/Users/hasan.nayir/ray_results/PPO_teastore_2024-03-01_14-47-44bip7rec8/checkpoint_000500"
algo = Algorithm.from_checkpoint(policy_name)
env = Teastore()


obs, info = env.reset()
obs = np.array([2,6,6,5,5,5,5, 24])
print("Initial state", obs)
done = False
truncated = False
sum_reward = 0
columns = ["replica", "cpu", "heap", "expected_tps", "reward"]
load = np.linspace(24, 160, 18)


output = pd.DataFrame(columns=columns)
if __name__ == "__main__":
    step = 0

    for tps in load:
        # obs = np.concatenate([obs, [tps]])
        obs[3] = tps
        action = algo.compute_single_action(obs)
        print(f"Prev state: {obs} ---- Action: {env.possible_state[action]}")

        # next_obs, reward, done, truncated, info= env.step(action)
        # sum_reward += reward
        # temp = [next_obs[0], next_obs[1], next_obs[2], next_obs[3], reward]


        # output.loc[step,:] = temp
        # print(output)
        # # output.to_csv("../results/output_pv.csv", index=False)

        # obs = next_obs
        # step += 1