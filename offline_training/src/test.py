from teastore import Teastore
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray
import numpy as np
import time
import pandas as pd

def reverse_state_status(observation):
    min_values = np.array([1, 100, 100])
    max_values = np.array([9, 900, 900])
    reversed_state = np.zeros((5))
    reversed_state[3:] = observation[3:]
    for i in range(len(min_values)):
        reversed_state[i] = (max_values[i]-min_values[i])*observation[i]/8.0 + min_values[i]

    return reversed_state.astype(np.float16)

def action_to_string(action):
    action_dict = {"DO_NOTHING": 0, "INCREASE_REPLICA": 1, "DECREASE_REPLICA": 2, "INCREASE_CPU": 3, "DECREASE_CPU": 4, "INCREASE_HEAP": 5, "DECREASE_HEAP": 6}
    for key, value in action_dict.items():
        if value == action:
            return key



ray.init(ignore_reinit_error=True)
register_env("teastore", lambda config: Teastore())
checkpoint_dir = "offline_training/policy_checkpoints/2023-08-18_12-18_LIGYeSIU/"
policy_name = "checkpoint_000100"
path_to_checkpoint = checkpoint_dir + policy_name
algo = Algorithm.from_checkpoint(path_to_checkpoint)
columns = ["action", "action_n","replica", "cpu", "heap", "inc_tps", "out_tps", "cpu_usage", "memory_usage", "cost", "reward", "sum_reward"]
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
    temp = [str_action, action,next_obs["0replica"], next_obs["1cpu"], next_obs["2heap"], info["inc_tps"],
            info["out_tps"], next_obs["3used_cpu"][0], next_obs["4used_ram"][0], info["cost"], reward, sum_reward]
    output.loc[step,:] = temp
    print(output)
    output.to_csv("offline_training/src/output.csv", index=False)
    obs = next_obs
    step += 1
    