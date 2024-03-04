from env import Teastore
import gym, ray
from ray.tune.registry import register_env
from gym.envs.registration import register
from ray.rllib.algorithms import ppo
import numpy as np
import pandas as pd



if __name__ == "__main__":
    env = Teastore()
    columns = ["replica", "cpu", "expected_tps", "reward"]
    output = pd.DataFrame(columns=columns)
    output_array = []
    env.reset()
    sum_reward = 0
    for _ in range(50):
        
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        temp = [state[0], state[1], state[2], reward]
        output_array.append(temp)

        output = pd.DataFrame(output_array, columns = columns)
        print(output)
