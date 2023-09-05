from app2scale_env_v2 import Teastore
import gym, ray
from ray.tune.registry import register_env
from gym.envs.registration import register
from ray.rllib.algorithms import ppo
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DO_NOTHING = 0
INCREASE_REPLICA = 1
DECREASE_REPLICA = 2
INCREASE_CPU = 3
DECREASE_CPU = 4
INCREASE_HEAP = 5
DECREASE_HEAP = 6

columns = ["action", "replica", "cpu", "heap", "inc_tps", "out_tps", 
           "cpu_usage", "memory_usage", "cost", "reward", "sum_reward", 
           #"response_time", "number_of_request", "number_of_failures"
           ]
output = pd.DataFrame(columns=columns)

def action_to_string(action):
    mapping = {0: "DO_NOTHING", 1: "INCREASE_REPLICA", 
               2: "DECREASE_REPLICA", 3: "INCREASE_CPU", 
               4: "DECREASE_CPU", 5: "INCREASE_HEAP", 6: "DECREASE_HEAP"}
    return mapping[action]

def run_one_episode(env):
    env.reset()
    sum_reward = 0

    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        str_action = action_to_string(action)
        temp = [str_action, state["0replica"], state["1cpu"], state["2heap"], 
                info["inc_tps"], info["out_tps"], info["cpu_usage"], 
                info["memory_usage"], info["cost"], reward, sum_reward, 
                #info["response_time"], info["number_of_request"], info["number_of_failures"]
                ]

        output.loc[i,:] = temp
        print(output)

        if truncated or terminated:
            print("done @ step {}".format(i+1))
            break
    
    print("sum reward: ", sum_reward)
    return sum_reward
 

if __name__ == "__main__":

    metric_dict = {"container_network_receive_bytes_total": "inc_tps",
                   "container_network_transmit_packets_total": "out_tps",
                   "container_cpu_usage_seconds_total": "cpu_usage",
                   "container_memory_working_set_bytes": "memory_usage"}
    
    env = Teastore(metric_name_dict=metric_dict)    
    history = []
    for _ in range(200):
        sum_reward = run_one_episode(env)
        history.append(sum_reward)
    
    avg_sum_reward = np.nanmean(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))