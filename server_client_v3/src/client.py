from locust.env import Environment
from ray.rllib.env.policy_client import PolicyClient
import pandas as pd
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
import time
import numpy as np
from collections import OrderedDict
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from locust import HttpUser, task, constant, constant_throughput, events
import ssl
import random
ssl._create_default_https_context = ssl._create_unverified_context


MAX_STEPS = 50
EXPECTED_TPS = 50
USERS = 1
ALPHA = 0.6
DATA_PATH = "/Users/hasan.nayir/Projects/Payten/APP2SCALE_RL/data/output_v5_offline_data.csv"
data = pd.read_csv(DATA_PATH)

def matching_indexes(data, target):
    equal_rows = np.all(data.iloc[:, 0:3].values == target, axis=1)
    matched_indexes = np.where(equal_rows)[0]
    return matched_indexes.tolist()


ACTION_SPACE = Tuple([Discrete(6), Discrete(6),Discrete(6)])
POLICY_CLIENT = PolicyClient("http://localhost:9900", inference_mode="local") 

def matching_indexes(data, target):
    equal_rows = np.all(data.iloc[:, 0:3].values == target, axis=1)
    matched_indexes = np.where(equal_rows)[0]
    return matched_indexes.tolist()

def step(action, state, data):
    try:
        assert ACTION_SPACE.contains(action)
        # print("Action:", action)
    except AssertionError:
        print(f"Invalid action: {action}")

    temp_state = state.copy()
    temp_state["replica"] = action[0] +1 
    temp_state["cpu"] = action[1] + 4
    temp_state["heap"] = action[2] + 4

    search_temp_state = list(temp_state.values())[0:3]
    idx = matching_indexes(data, search_temp_state)

    selected_row = random.choice(idx)
    new_state = dict({"replica": data.iloc[selected_row,0], 
                    "cpu": data.iloc[selected_row,1], 
                    "heap": data.iloc[selected_row,2],
                    "previous_tps": np.array([data.iloc[selected_row,3]]),
                    "instant_tps": np.array([data.iloc[selected_row,4]])})
    
    performance = round(data.iloc[selected_row, 7] /  (USERS * EXPECTED_TPS),6)
    utilization = (data.iloc[selected_row, 5]/(new_state["cpu"]/10)+data.iloc[selected_row, 6]/(new_state["heap"]/10))/2
    reward = ALPHA * performance + (1 - ALPHA) * utilization
    return new_state, reward



def initialize_state(data):
    idx = random.randint(0, len(data)-1)
    state = dict({"replica": data.iloc[idx,0], 
                    "cpu": data.iloc[idx,1], 
                    "heap": data.iloc[idx,2],
                    "previous_tps": np.array([data.iloc[idx,3]]),
                    "instant_tps": np.array([data.iloc[idx,4]])})
    
    return state

episode_id = POLICY_CLIENT.start_episode(training_enabled=True)
print('Episode started',episode_id)


prev_state = initialize_state(data)

step_count = 1
sum_reward = 0
columns = ["replica", "cpu", "heap", "previous_tps", "instant_tps", "reward", "sum_reward"]
output = pd.DataFrame(columns=columns)

while True:
    print('step',step_count)

    action = POLICY_CLIENT.get_action(episode_id, prev_state)
    state, reward = step(action, prev_state, data)

    print("action: ", action,"state:", state,"reward:", reward)

    sum_reward += reward
    print('cumulative reward calculated',sum_reward)

    POLICY_CLIENT.log_returns(episode_id, reward)
    print('policy_client.log_returns executed')


    if step_count % MAX_STEPS == 0:
        print("Total reward:", sum_reward)
        sum_reward = 0.0
        POLICY_CLIENT.end_episode(episode_id, state)
        print('episode ended...')
        episode_id = POLICY_CLIENT.start_episode(training_enabled=True)
        print('new episode started',episode_id)

    
    # temp = [state["replica"], state["cpu"], state["heap"], state["previous_tps"][0],state["instant_tps"][0],reward,sum_reward]
    # output.loc[step_count-1,:] = temp
    # output.to_csv("./results/test_output.csv", index=False)  
    prev_state = state
    step_count += 1

