

from ray.rllib.env.policy_client import PolicyClient
import pandas as pd
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
import time
import numpy as np
from collections import OrderedDict
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
import ssl
import random
import logging
ssl._create_default_https_context = ssl._create_unverified_context
from itertools import product
import time
import gymnasium as gym
import math



class Teastore(gym.Env):
    DATA_PATH = "/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/offline_training_v2/data/data.csv"
    MAX_STEPS = 100


    def __init__(self) -> None:
        df = pd.read_csv(self.DATA_PATH)
        drop_rows = (df["cpu_usage"] != 0) | (df["memory_usage"] != 0)
        self.data = df[drop_rows].reset_index(drop=True)
        self.action_space = Discrete(108) 
        self.observation_space = Box(low=np.array([1, 4, 4, 0]), high=np.array([3, 9, 9, 1000]), dtype=np.float32)
        replica = [1, 2, 3]
        cpu = [4, 5, 6, 7, 8, 9]
        heap = [4, 5, 6, 7, 8, 9]
        self.possible_state = np.array(list(product(replica, cpu, heap)))
        self.count = 0
        self.info = {}



    def find_next_state(self, target):
        equal_rows = np.all(self.data.iloc[:, :3].values == target, axis=1)
        matched_indexes = np.where(equal_rows)[0]
        return matched_indexes.tolist()
    
    def calculate_reward(self, data):
        cpu_utilization = np.minimum(data["cpu_usage"]/(data["cpu"]/10),1)
        # performance_resp = math.tanh((20-data["response_time"]))
        performance_num = np.minimum(round(data['num_request'] /  (data['expected_tps']),6),1)

        # reward = performance_resp + performance_num+ cpu_utilization
        reward = cpu_utilization*performance_num
        return reward

    # def calculate_reward(self, data):
    #     cpu_utilization = np.minimum(data["cpu_usage"]/(data["cpu"]/10),1)
    #     return cpu_utilization


    
    def reset(self, *, seed=None, options=None):
        idx = random.randint(0, len(self.data)-1)
        self.state = np.array(self.data.loc[idx, ["replica", "cpu", "heap", "expected_tps"]])
        self.truncated = False
        self.terminated = False
        self.reward = 0
        self.count = 0
        self.info = {}
        return self.state, self.info
    
    def step(self, action):
        selected_row_idx = 0
        self.count += 1

 
        temp_state = self.possible_state[action]
        idx = self.find_next_state(temp_state)

        selected_row_idx = random.choice(idx)
        selected_data = self.data.iloc[selected_row_idx]
        self.state = np.array(selected_data[["replica", "cpu", "heap", "expected_tps"]])
        self.reward = self.calculate_reward(selected_data)

        self.terminated = (self.count >= self.MAX_STEPS)
        self.truncated = self.terminated

        return self.state, self.reward, self.terminated, self.truncated, self.info







