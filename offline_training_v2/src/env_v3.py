from typing import Tuple
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
from itertools import product
import time
import gymnasium as gym
import math

class Teastore(gym.Env):
    DATA_PATH = "/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/offline_training_v2/data/data.csv"
    MAX_STEPS = 100

    def __init__(self) -> None:
        super().__init__()   
        df = pd.read_csv(self.DATA_PATH)
        drop_rows = (df["cpu_usage"] != 0) | (df["memory_usage"] != 0)
        self.data = df[drop_rows].reset_index(drop=True)
        self.action_space = Discrete(5) # do_nothing, increase_replica, decrease_replica, increase_cpu, decrease_cpu
        self.observation_space = Box(low=np.array([1,4,0]), high=np.array([3,9,1000]), dtype=np.float32)
        self.count = 0
        self.info = {}


    def find_next_state(self, target):
        equal_rows = np.all(self.data.iloc[:, :2].values == target, axis=1)
        matched_indexes = np.where(equal_rows)[0]
        return matched_indexes.tolist()
    
    def calculate_reward(self, data):
        cpu_utilization = np.minimum(data["cpu_usage"]/(data["cpu"]/10), 1)
        performance_num_request = np.minimum(round(data['num_request'] /  (data['expected_tps']),6),1)
        performance_resp = (math.tanh((20-data["response_time"]))+1)/2

        reward = 0.2*cpu_utilization + 0.6*performance_num_request + 0.2*performance_resp
        return reward
    

    def reset(self, *, seed=None, options=None):
        idx = random.randint(0, len(self.data)-1)
        self.state = np.array(self.data.loc[idx, ["replica", "cpu", "expected_tps"]])
        self.truncated = False
        self.terminated = False
        self.reward = 0
        self.count = 0
        self.info = {}
        return self.state, self.info
    

    def step(self, action):
        selected_row_idx = 0
        self.count += 1

        if action == 0:
            temp_state = self.state[:2] + np.array([0, 0])
        elif action == 1: # increase_replica
            temp_state = self.state[:2] + np.array([1,0])
        elif action == 2: # decrease_replica
            temp_state = self.state[:2] + np.array([-1,0])
        elif action == 3: # increase_cpu
            temp_state = self.state[:2] + np.array([0, 1])
        else: # decrease_cpu
            temp_state = self.state[:2] + np.array([0,-1])

        idx = self.find_next_state(temp_state)

        if idx:
            selected_row_idx = random.choice(idx)
            selected_data = self.data.iloc[selected_row_idx]
            self.state = np.array(selected_data[["replica", "cpu", "expected_tps"]])
            self.reward = self.calculate_reward(selected_data)
        else:
            self.reward = -2


        self.terminated = (self.count >= self.MAX_STEPS)
        self.truncated = self.terminated
        return self.state, self.reward, self.terminated, self.truncated, self.info