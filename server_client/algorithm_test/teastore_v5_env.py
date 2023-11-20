from typing import Optional, Tuple
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from collections import OrderedDict
import pandas as pd
import numpy as np
import random


class Teastore(gym.Env):
    DATA_PATH = "/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/server_client/algorithm_test/data/output_v5_offline_data.csv"
    DO_NOTHING = 0
    INCREASE_REPLICA = 1
    DECREASE_REPLICA = 2
    INCREASE_CPU = 3
    DECREASE_CPU = 4
    INCREASE_HEAP = 5
    DECREASE_HEAP = 6
    MAX_STEPS = 50


    def matching_indexes(self, target):
        equal_rows = np.all(self.data.iloc[:, 0:3].values == target, axis=1)
        matched_indexes = np.where(equal_rows)[0]
        return matched_indexes.tolist()


    def __init__(self):
        self.data = pd.read_csv(self.DATA_PATH)
        self.action_space = Discrete(7)
        self.observation_space = Dict({"replica": Discrete(6, start=1), 
                           "cpu": Discrete(6, start=4), 
                           "heap": Discrete(6, start=4),
                           "previous_tps": Box(self.data["previous_tps"].min(), self.data["previous_tps"].max()),
                           "instant_tps": Box(self.data["instant_tps"].min(), self.data["instant_tps"].max())}) 
    

    def reset(self, *, seed=None, options=None):
        idx = random.randint(0, len(self.data)-1)
        try:
            self.state = dict({"replica": self.data.iloc[idx,0], 
                            "cpu": self.data.iloc[idx,1], 
                            "heap": self.data.iloc[idx,2],
                            "previous_tps": np.array([self.data.iloc[idx,3]]),
                            "instant_tps": np.array([self.data.iloc[idx,4]])})
        except Exception as e:
            print(e)
        self.truncated = False
        self.terminated = False
        self.reward = self.data.iloc[idx,5] 
        self.count = 0
        self.info = {}
        return self.state, self.info
    
    def step(self, action):
        selected_row = 0
        self.count += 1
        if action == self.DO_NOTHING:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, 0, 0])
        elif action == self.INCREASE_REPLICA:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([1, 0, 0])

        elif action == self.DECREASE_REPLICA:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([-1, 0, 0])
                
        elif action == self.INCREASE_CPU:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, 1, 0])
                
        elif action == self.DECREASE_CPU:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, -1, 0])
                
        elif action == self.INCREASE_HEAP:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, 0, 1])
                
        elif action == self.DECREASE_HEAP:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, 0, -1])

        idx = self.matching_indexes(temp_state)
        if idx:
            selected_row = random.choice(idx)
            self.state = dict({"replica": self.data.iloc[selected_row,0], 
                            "cpu": self.data.iloc[selected_row,1], 
                            "heap": self.data.iloc[selected_row,2],
                            "previous_tps": np.array([self.data.iloc[selected_row,3]]),
                            "instant_tps": np.array([self.data.iloc[selected_row,4]])})
            
            self.reward = self.data.iloc[selected_row,5]
        else:
            pass
            # print("Outside of the observation space!")


        self.terminated = (self.count >= self.MAX_STEPS)
        self.truncated = self.terminated

        return self.state, self.reward, self.terminated, self.truncated, self.info

         
