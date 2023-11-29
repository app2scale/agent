from typing import Optional, Tuple
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from collections import OrderedDict
import pandas as pd
import numpy as np
import random


class Teastore(gym.Env):
    DATA_PATH = "/Users/hasan.nayir/Projects/Payten/APP2SCALE_RL/data/output_v5_offline_data.csv"
    DO_NOTHING = 0
    INCREASE_REPLICA = 1
    DECREASE_REPLICA = 2
    INCREASE_CPU = 3
    DECREASE_CPU = 4
    INCREASE_HEAP = 5
    DECREASE_HEAP = 6
    MAX_STEPS = 100
    EXPECTED_TPS = 50
    USERS = 1
    ALPHA = 0.6

    def matching_indexes(self, target):
        equal_rows = np.all(self.data.iloc[:, 0:3].values == target, axis=1)
        matched_indexes = np.where(equal_rows)[0]
        return matched_indexes.tolist()


    def __init__(self):
            self.data = pd.read_csv(self.DATA_PATH)
            # action_space is not similar to the observation_space. It should be start from zero. So, we will scale it whilen taking action in step function
            # replica : 1,2,3,4,5,6 -> 0,1,2,3,4,5 + 1
            # cpu : 4,5,6,7,8,9 -> 0,1,2,3,4,5   +   4
            # heap : 4,5,6,7,8,9 -> 0,1,2,3,4,5   +   4
            self.action_space = Tuple([Discrete(6), Discrete(6),Discrete(6)]) # This action_space definition is not supported by SAC. Tuple should not be used
            # self.action_space = Box(low=np.array([0,0,0]), high=np.array([5,5,5]), dtype=int)
            # self.action_space = Box(low = 0, high=5, shape=(3,), dtype=np.int32)
            self.observation_space = Dict({"replica": Discrete(6, start=1),
                                      "cpu": Discrete(9, start=4),
                                      "heap": Discrete(9, start=4),
                                      "previous_tps": Box(0, 200, dtype=np.float16),
                                      "instant_tps": Box(0, 200, dtype=np.float16)})

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
        performance = round(self.data.iloc[idx, 7] /  (self.USERS * self.EXPECTED_TPS),6)
        utilization = (self.data.iloc[idx, 5]/(self.state["cpu"]/10)+self.data.iloc[idx, 6]/(self.state["heap"]/10))/2
        self.reward = self.ALPHA * performance + (1 - self.ALPHA) * utilization
        self.count = 0
        self.info = {}
        return self.state, self.info    
    
    def step(self, action):
        # print("Action:", action)
        try:
            assert self.action_space.contains(action)
            # print("Action:", action)
        except AssertionError:
            print(f"Invalid action: {action}")

        selected_row = 0
        self.count += 1

        temp_state = self.state.copy()
        temp_state["replica"] = action[0] +1 
        temp_state["cpu"] = action[1] + 4
        temp_state["heap"] = action[2] + 4


        action_to_search = action + np.array([1, 4, 4])

        idx = self.matching_indexes(action_to_search)
        if idx:
            selected_row = random.choice(idx)
            self.state = dict({"replica": self.data.iloc[selected_row,0], 
                            "cpu": self.data.iloc[selected_row,1], 
                            "heap": self.data.iloc[selected_row,2],
                            "previous_tps": np.array([self.data.iloc[selected_row,3]]),
                            "instant_tps": np.array([self.data.iloc[selected_row,4]])})
            
            performance = round(self.data.iloc[selected_row, 7] /  (self.USERS * self.EXPECTED_TPS),6)
            utilization = (self.data.iloc[selected_row, 5]/(self.state["cpu"]/10)+self.data.iloc[selected_row, 6]/(self.state["heap"]/10))/2
            self.reward = self.ALPHA * performance + (1 - self.ALPHA) * utilization
        else:
            pass
            # print("Outside of the observation space!")


        self.terminated = (self.count >= self.MAX_STEPS)
        self.truncated = self.terminated

        return self.state, self.reward, self.terminated, self.truncated, self.info