
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
import pandas as pd
import numpy as np
import random
from collections import OrderedDict


class Teastore(gym.Env):
    DATA_PATH = "offline_training/data/teastore_data.csv"
    DO_NOTHING = 0
    INCREASE_REPLICA = 1
    DECREASE_REPLICA = 2
    INCREASE_CPU = 3
    DECREASE_CPU = 4
    INCREASE_HEAP = 5
    DECREASE_HEAP = 6
    MAX_STEPS = 50
    STATE_COLUMN_INDEXES = [0, 1, 2, 7, 8]
  
    def __init__(self):
        self.data = pd.read_csv(self.DATA_PATH, delim_whitespace=True)
        # self.data = self.data.sample(5)
        self.data_preprocess()
        self.action_space = Discrete(7) 
        self.observation_space = Dict({"0replica": Discrete(9, start=1), 
                                       "1cpu": Discrete(9, start=1), 
                                       "2heap": Discrete(9, start=1),
                                       "3used_cpu": Box(self.data["used_cpu"].min(), self.data["used_cpu"].max()),
                                       "4used_ram": Box(self.data["used_ram"].min(), self.data["used_ram"].max())})

    def reset(self, *, seed=None, options=None):
        # idx = random.randint(0, len(self.data)-1)
        idx = 558
        try:
            self.state = OrderedDict({"0replica": self.data.iloc[idx, 0], 
                                "1cpu": self.data.iloc[idx, 1], 
                                "2heap": self.data.iloc[idx, 2],
                                "3used_cpu": np.array([self.data.iloc[idx, 7]]),
                                "4used_ram": np.array([self.data.iloc[idx, 8]])})
        except Exception as e:
            print(e)
        self.truncated = False
        self.terminated = False
        self.reward = 0.0
        self.info = {"inc_tps": self.data.iloc[idx,3], "out_tps": self.data.iloc[idx,4], "cpu_usage":self.data.iloc[idx,7],
                     "memory_usage": self.data.iloc[idx,8], "cost": self.data.iloc[idx,9]}
        self.count = 0
        return self.state, self.info
    
    def matching_indexes(self, target):
        # equal_rows = np.all(self.data.iloc[:, 0:3].values == list(target.values())[:3], axis=1)
        equal_rows = np.all(self.data.iloc[:, 0:3].values == target, axis=1)
        matched_indexes = np.where(equal_rows)[0]
        return matched_indexes.tolist()

    def data_preprocess(self):
        temp_state = self.data.iloc[:, 0:10]
        temp_state = temp_state.rename(columns={'re': 'source_replica', 
                                                'heap': 'source_heap',
                                                'cpu': 'source_cpu',
                                                'cpu.1': 'used_cpu',
                                                'ram': 'used_ram'})
        
        for i in range(3):
            temp_state.iloc[:,i] = 1 + (8*(temp_state.iloc[:,i] - temp_state.iloc[:,i].min()) / (temp_state.iloc[:,i].max() - temp_state.iloc[:,i].min())).astype(int)

        self.out_tps_reward = (self.data.iloc[:, 4] - self.data.iloc[:, 4].min())/(self.data.iloc[:, 4].max()-self.data.iloc[:, 4].min())
        self.cost_reward = (self.data.iloc[:, 9] - self.data.iloc[:, 9].min())/(self.data.iloc[:, 9].max()-self.data.iloc[:, 9].min())
        self.data = temp_state
        self.data["used_cpu"] = round(self.data["used_cpu"], 2)
        self.data["used_ram"] = round(self.data["used_ram"], 2)
        self.data["out_tps"] = round(self.data["out_tps"])
        self.data["inc_tps"] = round(self.data["inc_tps"])
        self.data["cost"] = round(self.data["cost"], 2)
        return self.data
        
    
    def step(self, action):
        selected_row = 0

        assert self.action_space.contains(action)
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
            temp = np.array(self.data.iloc[selected_row, self.STATE_COLUMN_INDEXES])
            self.state = OrderedDict({"0replica": int(temp[0]), 
                                "1cpu": int(temp[1]), 
                                "2heap": int(temp[2]),
                                "3used_cpu": np.array([temp[3]]),
                                "4used_ram": np.array([temp[4]])})
            if np.isnan(self.data.iloc[selected_row, 4]):
                self.reward = 0
            else:
                self.reward = self.out_tps_reward[selected_row] - self.cost_reward[selected_row]
        else:
            self.reward = -5

        self.terminated = (self.count >= self.MAX_STEPS)
        self.truncated = self.terminated
        
        self.info = {"inc_tps": self.data.iloc[selected_row, 3], "out_tps": self.data.iloc[selected_row, 4], "cpu_usage": self.data.iloc[selected_row, 7],
                    "memory_usage": self.data.iloc[selected_row, 8], "cost": self.data.iloc[selected_row, 9]}
        return self.state, self.reward, self.terminated, self.truncated, self.info

    


