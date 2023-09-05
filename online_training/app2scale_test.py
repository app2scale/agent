from app2scale_env_v2 import Teastore
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorboard



def action_to_string(action):
    mapping = {0: "DO_NOTHING", 1: "INCREASE_REPLICA", 
               2: "DECREASE_REPLICA", 3: "INCREASE_CPU", 
               4: "DECREASE_CPU", 5: "INCREASE_HEAP", 6: "DECREASE_HEAP"}
    return mapping[action]


ray.init(ignore_reinit_error=True)
register_env("teastore", lambda config: Teastore())
policy_name = "app2scale_training/2023-08-18_14-20_pouRagM9/checkpoint_002500" # Saved policy path
algo = Algorithm.from_checkpoint(policy_name)
columns = ["action", "action_n","replica", "cpu", "heap", "inc_tps", "out_tps", "cpu_usage", "memory_usage", "cost", "reward", "sum_reward"]
output = pd.DataFrame(columns=columns)
env = Teastore()

obs, info = env.reset()
done = False
sum_reward = 0

if __name__ == "__main__":
    step = 0

    while not done:

        action = algo.compute_single_action(obs)
        next_obs, reward, done, truncated, info= env.step(action)
        sum_reward += reward
        str_action = action_to_string(action)
        # np.set_printoptions(formatter={'float': '{:0.4f}'.format})
        temp = [str_action, action,next_obs["0replica"], next_obs["1cpu"], next_obs["2heap"], 
                info["inc_tps"], info["out_tps"], info["cpu_usage"], 
                info["memory_usage"], info["cost"], reward, sum_reward]
        output.loc[step,:] = temp
        print(output)
        obs = next_obs

        step += 1



