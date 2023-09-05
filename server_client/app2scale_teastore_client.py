import gymnasium as gym
from ray.rllib.env.policy_client import PolicyClient
from app2scale_env_v2 import Teastore
from ray.tune.registry import register_env
import pandas as pd

def action_to_string(action):
    mapping = {0: "DO_NOTHING", 1: "INCREASE_REPLICA", 
               2: "DECREASE_REPLICA", 3: "INCREASE_CPU", 
               4: "DECREASE_CPU", 5: "INCREASE_HEAP", 6: "DECREASE_HEAP"}
    return mapping[action]

env = Teastore()  
client = PolicyClient("http://localhost:9900", inference_mode="local") 

obs, info = env.reset()
episode_id = client.start_episode(training_enabled=True)

sum_reward = 0
columns = ["action", "replica", "cpu", "heap", "inc_tps", "out_tps", "cpu_usage", "memory_usage", "cost", "reward", "sum_reward"]
output = pd.DataFrame(columns=columns)
ct = 0
while True:
    action = client.get_action(episode_id, obs)
    obs, reward, terminated, truncated, info = env.step(action)
    sum_reward += reward
    client.log_returns(episode_id, reward, info=info)
    str_action = action_to_string(action)

    temp = [str_action, obs["0replica"], obs["1cpu"], obs["2heap"], 
            info["inc_tps"], info["out_tps"], info["cpu_usage"], 
            info["memory_usage"], info["cost"], reward, sum_reward]
    output.loc[ct,:] = temp
    print(output)
    if terminated or truncated:
        print("Total reward:", sum_reward)

        sum_reward = 0.0

        client.end_episode(episode_id, obs)
        obs, info = env.reset()
        episode_id = client.start_episode(training_enabled=True)

    ct += 1
