from teastore import Teastore
import gym, ray
from ray.tune.registry import register_env
from gym.envs.registration import register
from ray.rllib.algorithms import ppo
import numpy as np
import pandas as pd



def run_one_episode(env):
    env.reset()
    sum_reward = 0

    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()
        
        state, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        print(f"action: {action + np.array([1, 4, 4])}, \nstate: {state}, reward: {reward}")
        if truncated or terminated:
            print("done @ step {}".format(i+1))
            break
    
    print("sum reward: ", sum_reward)
    return sum_reward

if __name__ == "__main__":

    env = Teastore()    
    history = []
    columns = ["replica", "cpu", "heap", "previous_tps", "instant_tps", "reward", "sum_reward"]
    output = pd.DataFrame(columns=columns)
    output_array = []
    ct = 0
    env.reset()
    for _ in range(10):
        sum_reward = 0
        for i in range(env.MAX_STEPS):
            action = tuple(env.data.iloc[ct,0:3])
            state, reward, terminated, truncated, info = env.step(action)
            sum_reward += reward
            print(f"action: {action + np.array([1, 4, 4])}, \nstate: {state}, reward: {reward}")
            if truncated or terminated:
                print("done @ step {}".format(i+1))
                break

            temp = [state["replica"], state["cpu"], state["heap"], state["previous_tps"][0],state["instant_tps"][0],reward,sum_reward]
            output_array.append(temp)
            ct +=1
        
        print("sum reward: ", sum_reward)

        history.append(sum_reward)
    
    output = pd.DataFrame(output_array, columns=columns)
    output.to_csv("./results/benchmark_output.csv", index=False)   
    avg_sum_reward = np.nanmean(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))