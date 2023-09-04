from teastore import Teastore
import gym, ray
from ray.tune.registry import register_env
from gym.envs.registration import register
from ray.rllib.algorithms import ppo
import numpy as np

def run_one_episode(env):
    env.reset()
    sum_reward = 0

    for i in range(env.MAX_STEPS):
        # action = 0
        action = env.action_space.sample()
        # action=1
        
        state, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        print(f"action: {action}, \nstate: {state}, \ninfo: {info}, reward: {reward}")
        if truncated or terminated:
            print("done @ step {}".format(i+1))
            break
    
    print("sum reward: ", sum_reward)
    return sum_reward

if __name__ == "__main__":

    env = Teastore()    
    history = []
    for _ in range(50):
        sum_reward = run_one_episode(env)
        history.append(sum_reward)
    
    avg_sum_reward = np.nanmean(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))