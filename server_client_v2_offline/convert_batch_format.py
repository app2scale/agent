import gymnasium as gym
import numpy as np
import os
import pandas as pd
import ray._private.utils
from gymnasium.spaces import Discrete, Box
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from itertools import product


def convert_data_to_batch(df, writer):
    number_episodes = int(df.shape[0]/episode_length)
    remained_steps = df.shape[0]-number_episodes*episode_length
    for eps_id in range(0, number_episodes+1):
        try:
            data = df.iloc[eps_id*episode_length:(eps_id+1)*episode_length]
        except:
            data = df.iloc[eps_id*episode_length:(eps_id+1)*remained_steps]

        first_row = data.iloc[0,:]
        obs = np.array([first_row['replica'], first_row['cpu'], first_row['heap'], first_row['previous_tps'], first_row['instant_tps']], dtype=np.float32)
        info = {}
        possible_state_value = np.array([first_row['replica'], first_row['cpu'], first_row['heap']])
        equal_rows = np.all(POSSIBLE_STATES == possible_state_value, axis=1)
        prev_action = np.where(equal_rows)[0][0]
        prev_reward = 0
        terminated = truncated = False
        for i in range(0, data.shape[0]):
            truncated = True if i == data.shape[0]-1 else False
            terminated = truncated
            selected_row = data.iloc[i,:]
            possible_state_value = np.array([selected_row['replica'], selected_row['cpu'], selected_row['heap']])
            equal_rows = np.all(POSSIBLE_STATES == possible_state_value, axis=1)
            action = np.where(equal_rows)[0][0]
            new_obs = np.array([selected_row['replica'], selected_row['cpu'], selected_row['heap'], selected_row['previous_tps'], selected_row['instant_tps']], dtype=np.float32)
            rew = selected_row["reward"]
            batch_builder.add_values(
                t=i,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0, 
                action_logp=0.0,
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                terminateds=terminated,
                truncateds=truncated,
                infos=info,
                new_obs=prep.transform(new_obs),

            )
            obs = new_obs
            prev_action = action
            prev_reward = rew
        writer.write(batch_builder.build_and_reset())




if __name__ == "__main__":
    batch_builder = SampleBatchBuilder()
    training_writer = JsonWriter(
        os.path.join(ray._private.utils.get_user_temp_dir(), "training-out")
    )
    eval_writer = JsonWriter(
        os.path.join(ray._private.utils.get_user_temp_dir(), "eval-out")
    )
    
    action_space = Discrete(216) #6*6*6
    observation_space = Box(low=np.array([1, 4, 4, 0, 0]), high=np.array([6, 9, 9, 200, 200]), dtype=np.float32)
    replica = [1, 2, 3, 4, 5, 6]
    cpu = [4, 5, 6, 7, 8, 9]
    heap = [4, 5, 6, 7, 8, 9]
    POSSIBLE_STATES = np.array(list(product(replica, cpu, heap)))


    prep = get_preprocessor(observation_space)(observation_space)
    print("The preprocessor is", prep)
    episode_length = 100
    # full_data_1 = pd.read_csv("/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/server_client_v2/output_new_1.csv")
    # full_data_2 = pd.read_csv("/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/server_client_v2/output_new_1_test_deployment.csv")
    # full_data = pd.concat([full_data_1, full_data_2]).reset_index()
    full_data = pd.read_csv("/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/server_client_v2_offline/response_time_new_data.csv")
    training_split_ratio = 0.8
    train_df = full_data.sample(frac=training_split_ratio, random_state=42)  # Eğitim verisi
    eval_df = full_data.drop(train_df.index) 

    convert_data_to_batch(train_df, training_writer)
    convert_data_to_batch(eval_df, eval_writer)
    



    