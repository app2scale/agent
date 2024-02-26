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


def convert_data_to_batch(df, writer, eps_id_list):
    number_episodes = int(df.shape[0]/episode_length)
    remained_steps = df.shape[0]-number_episodes*episode_length
    for eps_id,idx in zip(eps_id_list, range(len(eps_id_list))):
        print(eps_id)
        try:
            data = df.iloc[idx*episode_length:(idx+1)*episode_length]
        except:
            data = df.iloc[idx*episode_length:(idx+1)*remained_steps]

        first_row = data.iloc[0,:]
        obs = np.array([first_row['replica'], first_row['cpu'], first_row['heap'], first_row['previous_tps'], first_row['instant_tps']], dtype=np.float32)
        info = {}
        possible_state_value = np.array([first_row['replica'], first_row['cpu'], first_row['heap']])
        equal_rows = np.all(POSSIBLE_STATES == possible_state_value, axis=1)
        prev_action = np.where(equal_rows)[0][0]
        prev_reward = 0
        terminated = truncated = False
        # print("aaa",data.shape[0])
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


def convert_data_to_batch_v2(df, writer):
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
       
        for i in range(1, data.shape[0]):
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
    
    action_space = Discrete(108) #6*6*6
    observation_space = Box(low=np.array([1, 4, 4, 0, 0]), high=np.array([3, 9, 9, 500, 500]), dtype=np.float32)
    replica = [1, 2, 3]
    cpu = [4, 5, 6, 7, 8, 9]
    heap = [4, 5, 6, 7, 8, 9]
    POSSIBLE_STATES = np.array(list(product(replica, cpu, heap)))


    prep = get_preprocessor(observation_space)(observation_space)
    print("The preprocessor is", prep)
    episode_length = 100
    # full_data_1 = pd.read_csv("/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/server_client_v2/output_new_1.csv")
    # full_data_2 = pd.read_csv("/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/server_client_v2/output_new_1_test_deployment.csv")
    # full_data = pd.concat([full_data_1, full_data_2]).reset_index()
    full_data = pd.read_csv("./server_client_v2_offline/hybrid_1.csv")
    # full_data = full_data.iloc[:2000,:]
    training_split_ratio = 0.8
    train_df = full_data.sample(frac=training_split_ratio, random_state=42)  # Eğitim verisi
    eval_df = full_data.drop(train_df.index) 
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    number_episodes_training = int(train_df.shape[0]/episode_length)
    remained_steps_training = train_df.shape[0]-number_episodes_training*episode_length
    if remained_steps_training > 0:
        number_episodes_training += 1

    # episode_length_eval = eval_df.shape[0]
    number_episodes_eval= int(eval_df.shape[0]/episode_length)
    remained_steps_eval = eval_df.shape[0]-number_episodes_eval*episode_length
    if remained_steps_eval > 0:
        number_episodes_eval += 1
        
    eps_id_training = list(range(number_episodes_training))
    eps_id_eval = list(range(len(eps_id_training), number_episodes_eval+ len(eps_id_training)))
    convert_data_to_batch(train_df, training_writer, eps_id_training)
    convert_data_to_batch(eval_df, eval_writer, eps_id_eval)
    # convert_data_to_batch_v2(train_df, training_writer)
    # convert_data_to_batch_v2(eval_df, eval_writer)
    # !!!! When we create batch as train_df and eval_df, eps_id is repeated and it will raise error like ValueError: eps_id 0 was already passed to the peek function. Make sure dataset contains only unique episodes with unique ids.
    



    