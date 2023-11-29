import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
import gym
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from ray.tune.logger import pretty_print
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
import pandas as pd


CHECKPOINT_FILE = "last_checkpoint_{}.out"

def policy_input(context):
    return PolicyServerInput(context, "localhost", 9900)


def _input(ioctx):
    # We are remote worker or we are local worker with num_workers=0:
    # Create a PolicyServerInput.
    if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
        return PolicyServerInput(
            ioctx,
            "localhost",
            9900 + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
        )
    # No InputReader (PolicyServerInput) needed.
    else:
        return None


DATA_PATH = "/Users/hasan.nayir/Projects/Payten/APP2SCALE_RL/data/output_v5_offline_data.csv"
data = pd.read_csv(DATA_PATH)
ray.init(ignore_reinit_error=True)


config = (PPOConfig()
          .environment(
              env=None,
              action_space=Tuple([Discrete(6), Discrete(6),Discrete(6)]),
              observation_space=Dict({"replica": Discrete(6, start=1), 
                            "cpu": Discrete(6, start=4), 
                            "heap": Discrete(6, start=4),
                            "previous_tps": Box(data["previous_tps"].min(), data["previous_tps"].max()),
                            "instant_tps": Box(data["instant_tps"].min(), data["instant_tps"].max())}))

          .debugging(log_level="INFO")
          .rollouts(num_rollout_workers=0,
                    enable_connectors=False)
            .training(train_batch_size=64,sgd_minibatch_size=16,
                      model ={"fcnet_hiddens": [32, 32]},
                      lr=0.001)
          .offline_data(input_=_input)
        #   .evaluation(off_policy_estimation_methods={})
          )

config.rl_module(_enable_rl_module_api=False)
config.training(_enable_learner_api=False)

checkpoint_path = CHECKPOINT_FILE.format("PPO")
algo = config.build() 

path_to_checkpoint = "/Users/hasan.nayir/ray_results/PPO_teastore_2023-11-27_15-12-30auvasb4i/checkpoint_000001"


algo = algo.restore(path_to_checkpoint)


time_steps = 0
for epoch in range(1):
    print('server side epoch loop',epoch)
    results = algo.train() 
    print('algo.train executed')
    print(pretty_print(results))
    # if epoch % 20 == 0:
    #   checkpoint = algo.save()
    #   print("Last checkpoint", epoch, checkpoint)
checkpoint = algo.save()
print("Checkpoint has saved", checkpoint)
algo.stop()
