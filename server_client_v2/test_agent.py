from gevent import monkey
monkey.patch_all(thread=False, select=False)

from ray.rllib.env.policy_client import PolicyClient
import pandas as pd
import time
import numpy as np
from collections import OrderedDict
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from ray.rllib.algorithms.ppo import PPOConfig

import ssl
import random
import logging
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.policy_server_input import PolicyServerInput
#from locust import HttpUser, task, constant, constant_throughput, events
#from locust.shape import LoadTestShape


ssl._create_default_https_context = ssl._create_unverified_context





previous_tps = 0
METRIC_DICT = {
    "container_network_receive_bytes_total": "inc_tps",
    "container_network_transmit_packets_total": "out_tps",
    "container_cpu_usage_seconds_total": "cpu_usage",
    "container_memory_working_set_bytes": "memory_usage"
}
# How many seconds to wait for transition
COOLDOWN_PERIOD = 0
# How many seconds to wait for metric collection
COLLECT_METRIC_TIME = 15
# Maximum number of metric collection attempt
COLLECT_METRIC_MAX_TRIAL = 200
# How many seconds to wait when metric collection fails
COLLECT_METRIC_WAIT_ON_ERROR = 2
# How many seconds to wait if pods are not ready
CHECK_ALL_PODS_READY_TIME = 2
# Episode length (set to batch size on purpose)
EPISODE_LENGTH = 100
PROMETHEUS_HOST_URL = "http://localhost:9090"
# Weight of the performance in the reward function
ALPHA = 0.6

DEPLOYMENT_NAME = "teastore-webui"
NAMESPACE = "app2scale"

OBSERVATION_SPACE = Dict({"replica": Discrete(6, start=1), 
                           "cpu": Discrete(6, start=4), 
                           "heap": Discrete(6, start=4),
                           "previous_tps": Box(0, 200, dtype=np.float16),
                           "instant_tps": Box(0, 200, dtype=np.float16)})

ACTION_SPACE = Tuple([Discrete(6), Discrete(6),Discrete(6)])

logging.getLogger().setLevel(logging.INFO)

expected_tps = 1

def policy_input(context):
    return PolicyServerInput(context, "localhost", 9900)


ray.init(ignore_reinit_error=True)

config_ppo = (PPOConfig()
          .environment(
              env=None,
              action_space=Tuple([Discrete(6), Discrete(6),Discrete(6)]),
              observation_space=Dict({"replica": Discrete(6, start=1), 
                                      "cpu": Discrete(9, start=4), 
                                      "heap": Discrete(9, start=4),
                                      "previous_tps": Box(0, 200, dtype=np.float16),
                                      "instant_tps": Box(0, 200, dtype=np.float16)}))

          .debugging(log_level="INFO")
          .rollouts(num_rollout_workers=0, enable_connectors=False)
            .training(train_batch_size=32,sgd_minibatch_size=16,
                      model ={"fcnet_hiddens": [64, 64]}, lr=0.001)
         # .offline_data(input_=policy_input)
          .evaluation(off_policy_estimation_methods={})
          )

config_ppo.rl_module(_enable_rl_module_api=False)
config_ppo.training(_enable_learner_api=False)
algo = config_ppo.build() 


done = False
truncated = False
sum_reward = 0
checkpoint_dir = "/root/ray_results/PPO_None_2023-12-13_16-13-28gfb08q9q/"
policy_name = "checkpoint_000401"
path_to_checkpoint = checkpoint_dir + policy_name
algo.restore(path_to_checkpoint)
step_count = 1

for _ in range(0,2):
    obs = {'replica': 6, 'cpu': 9, 'heap': 6, 'previous_tps': np.array([50.], dtype=np.float16), 'instant_tps': np.array([50.], dtype=np.float16)}

    print("obsss", obs)
    action = algo.compute_single_action(obs)
    print("asdasdasd", obs, "asdasdasd", action)
    obs = {'replica': 3, 'cpu': 6, 'heap': 4, 'previous_tps': np.array([50.], dtype=np.float16), 'instant_tps': np.array([50.], dtype=np.float16)}


    
