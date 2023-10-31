import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
import gym
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from ray.tune.logger import pretty_print
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm

ray.init()
CHECKPOINT_FILE = "last_checkpoint_{}.out"

def policy_input(context):
    return PolicyServerInput(context, "localhost", 9900)

# DQN config yapılabilir. 
config = (PPOConfig()
          .environment(
              env=None,
              action_space=Discrete(7),
              observation_space=Dict({"replica": Discrete(6, start=1), 
                                      "cpu": Discrete(9, start=4), 
                                      "heap": Discrete(9, start=4),
                                      "previous_tps": Box(0, 200, dtype=np.float16),
                                      "instant_tps": Box(0, 200, dtype=np.float16)}))

          .debugging(log_level="INFO")
          .rollouts(num_rollout_workers=0,
                    enable_connectors=False)
            .training(train_batch_size=128,sgd_minibatch_size=128,
                      model ={"fcnet_hiddens": [64, 64]})
          .offline_data(input_=policy_input)
          .evaluation(off_policy_estimation_methods={}))

config.rl_module(_enable_rl_module_api=False)
config.training(_enable_learner_api=False)

checkpoint_path = CHECKPOINT_FILE.format("PPO")

algo = config.build() # It is commented to load saved pretrained model.
# algo = Algorithm.from_checkpoint("/root/ray_results/PPO_None_2023-10-17_15-44-345ng2ct98/checkpoint_003116/")
time_steps = 0
for epoch in range(5000):
    print('server side epoch loop',epoch)
    results = algo.train() 
    print('algo.train executed')
    print(pretty_print(results))
    if epoch % 20 == 0:
      checkpoint = algo.save()
      print("Last checkpoint", epoch, checkpoint)

algo.stop()
