import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
import gym
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from ray.tune.logger import pretty_print

ray.init()
CHECKPOINT_FILE = "last_checkpoint_{}.out"

def policy_input(context):
    return PolicyServerInput(context, "localhost", 9900)

# DQN config yapılabilir. 
config = (PPOConfig()
          .environment(
              env=None,
              action_space=Discrete(7),
              observation_space=Dict({"0replica": Discrete(9, start=1),
                                      "1cpu": Discrete(9, start=1),
                                      "2heap": Discrete(9, start=1),
                                      "3used_cpu": Box(-2,2),
                                      "4used_ram": Box(-2,2)}))
          .debugging(log_level="INFO")
          .rollouts(num_rollout_workers=0,
                    enable_connectors=False)
            .training(train_batch_size=128)
          .offline_data(input_=policy_input)
          .evaluation(off_policy_estimation_methods={}))

config.rl_module(_enable_rl_module_api=False)
config.training(_enable_learner_api=False)

checkpoint_path = CHECKPOINT_FILE.format("PPO")

algo = config.build()
time_steps = 0
for epoch in range(10000):
    results = algo.train() 
    print(pretty_print(results))
    if epoch % 1 == 0:
      checkpoint = algo.save()
      print("Last checkpoint", epoch, checkpoint)

algo.stop()
