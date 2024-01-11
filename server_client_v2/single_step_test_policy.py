from gymnasium.spaces import Discrete, Box
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor
import torch
import tree
import numpy as np
from ray.rllib.policy.policy import Policy

policy_path = "/root/ray_results/PPO_None_2023-12-13_16-13-28gfb08q9q/checkpoint_000201/policies/default_policy"
restored_policy = Policy.from_checkpoint(policy_path)

obs = {'replica': np.array([5]), 'cpu': np.array([8]), "heap": np.array([2]), 'previous_tps': np.array([45.53]), 'instant_tps': np.array([46.44])}

struct_torch = tree.map_structure(lambda s: torch.from_numpy(s), obs)
spaces = dict(
    {
        "replica": Discrete(6, start=1),
        "cpu": Discrete(9, start=4),
        "heap": Discrete(9, start=4),
        "previous_tps":Box(0, 200, dtype=np.float16),
        "instant_tps":Box(0, 200, dtype=np.float16)
    }
)
model_input = flatten_inputs_to_1d_tensor(struct_torch, spaces_struct=spaces)

action = restored_policy.compute_single_action(model_input)
print(action)


