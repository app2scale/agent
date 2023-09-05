from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from typing import Dict, Optional, Tuple, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
import numpy as np



class MetricCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))

        episode.user_data["inc_tps"] = []
        episode.user_data["out_tps"] = []
        episode.user_data["cpu_usage"] = []
        episode.user_data["memory_usage"] = []
        episode.user_data["cost"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        """
        Custom metrics and reward will be added.
        user data'da her key için append episode sonunda ortalama bunu da custom metrics
        start fonksiyonunda last_infos gelmiyor
        """

        for keys in episode._last_infos["agent0"].keys():
            episode.user_data[keys].append(episode._last_infos["agent0"].get(keys, 0))


    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        """
        Mean of the custom metrics and reward will be calculated.
        
        """
        episode.user_data["inc_tps"] = 0
        episode.user_data["out_tps"] = 0
        episode.user_data["cpu_usage"] = 0
        episode.user_data["memory_usage"] = 0
        episode.user_data["cost"] = 0

        for keys in episode._last_infos["agent0"].keys():
            episode.custom_metrics[keys] = np.nanmean(episode.user_data[keys])


    
