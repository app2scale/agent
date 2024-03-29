from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.offline.estimators import ImportanceSampling
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel
from gymnasium.spaces import Discrete, Box
import numpy as np
from ray.rllib.offline.estimators import ImportanceSampling, WeightedImportanceSampling
from itertools import product
import matplotlib.pyplot as plt
import inspect



def generate_config(train_path, eval_path, hyper_params):
    learning_rate = hyper_params[0]
    fcnet_hiddens = hyper_params[1]
    config = (
        DQNConfig()
        .environment(env=None,
                    action_space=Discrete(108), 
                    observation_space=Box(low=np.array([1, 4, 4, 0, 0]), high=np.array([3, 9, 9, 500, 500]), dtype=np.float32)
                    )
        .training(model={"fcnet_hiddens": fcnet_hiddens},
                  gamma=0.99,
                  lr=learning_rate,
                  train_batch_size=256
        )
                
        .offline_data(input_=train_path)
        .exploration(explore=False)
        .evaluation(evaluation_parallel_to_training=False,
                    evaluation_interval=2,
                    evaluation_duration=10,
                    evaluation_duration_unit="episodes",
                    evaluation_config={"input": eval_path},
                    off_policy_estimation_methods={"is": {"type": ImportanceSampling},
                                                   "wis": {"type": WeightedImportanceSampling}
                    }
        )
                    

    )
    return config


def generate_config_v2(train_path, eval_path, hyper_params):
    learning_rate = hyper_params[0]
    fcnet_hiddens = hyper_params[1]
    config = (
        DQNConfig()
        .environment(env=None,
                    action_space=Discrete(108), 
                    observation_space=Box(low=np.array([1, 4, 4]), high=np.array([3, 9, 9]), dtype=np.float32)
                    )
        .training(model={"fcnet_hiddens": fcnet_hiddens},
                  gamma=0.99,
                  lr=learning_rate,
                  train_batch_size=256
        )
                
        .offline_data(input_=train_path)
        .exploration(explore=False)
        .evaluation(evaluation_parallel_to_training=False,
                    evaluation_interval=2,
                    evaluation_duration=10,
                    evaluation_duration_unit="episodes",
                    evaluation_config={"input": eval_path},
                    off_policy_estimation_methods={"is": {"type": ImportanceSampling},
                                                   "wis": {"type": WeightedImportanceSampling}
                    }
        )
                    

    )
    return config

# Burada eval duration ve interval batch bağlantısı var. aynı eps_id üzerinden tekrar geçmemesi gerekir



def generate_plots(filename,  mean_q_list, is_v_gain_list,wis_v_gain_list):
    fig, ax = plt.subplots(nrows=len(inspect.signature(generate_plots).parameters)-1,ncols=1,figsize=(12,36))
    ax[0].plot(mean_q_list)
    ax[0].set_xlabel('step')
    ax[0].set_ylabel('mean_q')
    ax[1].plot(is_v_gain_list)
    ax[1].set_xlabel('step')
    ax[1].set_ylabel('is_v_gain')
    ax[2].plot(wis_v_gain_list)
    ax[2].set_xlabel('step')
    ax[2].set_ylabel('wis_v_gain')
    plt.savefig(filename)


hyperparameters = {"learning_rate": [1e-05],
                   "fcnet_hiddens": [[64,64]]
                   }
parameter_combinations = list(product(*hyperparameters.values())) # This variable includes all combinations of the hyperparameters. ex. (1e-05, [32, 32])


train_path = "/tmp/training-out"
eval_path = "/tmp/eval-out"
epoch_number = 4500

for comb in parameter_combinations:
    config = generate_config(train_path, eval_path, comb)
    print(f"Started training with lr: {comb[0]} and fcnet: {comb[1]}")
    mean_q_list = []
    is_v_gain_list = []
    wis_v_gain_list = []

    # config['observation_filter'] = "MeanStdFilter"
    algo = config.build()
    debug_dir = "{}checkpoints/".format(algo.logdir)
    filename = f"./server_client_v2_offline/results/{algo.logdir.split('/')[-2]}_lr_{comb[0]}_fcnet_{comb[1][0]}_{comb[1][1]}.pdf"
    for i in range(epoch_number):
        print("------------- Iteration", i+1, "-------------")
        results = algo.train()
        print("timesteps_total:", results['timesteps_total'])
        print("training_iteration_time_ms:", results['timers']['training_iteration_time_ms'])
        if 'evaluation' in results.keys():
            print("== Evaluation ==")
            if 'off_policy_estimator' in results['evaluation'].keys():
                print(results['evaluation']['off_policy_estimator'])
                is_v_gain_list.append(results['evaluation']['off_policy_estimator']["is"]["v_gain"])
                wis_v_gain_list.append(results['evaluation']['off_policy_estimator']["wis"]["v_gain"])


        if (i+1) % 1000 == 0:
            ma_checkpoint_dir = algo.save(checkpoint_dir=debug_dir)
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{ma_checkpoint_dir}'"
            )
        mean_q_list.append(results["info"]["learner"]["default_policy"]["learner_stats"]["mean_q"])

    algo.stop()
    generate_plots(filename, mean_q_list,is_v_gain_list,wis_v_gain_list)

    
