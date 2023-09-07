# Reinforcement Learning Based Autoscaler 
This repository contains the codes for the work done within the WP3 work package of the app2scale project


# Descriptions of the file and the scripts 
In this work package, an agent intended to function as an autoscaler has been attempted to be trained using different methods. The work from various trials has been grouped under different folders.
## kopf_operator
Kopf is a framework to build Kubernetes operators in Python. Like any framework, Kopf provides both the “outer” toolkit to run the operator, to talk to the Kubernetes cluster, and to marshal the Kubernetes events into the pure-Python functions of the Kopf-based operator, and the “inner” libraries to assist with a limited set of common tasks of manipulating the Kubernetes objects.

crd.yaml : This file is the custom resource definition for our object. Note the short names because they can be used as the aliases on the command line, when getting a list or an object of that kind.

kex_example_2.yaml: A demo custom resource for the Kopf example operators.

autoscaler_operator.py: The script is designed to monitor and autoscale a Kubernetes deployment based on metrics fetched from Prometheus. Every 30 seconds, it checks the metrics of the currently running pods of a specified deployment (teastore-webui by default). Metrics like incoming and outgoing packets, CPU usage, and memory consumption are aggregated from Prometheus and used to make decisions about the deployment's scaling. The script then adjusts the specifications of the deployment, such as the number of replicas, CPU limits, and heap limits. These adjustments, however, are currently hardcoded in the script. It relies on the kopf, kubernetes, and prometheus_api_client Python libraries to interact with both the Kubernetes cluster and the Prometheus monitoring tool.

### Short tutorial to run autoscaler_operator.py
Apply the CRD to the cluser:

```
kubectl apply -f crd.yaml
```
If you want to revert this operation (e.g., to try it again):

```
kubectl delete crd kopfexamples.kopf.dev
kubectl delete -f crd.yaml
```
Now, we can already create the objects of this kind, apply it to the cluster, modify and delete them.
```
kubectl apply -f kex_example_2.yaml
```
Get a list of the existing objects of this kind
```
kubectl get KopfExample
```

Finally, let's run the operator and see what will happen:
```
kopf run autoscaler_operator.py --verbose
```

Note: Before run the operator check the deployment name, namespace of the deployment. Also, to collect mectrics via prometheus, you need to write the URL address where you can access the Prometheus Dashboard for Prometheus_url in the script. After deciding that the server-client structure was more suitable for the autoscaler, the work on kopf_operator was not completed.

## offline_training
In this approach, an agent has been trained using offline data.

teastore_data.csv: Bu data 21168 sample içermektedir. Her bir sample ayrı state'ı ve bu statedeki deploymenta ait bazı metrikleri içermektedir. Data ile ilgili daha ayrıntılı bilgiye https://github.com/app2scale/gsa/tree/main reposundan erişilebilir.

teastore.py: The script defines a custom reinforcement learning environment named Teastore using the gymnasium library. The environment simulates a system for managing resources in a teastore. The agent in this environment has to decide on actions related to scaling resources, like adjusting replicas, CPU, and heap sizes based on data sourced from a CSV file (teastore_data.csv).

Key Points: 
- Actions: Seven defined actions allow the agent to make decisions like doing nothing, increasing/decreasing replicas, increasing/decreasing CPU limit, or increasing/decreasing heap limit.

- State: The state of the environment includes information about the current replica, CPU, heap, used CPU, and used RAM.

- Data Source: A CSV file (teastore_data.csv) provides data for the environment. It's preprocessed to adjust and rename columns, scale values, and compute rewards.

- Environment Dynamics: The step method, given an action by the agent, updates the state and computes the reward based on the changes in outgoing transactions per second (out_tps) and the associated costs. If the agent chooses an action that leads to an invalid state (not in the data), a penalty of -5 is applied. The episode terminates after a maximum of 50 steps.

- Additional Methods: 
    - `reset`: Reinitializes the environment by picking a random row from the dataset.
    - `matching_indexes`: Identifies rows in the dataset that align with a specified target state.
    - `data_preprocess`:  Scales, renames, and rounds off certain columns to prepare the data for the environment.


train.py: This code is designed to train a reinforcement learning agent in a specific environment named "Teastore" using the Proximal Policy Optimization (PPO) algorithm. During training, the script prints out results at specified intervals to monitor the training process and saves algorithm checkpoints. The training is initialized and terminated with the ray library. In essence, this script automates the training of an agent to solve a specific task while continually monitoring the progression of the training.

sample.py: This code is designed to test the performance of an agent within the "Teastore" environment without any prior training. Specifically, for 50 episodes, the agent takes random actions within the environment and collects the rewards. After performing each action, details such as the chosen action, the resulting state, accompanying information, and the reward received are printed to the console. At the end of all episodes, the average cumulative reward from these 50 episodes is computed and displayed. The main objective of this script is to test environment.

test.py: This code is designed to evaluate a previously trained policy on the "Teastore" environment. Initially, the code sets up the environment and loads a trained policy from a specified checkpoint directory. Once initialized, the agent starts taking actions in the environment based on the trained policy. After each action, the new observation (state), chosen action (in string format), received reward, and cumulative reward are printed to the console. To make the agent's progress more observable, there's a one-second delay (`time.sleep(1)`) between each action. The agent continues to take actions until the environment signals that the episode is done. The primary purpose is to observe how a trained policy performs in real-time within the "Teastore" environment.

## online_training
The approach defines a custom environment for the TeaStore application using the OpenAI Gym framework. The environment is designed to interact in real-time with a running Kubernetes cluster, which has the TeaStore deployment. Unlike offline_training, everything takes place in real-time based on metrics collected from the TeaStore application running on the Kubernetes cluster and the interventions made to this application.


app2scale_env_v2.py: 

- Initialization: The Teastore class establishes itself as a gym environment with specific observation and action spaces. It uses the Kubernetes API and Prometheus (a monitoring tool) to gather metrics about the running state of the TeaStore application. The environment's observation space consists of the number of replicas, CPU limits, heap size, current CPU usage, and RAM usage.
The action space defines potential actions an agent can take, such as increasing/decreasing the number of replicas, adjusting CPU limits, and altering heap memory.

- Metrics Collection: A series of methods (e.g., collect_metrics_by_pod_names) is provided to fetch and process various metrics from the running TeaStore pods using Prometheus. This includes incoming and outgoing traffic, CPU usage, and memory consumption.

- Interacting with Kubernetes: The script can adjust the deployment of TeaStore on Kubernetes in response to actions taken in the environment. For instance, it can change the number of replicas or the CPU limit. It ensures that changes are correctly applied and that pods are in a running state before moving on.

- Environment Dynamics:
The step function dictates the environment's behavior in response to an action. When an action is taken, the Kubernetes deployment is updated accordingly, and metrics are collected. The reward is computed based on the outgoing traffic of the application and its associated cost, effectively trying to optimize the performance-cost tradeoff. After a certain number of steps (MAX_STEPS), the environment is reset.

app2scale_train.py, app2scale_test.py and app2scale_sample.py: These codes work in the same way as those in offline_training.


## server_client
In many situations, it does not make sense for an environment to be “stepped” by RLlib. For example, if a policy is to be used in a web serving system, then it is more natural for an agent to query a service that serves policy decisions, and for that service to learn from experience over time. This case also naturally arises with external simulators that run independently outside the control of RLlib, but may still want to leverage RLlib for training.

RLlib provides the ExternalEnv class for this purpose. Unlike other envs, ExternalEnv has its own thread of control. At any point, agents on that thread can query the current policy for decisions via self.get_action() and reports rewards, done-dicts, and infos via self.log_returns(). This can be done for multiple concurrent episodes as well.

In this setup, after navigating to the directory where the script is located in a terminal, the server is started by calling the command below.
```
python3 app2scale_teastore_server_v2.py
```

In this implementation, it's assumed that our agent requires external data to support its learning process. This data is fetched via a policy server. In this code, it's assumed that this server operates on `localhost` at port 9900.

A configuration is established for the PPO algorithm. Within this setup:

- **Action and Observation Spaces**: Defined domains for actions the agent can take and observations it can perceive.
- **Training Parameters**: Parameters for the training phase are laid out.
- **Debug Information**: The level of debug information to be logged is set.
- **Data Source**: The policy server is designated as the primary source of data for training the agent.

Training Loop:

The agent undergoes iterative training based on the given configuration. During this:

- The agent trains in each iteration.
- Results from each training phase are printed to the console.
- The latest state of the trained model is saved for potential future use or evaluation.


After the server is up and running, the client side is initiated by executing the code below from the terminal.
```
python3 app2scale_teastore_client_v2.py
```
The script interacts with a Kubernetes cluster, Prometheus for monitoring, and an RL agent. It adjusts the deployment's replica count, CPU limits, and JVM heap size based on metrics from Prometheus. The RL agent guides the scaling decisions by suggesting actions to maximize a reward based on system performance and cost.

Configuration

Various configurations and constants are defined:

- **Actions**: Defines different types of actions the agent can take, e.g., increase replicas, decrease CPU, etc.
- **Metrics**: Dictionary to map Prometheus metrics to more readable names.
- **Observation Space**: Defines the space of observable states the agent can see. It includes metrics such as the number of replicas, CPU usage, RAM usage, etc.
- **Action Space**: Defines the possible actions the agent can take.

Functions

1. `action_to_string`: Converts action IDs to string representations.
2. `_get_deployment_info`: Fetches information about a Kubernetes deployment.
3. `update_deployment_specs`: Modifies a deployment's specs based on the current state.
4. `apply_deployment_changes`: Applies changes made to a deployment.
5. `get_running_pods`: Lists all running pods of a deployment.
6. `check_all_pods_running`: Checks if all pods of a deployment are running.
7. `collect_metrics_by_pod_names`: Collects metrics for given pod names using Prometheus.
8. `reset`: Resets the environment to its initial state.
9. `step`: Defines how the environment responds to an action taken by the agent.

Workflow

1. A connection is made to the RLlib policy server and Prometheus.
2. The locust environment is initialized and started with specified user load.
3. The environment is reset to its initial state.
4. In a loop, the agent decides an action based on the current state, the environment responds with a new state, reward, and additional info.
5. The agent logs returns for the taken action.
6. Information about actions taken and their results are stored in a Pandas DataFrame, which is printed to the console.
7. After a certain number of steps (`MAX_STEPS`), the episode is terminated, the environment is reset, and results are saved to a CSV file.

Notes

- Ensure `websiteuser.py` (or its equivalent) is in the same directory or is importable.
- Update the `DEPLOYMENT_NAME` and `NAMESPACE` variables to match your Kubernetes deployment.
- Make sure Prometheus has access to the Kubernetes cluster metrics. To access the Prometheus dashboard and collect metrics, it is necessary to do port-forwarding. The code below can be run in a separate terminal for this purpose.
```
kubectl port-forward svc/rancher-monitoring-prometheus 9090:9090 -n cattle-monitoring-system
```