from locust.env import Environment
from websiteuser import WebsiteUser
from ray.rllib.env.policy_client import PolicyClient
import pandas as pd
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
import time
import numpy as np
from collections import OrderedDict
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from locust import HttpUser, task, constant, constant_throughput, events
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DO_NOTHING = 0
INCREASE_REPLICA = 1
DECREASE_REPLICA = 2
INCREASE_CPU = 3
DECREASE_CPU = 4
INCREASE_HEAP = 5
DECREASE_HEAP = 6
MAX_STEPS = 100
CPU_COST = 0.031611
RAM_COST = 0.004237
METRIC_DICT = {
    "container_network_receive_bytes_total": "inc_tps",
    "container_network_transmit_packets_total": "out_tps",
    "container_cpu_usage_seconds_total": "cpu_usage",
    "container_memory_working_set_bytes": "memory_usage"
}
# How many seconds to wait for transition
COOLDOWN_PERIOD = 30
# How many seconds to wait for metric collection
COLLECT_METRIC_TIME = 15
# Maximum number of metric collection attempt
COLLECT_METRIC_MAX_TRIAL = 100
# How many seconds to wait when metric collection fails
COLLECT_METRIC_WAIT_ON_ERROR = 1
# Episode length (set to batch size on purpose)
EPISODE_LENGTH = 128
PROMETHEUS_HOST_URL = "http://localhost:9090"

DEPLOYMENT_NAME = "teastore-webui"
NAMESPACE = "app2scale"

OBSERVATION_SPACE = Dict({"replica": Discrete(9, start=1), 
                           "cpu": Discrete(9, start=1), 
                           "heap": Discrete(9, start=1)})
ACTION_SPACE = Discrete(7)

policy_client = PolicyClient("http://localhost:9900", inference_mode="local") 
METRIC_SERVER = PrometheusConnect(url=PROMETHEUS_HOST_URL, disable_ssl=True)



# Locust settings
expected_tps = 50
users = 1
class TeaStoreLocust(HttpUser):
    wait_time = constant_throughput(expected_tps)
    host = "http://teastore.local.payten.com.tr"

    @task
    def my_task(self):
        response = self.client.get("/tools.descartes.teastore.webui/")

env = Environment(user_classes=[TeaStoreLocust])
env.create_local_runner()
env.runner.start(users, spawn_rate=1) 



def action_to_string(action):
    mapping = {0: "DO_NOTHING", 1: "INCREASE_REPLICA", 
               2: "DECREASE_REPLICA", 3: "INCREASE_CPU", 
               4: "DECREASE_CPU", 5: "INCREASE_HEAP", 6: "DECREASE_HEAP"}
    return mapping[action]

def get_state(deployment):
    replica = deployment.spec.replicas
    cpu = int(int(deployment.spec.template.spec.containers[0].resources.limits["cpu"][:-1])/100)
    heap = int(int(deployment.spec.template.spec.containers[0].env[2].value[4:-1])/100)
    return {"replica": replica, "cpu": cpu, "heap": heap}
   
def get_deployment_info():
    config.load_kube_config()
    v1 = client.AppsV1Api()
    deployment = v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
    state = get_state(deployment)
    return deployment, state


def update_and_deploy_deployment_specs(target_state):
    deployment, _ = get_deployment_info()
    deployment.spec.replicas = int(target_state["replica"])
    deployment.spec.template.spec.containers[0].resources.limits["cpu"] = str(target_state["cpu"]*100) + "m"
    deployment.spec.template.spec.containers[0].env[2].value = "-Xmx" + str(target_state["heap"]*100) + "M"

    config.load_kube_config()
    v1 = client.AppsV1Api()
    v1.patch_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE, deployment)

def get_running_pods():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(NAMESPACE, label_selector=f"run={DEPLOYMENT_NAME}")
    running_pods = []
    for pod in pods.items:
        if pod.status.phase.lower() == "running" and pod.status.container_statuses[0].ready and pod.metadata.deletion_timestamp == None:
            running_pods.append(pod.metadata.name)
    return running_pods

def collect_metrics(env):
    env.runner.stats.reset_all()
    time.sleep(COLLECT_METRIC_TIME)

    n_trials = 0
    while n_trials < COLLECT_METRIC_MAX_TRIAL:
        print('try count for metric collection',n_trials)
        metrics = {}
        inc_tps = 0
        out_tps = 0
        cpu_usage = 0
        memory_usage = 0

        empty_metric_situation_occured = False
        running_pods = get_running_pods()
        for pod in running_pods:
            temp_inc_tps = METRIC_SERVER.custom_query(query=f'sum(irate(container_network_receive_packets_total{{pod="{pod}", namespace="{NAMESPACE}"}}[2m]))')
            temp_out_tps = METRIC_SERVER.custom_query(query=f'sum(irate(container_network_transmit_packets_total{{pod="{pod}", namespace="{NAMESPACE}"}}[2m]))')
            temp_cpu_usage = METRIC_SERVER.custom_query(query=f'sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{{pod="{pod}", namespace="{NAMESPACE}"}})')
            temp_memory_usage = METRIC_SERVER.custom_query(query=f'sum(container_memory_working_set_bytes{{pod="{pod}", namespace="{NAMESPACE}"}}) by (name)')
         
            if temp_inc_tps and temp_out_tps and temp_cpu_usage and temp_memory_usage:
                inc_tps += float(temp_inc_tps[0]["value"][1])
                out_tps += float(temp_out_tps[0]["value"][1])
                cpu_usage += float(temp_cpu_usage[0]["value"][1])
                memory_usage += float(temp_memory_usage[0]["value"][1])/1024/1024/1024
            else:
                empty_metric_situation_occured = True
                break
            

        if empty_metric_situation_occured:
            n_trials += 1
            time.sleep(COLLECT_METRIC_WAIT_ON_ERROR)
        else:
            deployment, state = get_deployment_info()
            metrics['replica'] = state['replica']
            metrics['cpu'] = state['cpu']
            metrics['heap'] = state['heap']
            metrics["inc_tps"] = round(inc_tps/len(running_pods))
            metrics["out_tps"] = round(out_tps/len(running_pods))
            metrics["cpu_usage"] = round(cpu_usage/len(running_pods),3)
            metrics["memory_usage"] = round(memory_usage/len(running_pods),3)
            metrics["cost"] = round((metrics["cpu_usage"]*CPU_COST + metrics["memory_usage"]*RAM_COST)*len(running_pods),5)
            metrics['num_requests'] = round(env.runner.stats.total.num_requests/COLLECT_METRIC_TIME,3)
            metrics['num_failures'] = round(env.runner.stats.total.num_failures,3)
            metrics['response_time'] = round(env.runner.stats.total.avg_response_time,3)
            metrics['performance'] = round(metrics['num_requests'] /  (users * expected_tps),6)
            metrics['expected_tps'] = users * expected_tps
            metrics['utilization'] = (metrics["cpu_usage"]/(state["cpu"]/10)+metrics["memory_usage"]/(state["heap"]/10))/2
            print('metric collection succesfull')
            return metrics
    return None

def step(action, state, env):
    print('Entering step function')
    if action == DO_NOTHING:
        temp_state = np.array(list(state.values())[:3]) + np.array([0, 0, 0])
    elif action == INCREASE_REPLICA:
        temp_state = np.array(list(state.values())[:3]) + np.array([1, 0, 0])
    elif action == DECREASE_REPLICA:
        temp_state = np.array(list(state.values())[:3]) + np.array([-1, 0, 0])
    elif action == INCREASE_CPU:
        temp_state = np.array(list(state.values())[:3]) + np.array([0, 1, 0])
    elif action == DECREASE_CPU:
        temp_state = np.array(list(state.values())[:3]) + np.array([0, -1, 0])
    elif action == INCREASE_HEAP:
        temp_state = np.array(list(state.values())[:3]) + np.array([0, 0, 1])
    elif action == DECREASE_HEAP:
        temp_state = np.array(list(state.values())[:3]) + np.array([0, 0, -1])
    
    updated_state = {"replica": temp_state[0], "cpu": temp_state[1], "heap": temp_state[2]}
    print('updated_state', updated_state)

    if OBSERVATION_SPACE.contains(updated_state):
        print('applying the state...')
        update_and_deploy_deployment_specs(updated_state)
  
    print('Entering cooldown period...')
    time.sleep(COOLDOWN_PERIOD)
    print('cooldown period ended...')
    print('entering metric collection...')
    metrics = collect_metrics(env)
    print('metrics collected',metrics)
    if metrics is None:
        return new_state, None, None

    _, new_state = get_deployment_info()
   
    
    if OBSERVATION_SPACE.contains(updated_state):
        alpha = 0.9 # It indicates the importance of the performance
        reward = alpha*metrics['performance'] + (1-alpha)*metrics['utilization']
    else:
        reward = -1
    metrics['reward'] = reward
    print('Calculated reward',reward)
    return new_state, reward, metrics

columns = ["action", "replica", 
           "cpu", "heap", "inc_tps", "out_tps", 
           "cpu_usage", "memory_usage", "cost", "reward", "sum_reward", 
           "response_time", "num_request", "num_failures","expected_tps"]
output = pd.DataFrame(columns=columns)


episode_id = policy_client.start_episode(training_enabled=True)
print('Episode started',episode_id)

_, prev_state = get_deployment_info()
step_count = 1
sum_reward = 0
while True:
    print('step',step_count)
    # First action will always be do nothing
    action = policy_client.get_action(episode_id, prev_state) if step_count > 1 else DO_NOTHING
    str_action = action_to_string(action)
    print('action selected',str_action)
    state, reward, info = step(action, prev_state, env)
    if info is None:
        print('info is None so skip')
        prev_state = state
        step_count += 1
        continue

    sum_reward += reward
    print('cumulative reward calculated',sum_reward)

    policy_client.log_returns(episode_id, reward, info=info)
    print('policy_client.log_returns executed')

    temp = [str_action,
            state["replica"], state["cpu"], state["heap"], 
            info["inc_tps"], info["out_tps"], info["cpu_usage"], 
            info["memory_usage"], info["cost"], reward, sum_reward, info["response_time"],
            info["num_requests"], info["num_failures"],info["expected_tps"]]
    output.loc[step_count-1,:] = temp
    print(output,flush=True)

    output.to_csv("output.csv", index=False)
    if step_count % EPISODE_LENGTH == 0:
        print("Total reward:", sum_reward)
        sum_reward = 0.0
        policy_client.end_episode(episode_id, state)
        print('episode ended...')
        episode_id = policy_client.start_episode(training_enabled=True)
        print('new episode started',episode_id)
        
    prev_state = state
    step_count += 1
