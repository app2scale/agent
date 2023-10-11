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
COLLECT_METRIC_TIME = 5
DEPLOYMENT_NAME = "teastore-webui"
NAMESPACE = "app2scale"

OBSERVATION_SPACE = Dict({"0replica": Discrete(9, start=1), 
                                       "1cpu": Discrete(9, start=1), 
                                       "2heap": Discrete(9, start=1),
                                       "3used_cpu": Box(-2, 2),
                                       "4used_ram": Box(-2, 2)})
ACTION_SPACE = Discrete(7)

PROMETHEUS_HOST_URL = "http://localhost:9090"


# Locust settings
expected_tps = 5
users = 1
class TeaStoreLocust(HttpUser):
    wait_time = constant_throughput(expected_tps)
    host = "http://10.27.41.24:30080"

    @task
    def my_task(self):
        response = self.client.get("/tools.descartes.teastore.webui/")


def action_to_string(action):
    mapping = {0: "DO_NOTHING", 1: "INCREASE_REPLICA", 
               2: "DECREASE_REPLICA", 3: "INCREASE_CPU", 
               4: "DECREASE_CPU", 5: "INCREASE_HEAP", 6: "DECREASE_HEAP"}
    return mapping[action]

def _get_deployment_info():
    config.load_kube_config()
    v1 = client.AppsV1Api()
    return v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)

def update_and_deploy_deployment_specs(deployment, state):
    deployment.spec.replicas = int(state["0replica"])
    deployment.spec.template.spec.containers[0].resources.limits["cpu"] = str(state["1cpu"]*100) + "m"
    deployment.spec.template.spec.containers[0].env[2].value = "-Xmx" + str(state["2heap"]*100) + "M"

    config.load_kube_config()
    v1 = client.AppsV1Api()
    v1.patch_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE, deployment)

def get_running_pods():
    time.sleep(1)
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(NAMESPACE, label_selector=f"run={DEPLOYMENT_NAME}")

    running_pods = []

    for pod in pods.items:
        if pod.status.phase.lower() == "running":
            running_pods.append(pod.metadata.name)

    return running_pods

def check_all_pods_running(deployment):
    return deployment.spec.replicas == len(get_running_pods())

def collect_metrics_by_pod_names(running_pods, prom):
    metrics = {}
    inc_tps = 0
    out_tps = 0
    cpu_usage = 0
    memory_usage = 0
    for pod in running_pods:
        while True:
            
            temp_inc_tps = prom.custom_query(query=f'sum(irate(container_network_receive_packets_total{{pod="{pod}", namespace="{NAMESPACE}"}}[2m]))')
            temp_out_tps = prom.custom_query(query=f'sum(irate(container_network_transmit_packets_total{{pod="{pod}", namespace="{NAMESPACE}"}}[2m]))')
            temp_cpu_usage = prom.custom_query(query=f'sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{{pod="{pod}", namespace="{NAMESPACE}"}})')
            temp_memory_usage = prom.custom_query(query=f'sum(container_memory_working_set_bytes{{pod="{pod}", namespace="{NAMESPACE}"}}) by (name)')

            if temp_inc_tps and temp_out_tps and temp_cpu_usage and temp_memory_usage:
                inc_tps += float(temp_inc_tps[0]["value"][1])
                out_tps += float(temp_out_tps[0]["value"][1])
                cpu_usage += float(temp_cpu_usage[0]["value"][1])
                memory_usage += float(temp_memory_usage[0]["value"][1])/1024/1024/1024
                break
            else:
                time.sleep(3)
                # print(running_pods)
                # print("Empty prometheus query... (Trying again)")
                continue

    metrics["inc_tps"] = round(inc_tps/len(running_pods))
    metrics["out_tps"] = round(out_tps/len(running_pods))
    metrics["cpu_usage"] = round(cpu_usage/len(running_pods),2)
    metrics["memory_usage"] = round(memory_usage/len(running_pods),2)

    return metrics



def reset():
    deployment = _get_deployment_info()
    last_values_deployment = np.array([deployment.spec.replicas,
                                          int(int(deployment.spec.template.spec.containers[0].resources.limits["cpu"][:-1])/100), # Eskisine göre daha farklı. cpu stringi yok.
                                          int(int(deployment.spec.template.spec.containers[0].env[2].value[4:-1])/100)])
    while not check_all_pods_running(deployment):
        print("Waiting for all pods to be running...")
        time.sleep(1)
        deployment = _get_deployment_info()
    
    running_pods = get_running_pods()
    metrics = collect_metrics_by_pod_names(running_pods,prom)
    state = OrderedDict({"0replica": last_values_deployment[0],
                                      "1cpu": last_values_deployment[1],
                                      "2heap": last_values_deployment[2]})
    state.update({"3used_cpu": np.array([metrics["cpu_usage"]], dtype=np.float32), "4used_ram": np.array([metrics["memory_usage"]], dtype=np.float32)})

    cost = round((metrics["cpu_usage"]*CPU_COST + metrics["memory_usage"]*RAM_COST)*deployment.spec.replicas,2)

    info = {"inc_tps": metrics["inc_tps"], 
            "out_tps": metrics["out_tps"], 
            "cpu_usage": metrics["cpu_usage"], 
            "memory_usage": metrics["memory_usage"],
            "cost": cost}

    return state, info


def step(action, state, env, prom):

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
    
    dummy_metrics = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
    updated_state = OrderedDict({"0replica": temp_state[0],
                                    "1cpu": temp_state[1],
                                    "2heap": temp_state[2]})
    updated_state.update({"3used_cpu": np.array([dummy_metrics[0]], dtype=np.float32), 
                          "4used_ram": np.array([dummy_metrics[1]],dtype=np.float32)})
    alpha = 0.9 # It indicates the importance of the performance
    if OBSERVATION_SPACE.contains(updated_state):
        state = updated_state
        deployment = _get_deployment_info()
        update_and_deploy_deployment_specs(deployment=deployment, state=state)
        while not check_all_pods_running(deployment):
            print("Waiting for all pods to be running...")
            time.sleep(1)
            deployment = _get_deployment_info()
        print("All pods are available!")
        print("Collecting metrics...")
        env.runner.stats.reset_all()
        time.sleep(COLLECT_METRIC_TIME)
        num_requests_locust = env.runner.stats.total.num_requests
        response_time = env.runner.stats.total.avg_response_time 
        running_pods = get_running_pods()
        metrics = collect_metrics_by_pod_names(running_pods, prom)
        state.update({"3used_cpu": np.array([metrics["cpu_usage"]],dtype=np.float32), "4used_ram": np.array([metrics["memory_usage"]],dtype=np.float32)})
        cost = round((metrics["cpu_usage"]*CPU_COST + metrics["memory_usage"]*RAM_COST)*deployment.spec.replicas,2)
        performance = num_requests_locust/(COLLECT_METRIC_TIME*users*expected_tps)
        # print("perfomance:", performance)
        utilization = (state["3used_cpu"][0]/(state["1cpu"]/10)+state["4used_ram"][0]/(state["2heap"]/10))/2 # it returns as an array
        # print("utilization:", utilization, state["3used_cpu"][0],state["1cpu"]/10, state["4used_ram"][0],state["2heap"]/10)
        reward = alpha*performance + (1-alpha)*utilization
        # reward = (1-response_time/100)/(metrics["cpu_usage"] + 0.0001)
    else:
        print("Updated state is outside the observation space.")
        running_pods = get_running_pods()
        deployment = _get_deployment_info()
        metrics = collect_metrics_by_pod_names(running_pods, prom)
        response_time = env.runner.stats.total.avg_response_time 
        num_requests_locust = env.runner.stats.total.num_requests
        state.update({"3used_cpu": np.array([metrics["cpu_usage"]],dtype=np.float32), "4used_ram": np.array([metrics["memory_usage"]],dtype=np.float32)})
        cost = round((metrics["cpu_usage"]*CPU_COST + metrics["memory_usage"]*RAM_COST)*deployment.spec.replicas,2)
        reward = -1

    info = {"inc_tps": metrics["inc_tps"], 
                "out_tps": metrics["out_tps"], 
                "cpu_usage": metrics["cpu_usage"], 
                "memory_usage": metrics["memory_usage"],
                "cost": cost,
                "response_time": response_time,
                "number_of_request": num_requests_locust,
                "number_of_failures": env.runner.stats.total.num_failures,
                "expected_tps": users * expected_tps*COLLECT_METRIC_TIME}

    return state, reward, info


policy_client = PolicyClient("http://localhost:9900", inference_mode="local") 
prom = PrometheusConnect(url=PROMETHEUS_HOST_URL, disable_ssl=True)


env = Environment(user_classes=[TeaStoreLocust])
env.create_local_runner()
env.runner.start(users, spawn_rate=1) 


obs, info = reset()
episode_id = policy_client.start_episode(training_enabled=True)

sum_reward = 0
columns = ["action", "replica", "cpu", "heap", "inc_tps", "out_tps", 
           "cpu_usage", "memory_usage", "cost", "reward", "sum_reward", 
           "response_time", "num_request", "num_failures","expected_tps"]
output = pd.DataFrame(columns=columns)
step_count = 0
ct = 0
truncated = False
while True:
    action = policy_client.get_action(episode_id, obs)
    obs, reward, info = step(action, obs, env, prom)
    step_count +=1
    if step_count == MAX_STEPS:
        truncated = True
        step_count = 0
    sum_reward += reward
    policy_client.log_returns(episode_id, reward, info=info)
    str_action = action_to_string(action)

    temp = [str_action, obs["0replica"], obs["1cpu"], obs["2heap"], 
            info["inc_tps"], info["out_tps"], info["cpu_usage"], 
            info["memory_usage"], info["cost"], reward, sum_reward, info["response_time"],
            info["number_of_request"], info["number_of_failures"],info["expected_tps"]]
    output.loc[ct,:] = temp
    print(output)

    output.to_csv("output.csv", index=False)
    if truncated:
        print("Total reward:", sum_reward)
        sum_reward = 0.0
        policy_client.end_episode(episode_id, obs)
        obs, info = reset()
        episode_id = policy_client.start_episode(training_enabled=True)
        truncated = False


    ct += 1
