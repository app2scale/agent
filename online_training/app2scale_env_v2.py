import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import numpy as np
from collections import OrderedDict
import time
from kubernetes.client.exceptions import ApiException
import math


class Teastore(gym.Env):

    DO_NOTHING = 0
    INCREASE_REPLICA = 1
    DECREASE_REPLICA = 2
    INCREASE_CPU = 3
    DECREASE_CPU = 4
    INCREASE_HEAP = 5
    DECREASE_HEAP = 6
    MAX_STEPS = 10
    CPU_COST = 0.031611
    RAM_COST = 0.004237
    METRIC_DICT = {
        "container_network_receive_bytes_total": "inc_tps",
        "container_network_transmit_packets_total": "out_tps",
        "container_cpu_usage_seconds_total": "cpu_usage",
        "container_memory_working_set_bytes": "memory_usage"
    }
    COLLECT_METRIC_TIME = 10


    def __init__(self, metric_name_dict=METRIC_DICT, prometheus_host_url = "http://localhost:9090", 
                 deployment_name = "teastore-webui", 
                 namespace = "app2scale"): # To scale different deployment change deployment_name and namespace
        self.metric_name_dict = metric_name_dict
        self.prometheus_host_url = prometheus_host_url
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.prom = PrometheusConnect(url=prometheus_host_url, disable_ssl=True)
        self.action_space = Discrete(7)
        self.observation_space = Dict({"0replica": Discrete(9, start=1), 
                                       "1cpu": Discrete(9, start=1), 
                                       "2heap": Discrete(9, start=1),
                                       "3used_cpu": Box(-2, 2),
                                       "4used_ram": Box(-2, 2)})
        self.deployment = self._get_deployment_info()


    def _get_deployment_info(self):
        config.load_kube_config()
        v1 = client.AppsV1Api()
        return v1.read_namespaced_deployment(self.deployment_name, self.namespace)
    
    
    def collect_metrics_by_pod_names(self, running_pods):
        metrics = {}
        inc_tps = 0
        out_tps = 0
        cpu_usage = 0
        memory_usage = 0
        for pod in running_pods:
            while True:
                
                temp_inc_tps = self.prom.custom_query(query=f'sum(irate(container_network_receive_packets_total{{pod="{pod}", namespace="{self.namespace}"}}[2m]))')
                temp_out_tps = self.prom.custom_query(query=f'sum(irate(container_network_transmit_packets_total{{pod="{pod}", namespace="{self.namespace}"}}[2m]))')
                temp_cpu_usage = self.prom.custom_query(query=f'sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{{pod="{pod}", namespace="{self.namespace}"}})')
                temp_memory_usage = self.prom.custom_query(query=f'sum(container_memory_working_set_bytes{{pod="{pod}", namespace="{self.namespace}"}}) by (name)')

                if temp_inc_tps and temp_out_tps and temp_cpu_usage and temp_memory_usage:
                    inc_tps += float(temp_inc_tps[0]["value"][1])
                    out_tps += float(temp_out_tps[0]["value"][1])
                    cpu_usage += float(temp_cpu_usage[0]["value"][1])
                    memory_usage += float(temp_memory_usage[0]["value"][1])/1024/1024/1024
                    break
                else:
                    time.sleep(3)
                    continue

        metrics["inc_tps"] = round(inc_tps/len(running_pods))
        metrics["out_tps"] = round(out_tps/len(running_pods))
        metrics["cpu_usage"] = round(cpu_usage/len(running_pods),2)
        metrics["memory_usage"] = round(memory_usage/len(running_pods),2)

        return metrics
    

    def get_running_pods(self):
        time.sleep(3)
        config.load_kube_config()
        v1 = client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace=self.namespace, label_selector=f"run={self.deployment_name}")

        running_pods = []
        for pod in pods.items:
            if pod.status.phase.lower() == "running":
                running_pods.append(pod.metadata.name)

        return running_pods
    

    def apply_deployment_changes(self):
        config.load_kube_config()
        v1 = client.AppsV1Api()
        v1.patch_namespaced_deployment(self.deployment_name, self.namespace, self.deployment)

    def update_deployment_specs(self):
        self.deployment.spec.replicas = int(self.state["0replica"])
        self.deployment.spec.template.spec.containers[0].resources.limits["cpu"] = str(self.state["1cpu"]*100) + "m"
        self.deployment.spec.template.spec.containers[0].env[2].value = "-Xmx" + str(self.state["2heap"]*100) + "M"

    def check_all_pods_running(self):
        return self.deployment.spec.replicas == len(self.get_running_pods())
        
    def reset(self, *, seed=None, options=None):
        temp_sample = self.observation_space.sample()
        last_values_deployment = np.array([self.deployment.spec.replicas,
                                          int(int(self.deployment.spec.template.spec.containers[0].resources.limits["cpu"][:-1])/100), # Eskisine göre daha farklı. cpu stringi yok.
                                          int(int(self.deployment.spec.template.spec.containers[0].env[2].value[4:-1])/100)])
        while not self.check_all_pods_running():
            print("Waiting for all pods to be running...")
            time.sleep(5)
        running_pods = self.get_running_pods()
        metrics = self.collect_metrics_by_pod_names(running_pods)
        self.state = OrderedDict(zip(temp_sample.keys(), last_values_deployment))  #self.observation_space.sample() # obversion space'den random alabilir miyim? #Deployment'ın güncel halini yaz.
        self.state.update({"3used_cpu": np.array([metrics["cpu_usage"]], dtype=np.float32), "4used_ram": np.array([metrics["memory_usage"]], dtype=np.float32)})
        self.truncated = False
        self.terminated = False
        self.reward = 0.0
        cost = round((metrics["cpu_usage"]*self.CPU_COST + metrics["memory_usage"]*self.RAM_COST)*self.deployment.spec.replicas,2)
        self.info = {"inc_tps": metrics["inc_tps"], 
                     "out_tps": metrics["out_tps"], 
                     "cpu_usage": metrics["cpu_usage"], 
                     "memory_usage": metrics["memory_usage"],
                     "cost": cost}
        self.count = 0
        return self.state, self.info
    

    def step(self, action):

        assert self.action_space.contains(action)
        self.count += 1
        
        if action == self.DO_NOTHING:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, 0, 0])

        elif action == self.INCREASE_REPLICA:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([1, 0, 0])

        elif action == self.DECREASE_REPLICA:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([-1, 0, 0])
                
        elif action == self.INCREASE_CPU:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, 1, 0])
                
        elif action == self.DECREASE_CPU:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, -1, 0])
                
        elif action == self.INCREASE_HEAP:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, 0, 1])
                
        elif action == self.DECREASE_HEAP:
            temp_state = np.array(list(self.state.values())[:3]) + np.array([0, 0, -1])

        dummy_metrics = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
        updated_state = OrderedDict(zip(self.state.keys(), temp_state))
        updated_state.update({"3used_cpu": np.array([dummy_metrics[0]], dtype=np.float32), "4used_ram": np.array([dummy_metrics[1]],dtype=np.float32)})

        if self.observation_space.contains(updated_state):
            self.state = updated_state
            self.deployment = self._get_deployment_info() 
            self.update_deployment_specs()
            self.apply_deployment_changes()
            # self._get_deployment_infos()
            while not self.check_all_pods_running():
                print("Waiting for all pods to be running...")
                time.sleep(5)
            print("All pods are available!")
            print("Collecting metrics...")
            #self.env.runner.stats.reset_all()
            time.sleep(self.COLLECT_METRIC_TIME)
            running_pods = self.get_running_pods()
            metrics = self.collect_metrics_by_pod_names(running_pods)
            self.state.update({"3used_cpu": np.array([metrics["cpu_usage"]],dtype=np.float32), "4used_ram": np.array([metrics["memory_usage"]],dtype=np.float32)})
            cost = round((metrics["cpu_usage"]*self.CPU_COST + metrics["memory_usage"]*self.RAM_COST)*self.deployment.spec.replicas,2)
            # response_time = self.env.runner.stats.total.avg_response_time 
            # response_time = self.env.runner.stats.total.get_current_response_time_percentile(0.95) # 95th perc
            # (0.2-x)/(y+0.005) from x=0 to 0.4 and from y=0 to 0.3
            # self.reward = (1-math.exp(-(1-response_time/80)))/(1-metrics["cpu_usage"])
            # self.reward = (1-response_time/100)/(metrics["cpu_usage"] + 0.0001)
            self.reward = metrics["out_tps"]/7253 - cost/0.27
        else:
            print("Updated state is outside the observation space.")
            running_pods = self.get_running_pods()
            metrics = self.collect_metrics_by_pod_names(running_pods)
            #response_time = self.env.runner.stats.total.avg_response_time 
            self.state.update({"3used_cpu": np.array([metrics["cpu_usage"]],dtype=np.float32), "4used_ram": np.array([metrics["memory_usage"]],dtype=np.float32)})
            cost = round((metrics["cpu_usage"]*self.CPU_COST + metrics["memory_usage"]*self.RAM_COST)*self.deployment.spec.replicas/0.27,2)
            self.reward = -100


        if self.count == (self.MAX_STEPS):
            self.truncated = True
        
        
        self.info = {"inc_tps": metrics["inc_tps"], 
                     "out_tps": metrics["out_tps"], 
                     "cpu_usage": metrics["cpu_usage"], 
                     "memory_usage": metrics["memory_usage"],
                     "cost": cost,
                     #"response_time": response_time,
                     #"number_of_request": self.env.runner.stats.total.num_requests,
                     #"number_of_failures": self.env.runner.stats.total.num_failures
                     }
        return self.state, self.reward, self.terminated, self.truncated, self.info
    


