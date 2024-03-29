from gevent import monkey
monkey.patch_all(thread=False, select=False)
from locust.env import Environment
from kubernetes import client, config

from prometheus_api_client import PrometheusConnect

from ray.rllib.env.policy_client import PolicyClient
import pandas as pd
import time
import numpy as np
from collections import OrderedDict
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig

import ssl
import random
import logging
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.policy_server_input import PolicyServerInput
from locust import HttpUser, task, constant, constant_throughput, events
from locust.shape import LoadTestShape

ssl._create_default_https_context = ssl._create_unverified_context
from itertools import product
import time


MAX_STEPS = 32
previous_tps = 0
METRIC_DICT = {
    "container_network_receive_bytes_total": "inc_tps",
    "container_network_transmit_packets_total": "out_tps",
    "container_cpu_usage_seconds_total": "cpu_usage",
    "container_memory_working_set_bytes": "memory_usage"
}

WARM_UP_PERIOD = 300
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
PROMETHEUS_HOST_URL = "http://prometheus.local.payten.com.tr"
# Weight of the performance in the reward function
ALPHA = 0.8

DEPLOYMENT_NAME = "teastore-webui"
NAMESPACE = "app2scale-test"

OBSERVATION_SPACE =Box(low=np.array([1, 4, 4, 0]), high=np.array([3, 9, 9, 1000]), dtype=np.float32)
"""
    # replica : 1,2,3,4,5,6 -> 0,1,2,3,4,5 + 1
    # cpu : 4,5,6,7,8,9 -> 0,1,2,3,4,5   +   4
    # heap : 4,5,6,7,8,9 -> 0,1,2,3,4,5   +   4
"""

ACTION_SPACE = Discrete(108) # index of the possible states
replica = [1, 2, 3]
cpu = [4, 5, 6, 7, 8, 9]
heap = [4, 5, 6, 7, 8, 9]

POSSIBLE_STATES = np.array(list(product(replica, cpu, heap)))
METRIC_SERVER = PrometheusConnect(url=PROMETHEUS_HOST_URL, disable_ssl=True)

logging.getLogger().setLevel(logging.INFO)


ray.init(ignore_reinit_error=True)

expected_tps = 1
class TeaStoreLocust(HttpUser):
    wait_time = constant_throughput(expected_tps)
    host = "http://teastore-test.local.payten.com.tr/tools.descartes.teastore.webui/"

    @task
    def load(self):
        self.visit_home()
        self.login()
        self.browse()
        # 50/50 chance to buy
        #choice_buy = random.choice([True, False])
        #if choice_buy:
        # self.buy()
        self.visit_profile()
        self.logout()

    def visit_home(self):

        # load landing page
        res = self.client.get('/')
        if res.ok:
            pass
        else:
            logging.error(f"Could not load landing page: {res.status_code}")

    def login(self):

        # load login page
        res = self.client.get('/login')
        if res.ok:
            pass
        else:
            logging.error(f"Could not load login page: {res.status_code}")
        # login random user
        user = random.randint(1, 99)
        login_request = self.client.post("/loginAction", params={"username": user, "password": "password"})
        if login_request.ok:
            pass
        else:
            logging.error(
                f"Could not login with username: {user} - status: {login_request.status_code}")
            

    def browse(self):

        # execute browsing action randomly up to 5 times
        for i in range(1, 2):
            # browses random category and page
            category_id = random.randint(2, 6)
            page = random.randint(1, 5)
            #page = 1
            category_request = self.client.get("/category", params={"page": page, "category": category_id})
            if category_request.ok:
                # logging.info(f"Visited category {category_id} on page 1")
                # browses random product
                product_id = random.randint((category_id-2)*100+7+(page-1)*20, (category_id-2)*100+26+(page-1)*20)
                product_request = self.client.get("/product", params={"id": product_id})
                if product_request.ok:
                    #logging.info(f"Visited product with id {product_id}.")
                    cart_request = self.client.post("/cartAction", params={"addToCart": "", "productid": product_id})
                    if cart_request.ok:
                        pass
                        #logging.info(f"Added product {product_id} to cart.")
                    else:
                        logging.error(
                            f"Could not put product {product_id} in cart - status {cart_request.status_code}")
                else:
                    logging.error(
                        f"Could not visit product {product_id} - status {product_request.status_code}")
            else:
                logging.error(
                    f"Could not visit category {category_id} on page 1 - status {category_request.status_code}")
                


    def buy(self):

        # sample user data
        user_data = {
            "firstname": "User",
            "lastname": "User",
            "adress1": "Road",
            "adress2": "City",
            "cardtype": "volvo",
            "cardnumber": "314159265359",
            "expirydate": "12/2050",
            "confirm": "Confirm"
        }
        buy_request = self.client.post("/cartAction", params=user_data)
        if buy_request.ok:
            pass
            # logging.info(f"Bought products.")
        else:
            logging.error("Could not buy products.")

    def visit_profile(self) -> None:

        profile_request = self.client.get("/profile")
        if profile_request.ok:
            pass
            # logging.info("Visited profile page.")
        else:
            logging.error("Could not visit profile page.")

    def logout(self):

        logout_request = self.client.post("/loginAction", params={"logout": ""})
        if logout_request.ok:
            pass
            # logging.info("Successful logout.")
        else:
            logging.error(f"Could not log out - status: {logout_request.status_code}")


class CustomLoad(LoadTestShape):

    trx_load_data = pd.read_csv("./transactions.csv")
    trx_load = trx_load_data["transactions"].values.tolist()
    trx_load = (trx_load/np.max(trx_load)*20).astype(int)+1
    indexes = [(177, 184), (661, 685), (1143, 1152), (1498, 1524), (1858, 1900)]
    clipped_data = []
    for idx in indexes:
        start, end = idx
        clipped_data.extend(trx_load[start:end+1])
    ct = 0

    #clipped_data = (np.linspace(20, 150, 27, dtype=np.int32)/5).astype(int) 
    def tick(self):
        if self.ct >= len(self.clipped_data):
            self.ct = 0
        user_count = self.clipped_data[self.ct]
        return (user_count, user_count) 

load = CustomLoad()
env = Environment(user_classes=[TeaStoreLocust], shape_class=load)
env.create_local_runner()
env.runner.start_shape() 


def get_state(deployment):
    replica = deployment.spec.replicas
    cpu = int(int(deployment.spec.template.spec.containers[0].resources.limits["cpu"][:-1])/100)
    heap = int(int(deployment.spec.template.spec.containers[0].env[2].value[4:-1])/100)
    return np.array([replica, cpu, heap])

def get_deployment_info():
    config.load_kube_config()
    v1 = client.AppsV1Api()
    deployment = v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
    state = get_state(deployment)
    return deployment, state

def update_and_deploy_deployment_specs(target_state):
    deployment, _ = get_deployment_info()
    deployment.spec.replicas = int(target_state[0])
    deployment.spec.template.spec.containers[0].resources.limits["cpu"] = str(target_state[1]*100) + "m"
    deployment.spec.template.spec.containers[0].env[2].value = "-Xmx" + str(target_state[2]*100) + "M"

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
    return running_pods, len(pods.items)

def get_usage_metrics_from_server(running_pods_array):
    config.load_kube_config()
    api = client.CustomObjectsApi()
    k8s_pods = api.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", "app2scale-test", "pods")
    usage_metric_server = {}

    for stats in k8s_pods['items']:
        if stats["metadata"]["name"] in running_pods_array:
            try:
                usage_metric_server[stats["metadata"]["name"]] = [round(float(stats["containers"][0]["usage"]["cpu"].rstrip('n'))/1e9, 3),
                                                            round(float(stats["containers"][0]["usage"]["memory"].rstrip('Ki'))/(1024*1024),3)]
            except:
                usage_metric_server[stats["metadata"]["name"]] = [round(float(stats["containers"][0]["usage"]["cpu"].rstrip('n'))/1e9, 3),
                                            round(float(stats["containers"][0]["usage"]["memory"].rstrip('M'))/(1024),3)]

    usage_metric_server["cpu"], usage_metric_server["memory"] = np.mean(np.array(list(usage_metric_server.values()))[:,0]), np.mean(np.array(list(usage_metric_server.values()))[:,1])
    return usage_metric_server


def collect_metrics(env):
    deployment, state = get_deployment_info()
    while True:
        running_pods, number_of_all_pods = get_running_pods()
        if len(running_pods) == state[0] and state[0] == number_of_all_pods and running_pods:
            print("İnitial running pods", running_pods)
            break
        else:
            time.sleep(CHECK_ALL_PODS_READY_TIME)
    time.sleep(WARM_UP_PERIOD)
    env.runner.stats.reset_all()
    time.sleep(COLLECT_METRIC_TIME)
    n_trials = 0
    while n_trials < COLLECT_METRIC_MAX_TRIAL:
        print('try count for metric collection',n_trials)
        metrics = {}
        cpu_usage = 0
        memory_usage = 0

        empty_metric_situation_occured = False
        #running_pods, _ = get_running_pods()
        print("collect metric running pods", running_pods)
        try:
            metric_server = get_usage_metrics_from_server(running_pods)
            if metric_server["cpu"] and metric_server["memory"]:
                cpu_usage = metric_server["cpu"]
                memory_usage = metric_server["memory"]
            else:
                empty_metric_situation_occured = True
                break
        except Exception as e:
            print(e)
            

        if empty_metric_situation_occured:
            n_trials += 1
            time.sleep(COLLECT_METRIC_WAIT_ON_ERROR)
        else:
            #print("TEST", running_pods, len(running_pods))
            metrics['replica'] = state[0]
            metrics['cpu'] = state[1]
            metrics['heap'] = state[2]
            metrics["cpu_usage"] = cpu_usage
            metrics["memory_usage"] = memory_usage
            metrics['num_requests'] = round(env.runner.stats.total.num_requests/(COLLECT_METRIC_TIME + n_trials * COLLECT_METRIC_WAIT_ON_ERROR),2)
            metrics['num_failures'] = round(env.runner.stats.total.num_failures,2)
            metrics['response_time'] = round(env.runner.stats.total.avg_response_time,2)
            #print(env.runner.target_user_count, expected_tps)
            temp_perf_request = min(round(metrics['num_requests'] /  (env.runner.target_user_count * expected_tps*8),6),1)
            temp_perf_response = 0
            if metrics["response_time"] >=20:
                temp_perf_response  = 20/metrics["response_time"]
            else:
                temp_perf_response = 1
            metrics["performance"] = 0.5*temp_perf_response + 0.5*temp_perf_request 
            metrics['expected_tps'] = env.runner.target_user_count * expected_tps*8 # 9 req for each user, it has changed now we just send request to the main page
            #metrics['utilization'] = 0.5*min(metrics["cpu_usage"]/(state[1]/10),1)+ 0.5*min(metrics["memory_usage"]/(state[2]/10),1)
            metrics["utilization"] = min(metrics["cpu_usage"]/(state[1]/10),1)
            print('metric collection succesfull')
            load.ct += 1
            return metrics
    return None

def step(action, state, env):
    global previous_tps
    print('Entering step function')
    temp_state = state.copy()
    temp_state = POSSIBLE_STATES[action]
    updated_state = temp_state
    temp_updated_state = np.array([temp_state[0], temp_state[1], temp_state[2], 50])

    print('applying the state...')
    print("updated state", updated_state)
    update_and_deploy_deployment_specs(updated_state)
    deployment_time = time.time()
    new_state = temp_updated_state
    print('Entering cooldown period...')
    time.sleep(WARM_UP_PERIOD)
    print('cooldown period ended...')
    print('entering metric collection...')
    metrics = collect_metrics(env)
    new_state[3] = metrics["expected_tps"]
    print('updated_state', new_state)
    print('metrics collected',metrics)
    reward = ALPHA*metrics['performance'] + (1-ALPHA)*metrics['utilization']
    if metrics is None:
        return new_state, None, None

    metrics['reward'] = reward
    print('Calculated reward',reward)

    return new_state, reward, metrics, deployment_time


config_dqn = (DQNConfig()
          .environment(
              env=None,
              action_space=ACTION_SPACE,
              observation_space=OBSERVATION_SPACE)

          .training(model={"fcnet_hiddens": [64,64]},
              gamma=0,
              lr=1e-05,
              train_batch_size=256)
          .debugging(log_level="INFO")
          .evaluation(off_policy_estimation_methods={})
          
          )


config_dqn.rl_module(_enable_rl_module_api=False)
config_dqn.training(_enable_learner_api=False)
algo = config_dqn.build() 

output_columns = ["replica", "cpu", "heap", "previous_tps", "instant_tps", 
           "cpu_usage", "memory_usage", "reward", "sum_reward", 
           "response_time", "num_request", "num_failures","expected_tps", "timestamp"]

output = pd.DataFrame(columns=output_columns)
_, obs = get_deployment_info()
obs = np.append(obs, [50])

done = False
truncated = False
sum_reward = 0

path_to_checkpoint = "./checkpoint_ppo_1"
algo.restore(path_to_checkpoint)
step_count = 1

for _ in range(0,120):

    action = algo.compute_single_action(obs)
    next_obs, reward, info, timestamp = step(action,obs,env)
    sum_reward += reward
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    temp_output = [next_obs[0], next_obs[1], next_obs[2], next_obs[3],
                   next_obs[4], info["cpu_usage"], 
                   info["memory_usage"], reward, sum_reward, info["response_time"],info["num_requests"], 
                   info["num_failures"],info["expected_tps"], timestamp]
    output.loc[step_count,:] = temp_output
    print(output)
    output.to_csv("./test_results_gamma_0_tps.csv", index=False)
    obs = next_obs
    step_count += 1
