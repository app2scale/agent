from gevent import monkey
monkey.patch_all(thread=False, select=False)

from locust.env import Environment
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
import random
from locust.shape import LoadTestShape
import logging
ssl._create_default_https_context = ssl._create_unverified_context


MAX_STEPS = 32
previous_tps = 0
METRIC_DICT = {
    "container_network_receive_bytes_total": "inc_tps",
    "container_network_transmit_packets_total": "out_tps",
    "container_cpu_usage_seconds_total": "cpu_usage",
    "container_memory_working_set_bytes": "memory_usage"
}

WARM_UP_PERIOD = 60
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
PROMETHEUS_HOST_URL = "http://localhost:9090"
# Weight of the performance in the reward function
ALPHA = 0.6

DEPLOYMENT_NAME = "teastore-webui"
NAMESPACE = "app2scale"

OBSERVATION_SPACE = Box(low=np.array([1, 4, 4, 0, 0]), high=np.array([6, 9, 9, 200, 200]), dtype=np.float32)

"""
    # replica : 1,2,3,4,5,6 -> 0,1,2,3,4,5 + 1
    # cpu : 4,5,6,7,8,9 -> 0,1,2,3,4,5   +   4
    # heap : 4,5,6,7,8,9 -> 0,1,2,3,4,5   +   4
"""

ACTION_SPACE = Discrete(556, start=144)

POLICY_CLIENT = PolicyClient("http://localhost:9900", inference_mode="local") 
METRIC_SERVER = PrometheusConnect(url=PROMETHEUS_HOST_URL, disable_ssl=True)

logging.getLogger().setLevel(logging.INFO)
# Locust settings
expected_tps = 1
class TeaStoreLocust(HttpUser):
    wait_time = constant_throughput(expected_tps)
    host = "http://teastore.local.payten.com.tr/tools.descartes.teastore.webui/"

    @task
    def load(self):
        self.visit_home()
        #self.login()
        #self.browse()
        # 50/50 chance to buy
        #choice_buy = random.choice([True, False])
        #if choice_buy:
        #self.buy()
        #self.visit_profile()
        #self.logout()

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
                product_id = random.randint(7, 506)
                #product_id = 8
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
    return running_pods, len(pods.items)


def collect_metrics(env):
    deployment, state = get_deployment_info()
    while True:
        running_pods, number_of_all_pods = get_running_pods()
        if len(running_pods) == state["replica"] and state["replica"] == number_of_all_pods and running_pods:
            print("İnitial running pods", running_pods)
            break
        else:
            time.sleep(CHECK_ALL_PODS_READY_TIME)
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
        #running_pods, _ = get_running_pods()
        print("collect metric running pods", running_pods)
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
            #print("TEST", running_pods, len(running_pods))
            metrics['replica'] = state['replica']
            metrics['cpu'] = state['cpu']
            metrics['heap'] = state['heap']
            metrics["inc_tps"] = round(inc_tps/len(running_pods))
            metrics["out_tps"] = round(out_tps/len(running_pods))
            metrics["cpu_usage"] = round(cpu_usage/len(running_pods),3)
            metrics["memory_usage"] = round(memory_usage/len(running_pods),3)
            metrics['num_requests'] = round(env.runner.stats.total.num_requests/(COLLECT_METRIC_TIME + n_trials * COLLECT_METRIC_WAIT_ON_ERROR),2)
            metrics['num_failures'] = round(env.runner.stats.total.num_failures,2)
            metrics['response_time'] = round(env.runner.stats.total.avg_response_time,2)
            #print(env.runner.target_user_count, expected_tps)
            metrics['performance'] = min(round(metrics['num_requests'] /  (env.runner.target_user_count * expected_tps),6),1)
            metrics['expected_tps'] = env.runner.target_user_count * expected_tps*9 # 9 req for each user
            metrics['utilization'] = 0.5*min(metrics["cpu_usage"]/(state["cpu"]/10),1)+ 0.5*min(metrics["memory_usage"]/(state["heap"]/10),1)
            print('metric collection succesfull')
            load.ct += 1
            return metrics
    return None



def step(action, state, env):
    global previous_tps
    print('Entering step function')
    temp_state = state.copy()
    temp_state[0] = action // 100
    temp_state[1] = (action // 10) % 10
    temp_state[2] = action % 10
    updated_state = np.array([temp_state[0], temp_state[1], temp_state[2]])
    temp_updated_state = np.array([temp_state[0], temp_state[1], temp_state[2], 50, 50])

    print('applying the state...')
    update_and_deploy_deployment_specs(updated_state)
    new_state = temp_updated_state
    print('Entering cooldown period...')
    time.sleep(WARM_UP_PERIOD)
    time.sleep(COOLDOWN_PERIOD)
    print('cooldown period ended...')
    print('entering metric collection...')
    new_state[3] = previous_tps
    metrics = collect_metrics(env)
    new_state[4] = metrics["num_requests"]
    previous_tps = metrics["num_requests"]  
    print('updated_state', new_state)
    print('metrics collected',metrics)
    reward = ALPHA*metrics['performance'] + (1-ALPHA)*metrics['utilization']
    if metrics is None:
        return new_state, None, None

    metrics['reward'] = reward
    print('Calculated reward',reward)

    return new_state, reward, metrics




output_columns = ["replica", "cpu", "heap", "previous_tps", "instant_tps", "inc_tps", "out_tps", 
           "cpu_usage", "memory_usage", "reward", "sum_reward", 
           "response_time", "num_request", "num_failures","expected_tps"]

output = pd.DataFrame(columns=output_columns)
episode_id = POLICY_CLIENT.start_episode(training_enabled=True)
print('Episode started',episode_id)

_, prev_state = get_deployment_info()
step_count = 1
sum_reward = 0

while True:
    print('step',step_count)
    # First action will always be do nothing
    
    if step_count > 1:
        action = POLICY_CLIENT.get_action(episode_id, prev_state)
    else:
        action = prev_state[0]*100+prev_state[1]*10+prev_state[2] - 144
    state, reward, info = step(action, prev_state, env)
    if info is None or reward is None:
        print('info or reward is None so skip')
        prev_state = state
        step_count += 1
        continue

    sum_reward += reward
    print('cumulative reward calculated',sum_reward)

    POLICY_CLIENT.log_returns(episode_id, reward, info=info)
    print('policy_client.log_returns executed')

    if step_count % EPISODE_LENGTH == 0:
        print("Total reward:", sum_reward)
        sum_reward = 0.0
        POLICY_CLIENT.end_episode(episode_id, state)
        print('episode ended...')
        episode_id = POLICY_CLIENT.start_episode(training_enabled=True)
        print('new episode started',episode_id)
        
    prev_state = state

    temp_output = [state[0], state[1], state[2], state[3],
                   state[4], info["inc_tps"], info["out_tps"], info["cpu_usage"], 
                   info["memory_usage"], reward, sum_reward, info["response_time"],info["num_requests"], 
                   info["num_failures"],info["expected_tps"]]
        
    output.loc[step_count-1,:] = temp_output
    output.to_csv("output_new_1.csv", index=False)
    print(output,flush=True)
    step_count += 1
