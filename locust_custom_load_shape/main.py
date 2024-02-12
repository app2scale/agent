from gevent import monkey
monkey.patch_all(thread=False, select=False)

import logging
from random import randint, choice
from kubernetes import client, config
from locust import HttpUser, task, constant_throughput
from locust.shape import LoadTestShape
from locust.env import Environment
import pandas as pd
import numpy as np
import time
import random
from prometheus_api_client import PrometheusConnect

# logging
logging.getLogger().setLevel(logging.INFO)

expected_tps = 5
class TeaStoreLocust(HttpUser):
    wait_time = constant_throughput(expected_tps)
    host = "http://teastore-test.local.payten.com.tr/tools.descartes.teastore.webui/"
    
    @task
    def load(self) -> None:
        """
        Simulates user behaviour.
        :return: None
        """
        # logging.info("Starting user.")
        self.visit_home()
        self.login()
        self.browse()
        # 50/50 chance to buy
        #choice_buy = choice([True, False])
        #if choice_buy:
       # self.buy()
        self.visit_profile()
        self.logout()
        # logging.info("Completed user.")

    def visit_home(self) -> None:
        """
        Visits the landing page.
        :return: None
        """
        # load landing page
        res = self.client.get('/')
        if res.ok:
            pass
            # logging.info("Loaded landing page.")
        else:
            logging.error(f"Could not load landing page: {res.status_code}")

    def login(self) -> None:
        """
        User login with random userid between 1 and 90.
        :return: categories
        l
        """
        # load login page
        res = self.client.get('/login')
        if res.ok:
            pass
            # logging.info("Loaded login page.")
        else:
            logging.error(f"Could not load login page: {res.status_code}")
        # login random user
        user = randint(1, 99)
        #user = 2
        login_request = self.client.post("/loginAction", params={"username": user, "password": "password"})
        if login_request.ok:
            pass
            # logging.info(f"Login with username: {user}")
        else:
            logging.error(
                f"Could not login with username: {user} - status: {login_request.status_code}")

    def browse(self) -> None:
        """
        Simulates random browsing behaviour.
        :return: None
        """
        # execute browsing action randomly up to 5 times
        for i in range(1, 2):
            # browses random category and page
            category_id = randint(2, 6)
            page = randint(1, 5)
            category_request = self.client.get("/category", params={"page": page, "category": category_id})
            if category_request.ok:
                # logging.info(f"Visited category {category_id} on page 1")
                # browses random product
                product_id = random.randint((category_id-2)*100+7+(page-1)*20, (category_id-2)*100+26+(page-1)*20)
                product_request = self.client.get("/product", params={"id": product_id})
                if product_request.ok:
                    pass
                    # logging.info(f"Visited product with id {product_id}.")
                    cart_request = self.client.post("/cartAction", params={"addToCart": "", "productid": product_id})
                    if cart_request.ok:
                        pass
                        # logging.info(f"Added product {product_id} to cart.")
                    else:
                        logging.error(
                            f"Could not put product {product_id} in cart - status {cart_request.status_code}")
                else:
                    logging.error(
                        f"Could not visit product {product_id} - status {product_request.status_code}")
            else:
                logging.error(
                    f"Could not visit category {category_id} on page 1 - status {category_request.status_code}")

    def buy(self) -> None:
        """
        Simulates to buy products in the cart with sample user data.
        :return: None
        """
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
        """
        Visits user profile.
        :return: None
        """
        profile_request = self.client.get("/profile")
        if profile_request.ok:
            pass
            # logging.info("Visited profile page.")
        else:
            logging.error("Could not visit profile page.")

    def logout(self) -> None:
        """
        User logout.
        :return: None
        """
        logout_request = self.client.post("/loginAction", params={"logout": ""})
        if logout_request.ok:
            pass
            # logging.info("Successful logout.")
        else:
            logging.error(f"Could not log out - status: {logout_request.status_code}")



class CustomLoad(LoadTestShape):
    trx_load_data = pd.read_csv("./transactions.csv")
    trx_load = trx_load_data["transactions"].values.tolist()
    trx_load = (trx_load/np.max(trx_load)*20).astype(int)
    ct = 0
    
    def tick(self):
        if self.ct >= len(self.trx_load):
            self.ct = 0
        user_count = self.trx_load[self.ct]
        return (3, 3) 

load = CustomLoad()
env = Environment(user_classes=[TeaStoreLocust], shape_class=load)
env.create_local_runner()
#web_ui = env.create_web_ui("127.0.0.1", 8089)
env.runner.start_shape() 
env.stop_timeout=10000

def get_running_pods():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod("app2scale-test", label_selector="run=teastore-webui")
    running_pods = []
    for pod in pods.items:
        if pod.status.phase.lower() == "running" and pod.status.container_statuses[0].ready and pod.metadata.deletion_timestamp == None:
            running_pods.append(pod.metadata.name)
    return running_pods, len(pods.items)

def get_deployment_info():
    config.load_kube_config()
    v1 = client.AppsV1Api()
    deployment = v1.read_namespaced_deployment("teastore-webui", "app2scale-test")
    state = get_state(deployment)
    return deployment, state

def get_state(deployment):
    replica = deployment.spec.replicas
    cpu = int(int(deployment.spec.template.spec.containers[0].resources.limits["cpu"][:-1])/100)
    heap = int(int(deployment.spec.template.spec.containers[0].env[2].value[4:-1])/100)
    return np.array([replica, cpu, heap])

def get_usage_metrics_from_server(running_pods_array):
    config.load_kube_config()
    api = client.CustomObjectsApi()
    k8s_pods = api.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", "app2scale-test", "pods")
    usage_metric_server = {}

    for stats in k8s_pods['items']:
        if stats["metadata"]["name"] in running_pods_array:
            usage_metric_server[stats["metadata"]["name"]] = [round(float(stats["containers"][0]["usage"]["cpu"].rstrip('n'))/1e6), 
                                                          round(float(stats["containers"][0]["usage"]["memory"].rstrip('Ki'))/1024)]
            
    usage_metric_server["cpu"], usage_metric_server["memory"] = np.mean(np.array(list(usage_metric_server.values()))[:,0]), np.mean(np.array(list(usage_metric_server.values()))[:,1])
    return usage_metric_server

WARM_UP_PERIOD = 60
COOLDOWN_PERIOD = 0
# How many seconds to wait for metric collection
COLLECT_METRIC_TIME = 15
# Maximum number of metric collection attempt
COLLECT_METRIC_MAX_TRIAL = 200
# How many seconds to wait when metric collection fails
COLLECT_METRIC_WAIT_ON_ERROR = 2
# How many seconds to wait if pods are not ready
CHECK_ALL_PODS_READY_TIME = 2
PROMETHEUS_HOST_URL = "http://localhost:9090"
METRIC_SERVER=PrometheusConnect(url=PROMETHEUS_HOST_URL, disable_ssl=True)
DEPLOYMENT_NAME = "teastore-webui"
NAMESPACE = "app2scale-test"

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
        for pod in running_pods:
            temp_cpu_usage = METRIC_SERVER.custom_query(query=f'sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{{pod="{pod}", namespace="{NAMESPACE}"}})')
            temp_memory_usage = METRIC_SERVER.custom_query(query=f'sum(container_memory_working_set_bytes{{pod="{pod}", namespace="{NAMESPACE}"}}) by (name)')
         
            if temp_cpu_usage and temp_memory_usage:
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
            metric_server = get_usage_metrics_from_server(running_pods)
            metrics['replica'] = state[0]
            metrics['cpu'] = state[1]
            metrics['heap'] = state[2]
            metrics["cpu_usage_prom"] = round(cpu_usage/len(running_pods),3)
            metrics["memory_usage_prom"] = round(memory_usage/len(running_pods),3)
            metrics["cpu_usage_server"] = metric_server["cpu"]
            metrics["memory_usage_server"] = metric_server["memory"]
            metrics['num_requests'] = round(env.runner.stats.total.num_requests/(COLLECT_METRIC_TIME + n_trials * COLLECT_METRIC_WAIT_ON_ERROR),2)
            metrics['num_failures'] = round(env.runner.stats.total.num_failures,2)
            metrics['response_time'] = round(env.runner.stats.total.avg_response_time,2)
            #print(env.runner.target_user_count, expected_tps)
            #metrics['performance'] = min(round(metrics['num_requests'] /  (env.runner.target_user_count * expected_tps),6),1)
            metrics['expected_tps'] = env.runner.target_user_count * expected_tps*8 # 9 req for each user, it has changed now we just send request to the main page
            #metrics['utilization'] = 0.5*min(metrics["cpu_usage"]/(state[1]/10),1)+ 0.5*min(metrics["memory_usage"]/(state[2]/10),1)



 
            print('metric collection succesfull')
            load.ct += 1
            return metrics
    return None


columns = ["replica", "cpu", "heap", 
           "cpu_usage_prom", "memory_usage_prom","cpu_usage_server","memory_usage_server",
             "num_request", "num_failures","response_time","expected_tps"]
result_df = pd.DataFrame(columns=columns)

step = 0
while True:
 
  metrics = collect_metrics(env)
  result_df.loc[step,:] =list(metrics.values())
  print(result_df)
  #result_df.to_csv("./load_test.csv", index=False)
  step = step + 1


env.runner.quit()
