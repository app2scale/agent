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


# logging
logging.getLogger().setLevel(logging.INFO)

expected_tps = 1
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
        #self.login()
        #self.browse()
        # 50/50 chance to buy
        #choice_buy = choice([True, False])
        #if choice_buy:
       # self.buy()
        #self.visit_profile()
        #self.logout()
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
        """
        # load login page
        res = self.client.get('/login')
        if res.ok:
            pass
            # logging.info("Loaded login page.")
        else:
            logging.error(f"Could not load login page: {res.status_code}")
        # login random user
        #user = randint(1, 99)
        user = 2
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
            #category_id = randint(2, 6)
            #page = randint(1, 5)
            category_id = 3
            page = 2
            category_request = self.client.get("/category", params={"page": page, "category": category_id})
            if category_request.ok:
                # logging.info(f"Visited category {category_id} on page 1")
                # browses random product
                #product_id = randint(7, 506)
                product_id = 10
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
    trx_load = (trx_load/np.max(trx_load)*10).astype(int)
    ct = 0
    
    def tick(self):
        if self.ct >= len(self.trx_load):
            self.ct = 0
        user_count = self.trx_load[self.ct]
        self.ct += 1
        return (10, 10) 

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



columns = ['avg_response_time', 'current_rps', 'num_requests', 'num_failures', 'expected_tps']

result_df = pd.DataFrame(columns=columns)

step = 0
while True:
  deployment, state = get_deployment_info()
  while True:
      running_pods, number_of_all_pods = get_running_pods()
      if len(running_pods) == state[0] and state[0] == number_of_all_pods and running_pods:
        break
      else:
        time.sleep(2)
  env.runner.stats.reset_all()
  time.sleep(1)
  s = dict()
  s['avg_response_time'] = env.runner.stats.total.avg_response_time 
  #s['med_response_time'] = env.runner.stats.total.median_response_time
  s['current_rps'] = env.runner.stats.total.current_rps
  #s['current_fail_per_sec'] = env.runner.stats.total.current_fail_per_sec
  #s['total_rps'] = env.runner.stats.total.total_rps
  s['num_requests'] = env.runner.stats.total.num_requests
  #s['num_none_requests'] = env.runner.stats.total.num_none_requests
  s['num_failures'] = env.runner.stats.total.num_failures
  s['expected_tps'] = env.runner.target_user_count * expected_tps * 9

  #if step % 10 == 0:
  #    for key in s.keys():
  #        print(f'{key:25s} ',end='')
  #    print()
  #for key,value in s.items():
  #    print(f'{value:25.3f} ',end='') 
  #print()
  
  result_df.loc[step,:] = list(s.values())
  print(result_df)
  result_df.to_csv("./load_test.csv", index=False)
  step = step + 1


env.runner.quit()
