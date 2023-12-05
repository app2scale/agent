import logging
from random import randint, choice

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
    host = "http://teastore.local.payten.com.tr/tools.descartes.teastore.webui/"
    
    @task
    def load(self) -> None:
        """
        Simulates user behaviour.
        :return: None
        """
        logging.info("Starting user.")
        self.visit_home()
        self.login()
        self.browse()
        # 50/50 chance to buy
        choice_buy = choice([True, False])
        if choice_buy:
            self.buy()
        self.visit_profile()
        self.logout()
        logging.info("Completed user.")

    def visit_home(self) -> None:
        """
        Visits the landing page.
        :return: None
        """
        # load landing page
        res = self.client.get('/')
        if res.ok:
            logging.info("Loaded landing page.")
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
            logging.info("Loaded login page.")
        else:
            logging.error(f"Could not load login page: {res.status_code}")
        # login random user
        user = randint(1, 99)
        login_request = self.client.post("/loginAction", params={"username": user, "password": "password"})
        if login_request.ok:
            logging.info(f"Login with username: {user}")
        else:
            logging.error(
                f"Could not login with username: {user} - status: {login_request.status_code}")

    def browse(self) -> None:
        """
        Simulates random browsing behaviour.
        :return: None
        """
        # execute browsing action randomly up to 5 times
        for i in range(1, randint(2, 5)):
            # browses random category and page
            category_id = randint(2, 6)
            page = randint(1, 5)
            category_request = self.client.get("/category", params={"page": page, "category": category_id})
            if category_request.ok:
                logging.info(f"Visited category {category_id} on page 1")
                # browses random product
                product_id = randint(7, 506)
                product_request = self.client.get("/product", params={"id": product_id})
                if product_request.ok:
                    logging.info(f"Visited product with id {product_id}.")
                    cart_request = self.client.post("/cartAction", params={"addToCart": "", "productid": product_id})
                    if cart_request.ok:
                        logging.info(f"Added product {product_id} to cart.")
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
            logging.info(f"Bought products.")
        else:
            logging.error("Could not buy products.")

    def visit_profile(self) -> None:
        """
        Visits user profile.
        :return: None
        """
        profile_request = self.client.get("/profile")
        if profile_request.ok:
            logging.info("Visited profile page.")
        else:
            logging.error("Could not visit profile page.")

    def logout(self) -> None:
        """
        User logout.
        :return: None
        """
        logout_request = self.client.post("/loginAction", params={"logout": ""})
        if logout_request.ok:
            logging.info("Successful logout.")
        else:
            logging.error(f"Could not log out - status: {logout_request.status_code}")



class CustomLoad(LoadTestShape):
    trx_load_data = pd.read_csv("/Users/hasan.nayir/Projects/Payten/app2scale_reinforcement_learning/locust_custom_load_shape/transactions.csv")
    trx_load = trx_load_data["transactions"].values.tolist()
    trx_load = (trx_load/np.max(trx_load)*100).astype(int)
    ct = 0
    
    def tick(self):
        if self.ct >= len(self.trx_load):
            self.ct = 0
        user_count = self.trx_load[self.ct]
        self.ct += 1
        return (user_count, user_count) 

load = CustomLoad()
env = Environment(user_classes=[TeaStoreLocust], shape_class=load)
env.create_local_runner()
web_ui = env.create_web_ui("127.0.0.1", 8089)
env.runner.start_shape() 
env.stop_timeout=10000




step = 0
while True:
  time.sleep(1)
  s = dict()
  s['avg_response_time'] = env.runner.stats.total.avg_response_time 
  s['med_response_time'] = env.runner.stats.total.median_response_time
  s['current_rps'] = env.runner.stats.total.current_rps
  s['current_fail_per_sec'] = env.runner.stats.total.current_fail_per_sec
  s['total_rps'] = env.runner.stats.total.total_rps
  s['num_requests'] = env.runner.stats.total.num_requests
  s['num_none_requests'] = env.runner.stats.total.num_none_requests
  s['num_failures'] = env.runner.stats.total.num_failures
  s['current_fail_per_sec'] = env.runner.stats.total.current_fail_per_sec
  s['expected_tps'] = env.runner.target_user_count * expected_tps

  if step % 10 == 0:
      for key in s.keys():
          print(f'{key:25s} ',end='')
      print()
  for key,value in s.items():
    print(f'{value:25.3f} ',end='')
  print()
  
  
  # env.runner.stats.reset_all()
  step = step + 1


env.runner.quit()