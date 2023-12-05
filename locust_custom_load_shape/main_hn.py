from typing import List, Optional, Tuple, Type
from locust.env import Environment
from locust import HttpUser, User, task, constant, constant_throughput, events, TaskSet
from locust.shape import LoadTestShape
from bs4 import BeautifulSoup
import time
import math
import random
import pandas as pd
import numpy as np
import re


"""
wait_time sabit kalsa user değiştirilerek istediğim expected_tps değerini elde edebilir miyim?
"""


expected_tps = 1
class TeaStoreLocust(HttpUser):
    wait_time = constant_throughput(expected_tps)
    host = "http://teastore.local.payten.com.tr/tools.descartes.teastore.webui/"

    @task
    def browse_store(self):
        user_number = 1 + math.random(90)
        categoryid = self.get_random_category_id()
        page = self.get_random_page_number()
        productid = self.get_random_product_id(categoryid, page)

        self.perform_get("login")
        self.perform_post(f"loginAction?username=user{user_number}&password=password")
        self.perform_get(f"category?category={categoryid}&page=1")
        self.perform_get(f"category?category={categoryid}&page={page}")
        self.perform_post(f"cartAction?addToCart=&productid={productid}")
        self.perform_get("profile")
    
    def perform_get(self, endpoint):
        url = f"{self.host}{endpoint}"
        self.client.get(url)

    def perform_post(self, endpoint):
        url = f"{self.host}{endpoint}"
        self.client.post(url)

    def get_random_category_id(self):
        html_content = self.get_html_content() 
        soup = BeautifulSoup(html_content, "html.parser")
        category_links = soup.find_all("a", href=re.compile(r".*category.*category="))
        category_ids = [re.search(r"category=(\d+)", link["href"]).group(1) for link in category_links]
        return random.choice(category_ids)
    
    def get_random_page_number(self):
        html_content = self.get_html_content() 
        soup = BeautifulSoup(html_content, "html.parser")
        page_links = soup.find_all("a", href=re.compile(r".*category.*category=\d+&page=\d+"))
        page_numbers = [int(re.search(r"page=(\d+)", link["href"]).group(1)) for link in page_links]
        return random.choice(page_numbers)
    
    def get_random_product_id(self, category_id, page):
        html_content = self.get_html_content(f"category?category={category_id}&page={page}")
        soup = BeautifulSoup(html_content, "html.parser")
        input_tags = soup.find_all('input')
        product_ids = [tag['value'] for tag in input_tags if tag.get('name') == 'productid']
        return random.choice(product_ids)
    
    def get_html_content(self, endpoint=""):
        url = f"{self.host}{endpoint}" 
        response = self.client.get(url)
        return response.text


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
