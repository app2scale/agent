from typing import List, Optional, Tuple, Type
from locust.env import Environment
from locust import HttpUser, User, task, constant, constant_throughput, events, TaskSet
from locust.shape import LoadTestShape
import time
import math
import random
import pandas as pd
import numpy as np

"""
wait_time sabit kalsa user değiştirilerek istediğim expected_tps değerini elde edebilir miyim?


"""




expected_tps = 1
class TeaStoreLocust(HttpUser):
    wait_time = constant_throughput(expected_tps)
    host = "http://teastore.local.payten.com.tr"
    @task
    def my_task(self):
        response = self.client.get("/tools.descartes.teastore.webui/")


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

    # def tick(self):

    #   if self.ct >= len(self.trx_load):
    #     self.ct = 0

    #   user_count = self.trx_load[self.ct]

    #   return (user_count, user_count)

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
