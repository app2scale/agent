from locust.env import Environment
from locust import HttpUser, task, constant, constant_throughput, events, TaskSet, LoadTestShape
import time
import math

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
# web_ui = env.create_web_ui("127.0.0.1", 8089)
env.runner.start(users,spawn_rate=1) 


# class DoubleWave(LoadTestShape):
    
   
#     min_users = 20
#     peak_one_users = 60
#     peak_two_users = 40
#     time_limit = 600
#     def tick(self):
#         run_time = round(self.get_run_time())

#         if run_time < self.time_limit:
#             user_count = (
#                 (self.peak_one_users - self.min_users)
#                 * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 5) ** 2)
#                 + (self.peak_two_users - self.min_users)
#                 * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 10) ** 2)
#                 + self.min_users
#             )
#             return (round(user_count), round(user_count))
#         else:
#             return None



# env = Environment(user_classes=[TeaStoreLocust])
# env.create_local_runner()
# web_ui = env.create_web_ui("127.0.0.1", 8089)
# env.runner.start(users,spawn_rate=1) 
#env.stop_timeout=10000


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
  #s['exp_tps'] = users * expected_tps 
  #s['wait_time'] = waiting_time
  #elapsed_time = s['avg_response_time'] +  waiting_time*1000
  #s['obs_tps'] = users * 1000 / elapsed_time if elapsed_time > 0 else 0
  s['expected_tps'] = users * expected_tps

  if step % 10 == 0:
      for key in s.keys():
          print(f'{key:25s} ',end='')
      print()
  for key,value in s.items():
    print(f'{value:25.3f} ',end='')
  print()
  
  
  env.runner.stats.reset_all()
  step = step + 1

env.runner.quit()
