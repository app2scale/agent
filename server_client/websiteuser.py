
from locust import HttpUser, task, constant, events
from locust.env import Environment

RPS = 100


class WebsiteUser(HttpUser):
    wait_time = constant(1/RPS)
    host = "http://10.27.41.24:30080"


    @task
    def my_task(self):
        response = self.client.get("/tools.descartes.teastore.webui/")


