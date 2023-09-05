import kopf
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import pandas as pd
import random
import time


prom = PrometheusConnect(url ="http://localhost:9090", disable_ssl=True)
metric_dict = {"container_network_receive_bytes_total": "inc_tps",
               "container_network_transmit_packets_total": "out_tps",
               "container_cpu_usage_seconds_total": "cpu_usage",
               "container_memory_working_set_bytes": "memory_usage"}
deployment_name = "teastore-webui"
namespace = "app2scale"
metrics = {}
@kopf.timer('kopfexamples', interval=30)
def deneme(spec, logger, **_): # This function will be called once in 2 minutes and it will return state.
    # deployment = spec['deployment']
    # logger.info(f"Apply logic to autoscale {deployment}")
    deployment = get_deployment_infos()
    running_pods = get_running_pods()
    metric_list = collect_metrics_by_pod_names(running_pods, prom)
    replica, cpu, heap = get_deployment_spec()
    output = f"replica: {replica}, cpu: {cpu}, heap: {heap}, inc_tps: {metric_list['inc_tps']}, out_tps: {metric_list['out_tps']}, cpu_usage: {metric_list['cpu_usage']}, memory_usage: {metric_list['memory_usage']}"
    print(output)
    desired_replicas = 1 #random.randint(0, 2)
    desired_cpu_limit = "500m"
    desired_heap_limit = "-Xmx500M"
    update_deployment_specs(deployment, desired_replicas, desired_cpu_limit, desired_heap_limit)
    print(f"Current replicas: {deployment.status.replicas}, Desired replicas: {desired_replicas}")
    apply_horizontal_scaling(deployment)

def get_running_pods():
    time.sleep(3)
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace, label_selector=f"run={deployment_name}")

    running_pods = []

    for pod in pods.items:
        if pod.status.phase.lower() == "running":
            running_pods.append(pod.metadata.name)

    return running_pods

def collect_metrics_by_pod_names(running_pods, prom):
    metrics = {}
    inc_tps = 0
    out_tps = 0
    cpu_usage = 0
    memory_usage = 0
    for pod in running_pods:
        while True:
            
            temp_inc_tps = prom.custom_query(query=f'sum(irate(container_network_receive_packets_total{{pod="{pod}", namespace="{namespace}"}}[2m]))')
            temp_out_tps = prom.custom_query(query=f'sum(irate(container_network_transmit_packets_total{{pod="{pod}", namespace="{namespace}"}}[2m]))')
            temp_cpu_usage = prom.custom_query(query=f'sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{{pod="{pod}", namespace="{namespace}"}})')
            temp_memory_usage = prom.custom_query(query=f'sum(container_memory_working_set_bytes{{pod="{pod}", namespace="{namespace}"}}) by (name)')

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

def get_deployment_spec():
  deployment = get_deployment_infos()
  replica = deployment.spec.replicas
  cpu = deployment.spec.template.spec.containers[0].resources.limits["cpu"]
  heap = deployment.spec.template.spec.containers[0].env[2].value

  return replica, cpu, heap

def get_deployment_infos():
  config.load_kube_config() 
  v1 = client.AppsV1Api()
  deployment = v1.read_namespaced_deployment(deployment_name, namespace)

  return deployment

def apply_horizontal_scaling(deployment):
  config.load_kube_config() 
  v1 = client.AppsV1Api()
  v1.patch_namespaced_deployment(deployment_name, namespace, deployment)
  

def update_deployment_specs(deployment, desired_replicas, desired_cpu_limit, desired_heap_limit):
  deployment.status.replicas = desired_replicas
  deployment.spec.template.spec.containers[0].resources.limits["cpu"] = desired_cpu_limit
  deployment.spec.template.spec.containers[0].env[2].value = desired_heap_limit




