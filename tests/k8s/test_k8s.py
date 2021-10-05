from http import HTTPStatus
from pytest_kind import cluster

# kind version has to be bumped to v0.11.1 since pytest-kind is just using v0.10.0 which does not work on ubuntu in ci
# TODO don't use pytest-kind anymore
cluster.KIND_VERSION = 'v0.11.1'
import pytest
import requests
import multiprocessing
import time

from jina import Flow
from jina.peapods.pods.k8slib.kubernetes_tools import (
    get_port_forward_contextmanager,
    K8sClients,
)


def run_test(flow, logger, app_names, expected_replicas, endpoint, port_expose):
    event = multiprocessing.Event()
    watch_process = multiprocessing.Process(
        target=watch_pods,
        kwargs={
            'flow_name': flow.args.name,
            'app_names': app_names,
            'expected_replicas': expected_replicas,
            'event': event,
        },
        daemon=True,
    )
    watch_process.start()
    with flow:
        resp = send_dummy_request(endpoint, flow, logger, port_expose=port_expose)
        event.set()
    watch_process.join()
    watch_process.terminate()
    return resp


def watch_pods(flow_name, app_names, expected_replicas, event):
    # using Pykube to query pods
    from jina.logging.logger import JinaLogger

    logger = JinaLogger(f'test_watch_cluster')
    time.sleep(10)
    k8s_client = K8sClients()
    while not event.is_set():
        with logger:
            logger.info(
                f' Wants to watch pods for the following deployments {app_names}'
            )
            try:
                pods_by_app = dict()
                for app, expected_repl in zip(app_names, expected_replicas):
                    api_response = k8s_client.apps_v1.read_namespaced_deployment(
                        name=app, namespace=flow_name
                    )
                    assert api_response.status.replicas == expected_repl
                    if (
                        api_response.status.ready_replicas is not None
                        and api_response.status.ready_replicas == expected_repl
                    ):
                        continue
                    else:
                        logger.info(f' {app} is not ready')
                        pods = k8s_client.core_v1.list_namespaced_pod(
                            namespace=flow_name, label_selector=f"app={app}"
                        )
                        pod_names = [item.metadata.name for item in pods.items]
                        pods_by_app[app] = pod_names
                for app, pod_names in pods_by_app.items():
                    for pod_name in pod_names:
                        logs = k8s_client.core_v1.read_namespaced_pod_log(
                            name=pod_name, namespace=flow_name
                        )
                        status = k8s_client.core_v1.read_namespaced_pod_status(
                            name=pod_name, namespace=flow_name, pretty=True
                        )
                        all = k8s_client.core_v1.read_namespaced_pod(
                            name=pod_name, namespace=flow_name, pretty=True
                        )
                        logger.info(f' \n\n\n =======================\n\n\n')
                        logger.info(f' app {app}, pod {pod_name}: logs:\n')
                        logger.info(f' status {status} \n')
                        logger.info(f' logs {logs} \n')
                        logger.info(f' all {all} \n')
            except Exception:
                continue
            finally:
                time.sleep(0.5)


def send_dummy_request(endpoint, flow, logger, port_expose):
    logger.debug(f'Starting port-forwarding to gateway service...')
    with get_port_forward_contextmanager(
        namespace=flow.args.name, port_expose=port_expose
    ):
        logger.debug(f'Port-forward running...')
        resp = requests.post(
            f'http://localhost:{port_expose}/{endpoint}',
            json={'data': [{} for _ in range(10)]},
        )
    return resp


@pytest.fixture()
def k8s_flow_with_init_container(
    test_executor_image: str, executor_merger_image: str, dummy_dumper_image: str
) -> Flow:
    flow = Flow(
        name='test-flow-with-init-container',
        port_expose=9090,
        infrastructure='K8S',
        protocol='http',
        timeout_ready=120000,
    ).add(
        name='test_executor',
        uses=test_executor_image,
        k8s_init_container_command=["python", "dump.py", "/shared/test_file.txt"],
        k8s_uses_init=dummy_dumper_image,
        k8s_mount_path='/shared',
        timeout_ready=120000,
    )
    return flow


@pytest.fixture()
def k8s_flow_with_sharding(
    test_executor_image: str, executor_merger_image: str, dummy_dumper_image: str
) -> Flow:
    flow = Flow(
        name='test-flow-with-sharding',
        port_expose=9090,
        infrastructure='K8S',
        protocol='http',
        timeout_ready=120000,
    ).add(
        name='test_executor',
        shards=2,
        replicas=2,
        uses=test_executor_image,
        uses_after=executor_merger_image,
        timeout_ready=360000,
    )
    return flow


@pytest.mark.timeout(3600)
@pytest.mark.parametrize('k8s_connection_pool', [True, False])
def test_flow_with_needs(
    k8s_cluster,
    test_executor_image: str,
    executor_merger_image: str,
    load_images_in_kind,
    set_test_pip_version,
    logger,
    k8s_connection_pool: bool,
):
    name = 'test-flow-with-needs'
    if k8s_connection_pool:
        name += '-pool'
    flow = (
        Flow(
            name=name,
            port_expose=9090,
            infrastructure='K8S',
            protocol='http',
            timeout_ready=120000,
            k8s_connection_pool=k8s_connection_pool,
        )
        .add(
            name='segmenter',
            uses=test_executor_image,
            timeout_ready=120000,
        )
        .add(
            name='textencoder',
            uses=test_executor_image,
            needs='segmenter',
            timeout_ready=120000,
        )
        .add(
            name='imageencoder',
            uses=test_executor_image,
            needs='segmenter',
            timeout_ready=120000,
        )
        .add(
            name='merger',
            uses=executor_merger_image,
            timeout_ready=120000,
            needs=['imageencoder', 'textencoder'],
        )
    )
    resp = run_test(
        flow,
        logger,
        app_names=['gateway', 'segmenter', 'imageencoder', 'textencoder'],
        expected_replicas=[1, 1, 1, 1],
        endpoint='index',
        port_expose=9090,
    )

    expected_traversed_executors = {
        'segmenter',
        'imageencoder',
        'textencoder',
    }

    assert resp.status_code == HTTPStatus.OK
    docs = resp.json()['data']['docs']
    assert len(docs) == 10
    for doc in docs:
        assert set(doc['tags']['traversed-executors']) == expected_traversed_executors


@pytest.mark.timeout(3600)
def test_flow_with_init(
    k8s_cluster,
    k8s_flow_with_init_container: Flow,
    load_images_in_kind,
    set_test_pip_version,
    logger,
):
    resp = run_test(
        k8s_flow_with_init_container,
        logger,
        app_names=['test-executor', 'gateway'],
        expected_replicas=[1, 1],
        endpoint='search',
        port_expose=9090,
    )

    assert resp.status_code == HTTPStatus.OK
    docs = resp.json()['data']['docs']
    assert len(docs) == 10
    for doc in docs:
        assert doc['tags']['file'] == ['1\n', '2\n', '3']


@pytest.mark.timeout(3600)
def test_flow_with_sharding(
    k8s_cluster,
    k8s_flow_with_sharding: Flow,
    load_images_in_kind,
    set_test_pip_version,
    logger,
):
    resp = run_test(
        k8s_flow_with_sharding,
        logger,
        app_names=[
            'gateway',
            'test-executor-head',
            'test-executor-0',
            'test-executor-1',
            'test-executor-tail',
        ],
        expected_replicas=[1, 1, 2, 2, 1],
        endpoint='index',
        port_expose=9090,
    )

    expected_traversed_executors = {
        'test_executor',
    }

    assert resp.status_code == HTTPStatus.OK
    docs = resp.json()['data']['docs']
    assert len(docs) == 10
    for doc in docs:
        assert set(doc['tags']['traversed-executors']) == expected_traversed_executors
