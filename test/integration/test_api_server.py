import time, os

import pytest
import docker
import requests

host, port = "127.0.0.1", 8000


@pytest.fixture(scope="module")
def api_server():
    aws_region = os.environ["AWS_DEFAULT_REGION"]
    aws_account_id = os.environ["AWS_ACCOUNT_ID"]
    repository = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/emo-classifier"

    version = os.environ["VERSION"]
    image_uri = f"{repository}:{version}"

    docker_client = docker.from_env()
    api_container: docker.models.containers.Container = docker_client.containers.run(
        image_uri, ports={port: port}, detach=True
    )

    successfully_connected = False
    for _ in range(6):
        time.sleep(10)
        try:
            r = requests.get(f"http://{host}:{port}/")
            if r.status_code == 200:
                successfully_connected = True
                break
        except Exception:
            pass

    if successfully_connected:
        yield api_container

    api_container.stop()
    api_container.remove()

    if not successfully_connected:
        raise RuntimeError("Docker API server failed to start.")


def test_send_a_request(api_server):
    case1 = {"id": "dummy_id", "text": "pathlib-like API for cloud storage. Looks very nice 👍‍"}
    expected1 = {"id": "dummy_id", "labels": ["admiration"]}

    r = requests.post(f"http://{host}:{port}/prediction", json=case1)
    assert r.status_code == 200

    actual1 = r.json()
    assert expected1 == actual1
