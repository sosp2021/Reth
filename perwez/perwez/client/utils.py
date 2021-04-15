from urllib.parse import urljoin

import requests
import zmq
from tenacity import Retrying, stop_after_delay, wait_fixed

ZMQ_REVERSE = {
    zmq.SUB: zmq.PUB,
    zmq.PUB: zmq.SUB,
    zmq.PULL: zmq.PUSH,
    zmq.PUSH: zmq.PULL,
}

ZMQ_IN = {zmq.PUB: False, zmq.PUSH: False, zmq.SUB: True, zmq.PULL: True}


def _send_heartbeat_inner(server_url, payload):
    res = requests.post(urljoin(server_url, "/endpoints"), json=payload)
    res.raise_for_status()


def send_heartbeat(server_url, payload, timeout=3):
    retryer = Retrying(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(1),
        reraise=True,
    )
    retryer(_send_heartbeat_inner, server_url, payload)
