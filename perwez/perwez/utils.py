import threading
from functools import lru_cache
from urllib.parse import urljoin

import psutil
import requests
import netifaces
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)

DEFAULT_ZMQ_IO_THREADS = 1

HEARTBEAT_INTERVAL = 1200
HEARTBEAT_TIMEOUT = HEARTBEAT_INTERVAL * 1.5


def get_local_ip():
    _, nic = netifaces.gateways()["default"][netifaces.AF_INET]
    addrs = netifaces.ifaddresses(nic)
    return addrs[netifaces.AF_INET][0]["addr"]


class RetryException(Exception):
    pass


def _ping_url_inner(url, pid=None):
    if pid is not None and not psutil.pid_exists(pid):
        raise Exception(f"Server with pid {pid} doesn't exist")

    try:
        res = requests.get(url, timeout=0.5)
        res.raise_for_status()
    except requests.RequestException:
        raise RetryException(f"Failed to connect to {url}") from None

    return res


def ping_url(url, pid=None, timeout=3):
    retryer = Retrying(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(0.2),
        retry=retry_if_exception_type(RetryException),
        reraise=True,
    )
    res = retryer(_ping_url_inner, url, pid)
    return res.json()


GET_CONFIG_LOCK = threading.Lock()


@lru_cache(maxsize=32)
def get_server_config(server_url):
    with GET_CONFIG_LOCK:
        config = ping_url(urljoin(server_url, "/echo"))
        return config
