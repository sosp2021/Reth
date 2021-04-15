import os

import psutil
import requests
import netifaces
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_delay,
    wait_fixed,
)

DEFAULT_HWM = 5
DEFAULT_ZMQ_IO_THREADS = 4
ROOT_DIR_PREFIX = os.path.expanduser("~/.perwez_")
INPROC_INFO_ADDR = "inproc://perwez-endpoints"
INPROC_WATCHER_PREFIX = "inproc://perwez-watcher-"


def get_root_dir(name):
    return ROOT_DIR_PREFIX + name


def get_config_path(name):
    return os.path.join(get_root_dir(name), "config.json")


def get_lock_path(name):
    return get_root_dir(name) + "_lock"


def get_local_ip():
    _, nic = netifaces.gateways()["default"][netifaces.AF_INET]
    addrs = netifaces.ifaddresses(nic)
    return addrs[netifaces.AF_INET][0]["addr"]


class RetryException(Exception):
    pass


def _ping_server_inner(url, pid=None):
    if pid is not None and not psutil.pid_exists(pid):
        raise Exception(f"Server with pid {pid} doesn't exist")

    try:
        res = requests.get(url, timeout=0.5)
        res.raise_for_status()
    except requests.RequestException:
        raise RetryException(f"Failed to connect to {url}") from None

    return res


def ping_server(url, pid=None, timeout=3):
    retryer = Retrying(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(0.2),
        retry=retry_if_exception_type(RetryException),
        reraise=True,
    )
    retryer(_ping_server_inner, url, pid)
