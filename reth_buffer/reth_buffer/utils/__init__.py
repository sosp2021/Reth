import os
from functools import lru_cache

import lmdb
import netifaces
import portpicker
import zmq

from .schedule import Schedule
from .sumtree import NumbaSumTree

ZMQ_DEFAULT_HWM = 10

DEFAULT_REQ_TOPIC = "reth_buffer_req"
DEFAULT_SAMPLE_TOPIC = "reth_buffer_sample"


def get_tcp_addr(host="0.0.0.0"):
    port = portpicker.pick_unused_port()
    return f"tcp://{host}:{port}"


def get_ipc_addr(folder, name):
    if os.name == "nt":
        return get_tcp_addr("127.0.0.1")
    return f"ipc://{folder}/{name}.sock"


def get_local_ip():
    _, nic = netifaces.gateways()["default"][netifaces.AF_INET]
    addrs = netifaces.ifaddresses(nic)
    return addrs[netifaces.AF_INET][0]["addr"]


def init_lmdb(path, map_size=0):
    env = lmdb.open(
        path,
        map_size=map_size,
        metasync=False,
        sync=False,
        readahead=False,
        writemap=True,
        meminit=False,
        lock=True,
    )

    return env


@lru_cache(maxsize=32)
def get_meta(meta_addr):
    ctx = zmq.Context.instance()
    meta_sock = ctx.socket(zmq.REQ)
    meta_sock.setsockopt(zmq.IMMEDIATE, 1)
    meta_sock.connect(meta_addr)
    meta_sock.send(b"")
    res = meta_sock.recv_json()
    meta_sock.close(linger=0)
    return res
