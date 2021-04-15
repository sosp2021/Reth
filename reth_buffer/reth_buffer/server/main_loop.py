import copy
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
import threading
import urllib.parse

import zmq
from loguru import logger

from .sampler_loop import sampler_loop
from ..cache_policy import FIFOPolicy
from ..sampler import PERSampler
from ..utils import ZMQ_DEFAULT_HWM, get_ipc_addr, get_local_ip, get_tcp_addr, init_lmdb
from ..utils.pack import deserialize, serialize


def append_loop(in_addr, sampler_addr, lmdb_path, cache_policy, capacity):
    ctx = zmq.Context.instance(4)
    ctx.setsockopt(zmq.LINGER, 0)
    in_sock = ctx.socket(zmq.PULL)
    in_sock.set_hwm(ZMQ_DEFAULT_HWM)
    in_sock.bind(in_addr)
    sampler_sock = ctx.socket(zmq.PUSH)
    sampler_sock.set_hwm(ZMQ_DEFAULT_HWM)
    sampler_sock.connect(sampler_addr)

    lmdb_env = None
    step_counter = 0
    busy_counter = 0
    try:
        while True:
            step_counter += 1
            if in_sock.poll(timeout=0, flags=zmq.POLLIN):
                busy_counter += 1
            if step_counter >= 100 and busy_counter / step_counter > 0.7:
                logger.warning(f"append loop ({busy_counter}/{step_counter})")
                busy_counter = 0
                step_counter = 0

            req_frame = in_sock.recv(copy=False)
            data, weights = deserialize(req_frame.buffer)
            indices = cache_policy.get_indices(len(weights))

            if lmdb_env is None:
                map_size = 1.5 * capacity * memoryview(data[0]).nbytes
                lmdb_env = init_lmdb(lmdb_path, map_size)
                logger.info(
                    f"lmdb created, map size: {map_size / 1024 / 1024 / 1024:.2f} GB"
                )

            with lmdb_env.begin(write=True) as txn:
                for i, idx in enumerate(indices):
                    txn.put(str(idx).encode(), data[i])
            sampler_sock.send(serialize([indices, weights, False]))
    except KeyboardInterrupt:
        # silent exit
        pass


def meta_loop(meta_addr, config):
    config = copy.deepcopy(config)
    # process config
    for info in config["sampler_info"].values():
        info["sampler_cls"] = info["sampler_cls"].__name__
    ctx = zmq.Context.instance()
    ctx.setsockopt(zmq.LINGER, 0)
    sock = ctx.socket(zmq.REP)
    sock.set_hwm(ZMQ_DEFAULT_HWM)
    sock.bind(meta_addr)
    while True:
        sock.recv()
        sock.send_json(config)


def update_proxy_loop(in_addr, out_addr):
    ctx = zmq.Context.instance()
    ctx.setsockopt(zmq.LINGER, 0)
    in_sock = ctx.socket(zmq.PULL)
    in_sock.set_hwm(ZMQ_DEFAULT_HWM)
    in_sock.bind(in_addr)
    out_sock = ctx.socket(zmq.PUB)
    out_sock.bind(out_addr)

    zmq.proxy(in_sock, out_sock)


class SignalHandler:
    def __init__(self, proc_list, temp_folder):
        self.proc_list = proc_list
        self.temp_folder = temp_folder
        self.exit_flag = False

    def on_signal(self, signum, *_):
        if self.exit_flag:
            return
        self.exit_flag = True
        logger.info(f"signal {signum} recieved, terminating...")
        for proc in self.proc_list:
            proc.terminate()
            proc.join()
        shutil.rmtree(self.temp_folder)
        logger.info("replay buffer terminated")
        sys.exit(signum)


def main_loop(capacity, batch_size, meta_addr=None, samplers=None, cache_policy=None):
    mp_ctx = mp.get_context("spawn")

    # default args
    if cache_policy is None:
        cache_policy = FIFOPolicy(capacity)
    if samplers is None:
        samplers = [
            {
                "sampler_cls": PERSampler,
                "num_procs": 1,
                "sample_start": 1000,
            }
        ]
    if meta_addr is None:
        host = get_local_ip()
    else:
        host = urllib.parse.urlparse(meta_addr).hostname
    if os.name == "nt":
        temp_dir = None
    else:
        temp_dir = "/dev/shm"
    temp_folder = tempfile.mkdtemp(prefix="reth_buffer", dir=temp_dir)
    # normalize sampler param
    sampler_info = {}
    for item in samplers:
        topic = item.get("topic", "default")
        assert topic not in sampler_info

        num_procs = item.get("num_procs", 1)

        kwargs = item.get("kwargs", {})
        kwargs["capacity"] = capacity

        sample_start = item.get("sample_start", 1000)
        sample_start = max(sample_start, batch_size)

        info = {
            "topic": topic,
            "sampler_cls": item["sampler_cls"],
            "num_procs": num_procs,
            "kwargs": kwargs,
            "addrs": [
                get_ipc_addr(temp_folder, f"sampler-{topic}-{idx}")
                for idx in range(num_procs)
            ],
            "sample_start": sample_start,
            "batch_size": batch_size,
        }
        sampler_info[topic] = info
    # generate addrs
    if meta_addr is None:
        meta_addr = get_tcp_addr(host)
    append_addr = get_tcp_addr(host)
    update_addr = get_ipc_addr(temp_folder, "update")
    _update_proxy_out_addr = get_ipc_addr(temp_folder, "update-proxy-out")
    lmdb_path = os.path.join(temp_folder, "lmdb")
    os.makedirs(lmdb_path, exist_ok=True)

    config = {
        "capacity": capacity,
        "batch_size": batch_size,
        "lmdb_path": lmdb_path,
        "meta_addr": meta_addr,
        "append_addr": append_addr,
        "update_addr": update_addr,
        "sampler_info": sampler_info,
    }

    # start processes
    created_procs = []
    signal_handler = SignalHandler(created_procs, temp_folder)
    signal.signal(signal.SIGINT, signal_handler.on_signal)
    signal.signal(signal.SIGTERM, signal_handler.on_signal)

    # main proc threads
    meta_thread = threading.Thread(
        target=meta_loop, args=(meta_addr, config), daemon=True
    )
    meta_thread.start()
    update_proxy_thread = threading.Thread(
        target=update_proxy_loop,
        args=(update_addr, _update_proxy_out_addr),
        daemon=True,
    )
    update_proxy_thread.start()

    # append
    append_proc = mp_ctx.Process(
        target=append_loop,
        args=(append_addr, update_addr, lmdb_path, cache_policy, capacity),
    )
    append_proc.start()
    created_procs.append(append_proc)

    # sampler
    for topic, info in sampler_info.items():
        for idx in range(info["num_procs"]):
            sampler_proc = mp_ctx.Process(
                target=sampler_loop,
                args=(_update_proxy_out_addr, info, idx),
            )
            sampler_proc.start()
            created_procs.append(sampler_proc)

    meta_thread.join()
    update_proxy_thread.join()
