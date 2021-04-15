import multiprocessing as mp

import zmq
from portpicker import pick_unused_port

from .client import Client
from ..server import start


def start_server(
    name="default",
    port=None,
    root_dir=None,
    batch_size=64,
    sample_max_threads=4,
    buffer_capacity=50000,
    buffer_alpha=0.6,
    buffer_beta=0.4,
    buffer_sample_starts=1000,
):
    buffer_config = {
        "capacity": buffer_capacity,
        "alpha": buffer_alpha,
        "beta": buffer_beta,
        "sample_start_threshold": buffer_sample_starts,
    }
    if port is None:
        port = pick_unused_port()
    process = mp.Process(
        target=start,
        args=(
            "zmq",
            name,
            port,
            root_dir,
            batch_size,
            buffer_config,
            sample_max_threads,
        ),
        daemon=True,
    )
    process.start()

    # ensure started
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.IMMEDIATE, True)
    socket.connect(f"tcp://localhost:{port}")
    socket.send_json({"type": "echo"})
    res = socket.recv_json()
    if "error" in res:
        raise Exception(res)
    socket.close()
    return process, port


def connect(name="default"):
    return Client(name)
