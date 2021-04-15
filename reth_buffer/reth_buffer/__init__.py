import multiprocessing as mp

import portpicker

from .client import Client, NumpyLoader, TorchCudaLoader
from .sampler import PERSampler
from .server.main_loop import main_loop
from .utils import get_local_ip


def start_server(
    capacity, batch_size, host=None, port=None, samplers=None, cache_policy=None
):
    if host is None:
        host = get_local_ip()
    if port is None:
        port = portpicker.pick_unused_port()
    meta_addr = f"tcp://{host}:{port}"

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=main_loop,
        args=(capacity, batch_size, meta_addr, samplers, cache_policy),
    )
    proc.start()

    return proc, meta_addr


def start_per(
    capacity,
    batch_size,
    alpha=0.6,
    beta=0.4,
    sample_start=1000,
    num_sampler_procs=1,
    host=None,
    port=None,
    cache_policy=None,
):
    samplers = [
        {
            "sampler_cls": PERSampler,
            "num_procs": num_sampler_procs,
            "sample_start": sample_start,
            "kwargs": {"alpha": alpha, "beta": beta},
        }
    ]
    return start_server(capacity, batch_size, host, port, samplers, cache_policy)
