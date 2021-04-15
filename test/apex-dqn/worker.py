import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import io
import multiprocessing as mp
import os.path as path
import signal

import numpy as np
import perwez
import yaml
import reth_buffer
from reth.buffer import NumpyBuffer
from reth.presets.config import get_solver, get_worker
from reth.utils import getLogger, NStepAdder

EXITED = False


def worker_main(config, idx, size, perwez_url, rb_addr):
    rb_client = reth_buffer.Client(rb_addr)
    weight_recv = perwez.RecvSocket(perwez_url, "local-weights", broadcast=True)

    batch_size = config["common"]["rollout_batch_size"]
    eps = 0.4 ** (1 + (idx / (size - 1)) * 7)
    solver = get_solver(config, device="cpu")
    worker = get_worker(
        config, exploration=eps, solver=solver, logger=getLogger(f"worker{idx}")
    )

    recv_weights_interval = config["common"]["recv_weights_interval"]
    prev_load = 0

    adder = NStepAdder(config["solver"]["gamma"], config["solver"]["n_step"])
    buffer = NumpyBuffer(batch_size, circular=False)
    while True:
        # load weights
        if (
            worker.cur_step - prev_load
        ) > recv_weights_interval and not weight_recv.empty():
            worker.load_weights(io.BytesIO(weight_recv.recv()))
            prev_load = worker.cur_step

        # step
        s0, a, r, s1, done = worker.step()
        s0 = np.asarray(s0, dtype="f4")
        a = np.asarray(a, dtype="i8")
        r = np.asarray(r, dtype="f4")
        s1 = np.asarray(s1, dtype="f4")
        done = np.asarray(done, dtype="f4")
        # adder
        row = adder.push(s0, a, r, s1, done)
        if row is None:
            continue
        buffer.append(row)
        if buffer.size == buffer.capacity:
            loss = worker.solver.calc_loss(buffer.data)
            loss = np.asarray(loss, dtype="f4")
            rb_client.append(buffer.data, loss, compress=True)
            buffer.clear()


def weights_proxy(perwez_url):
    weight_recv = perwez.RecvSocket(perwez_url, "weights", broadcast=True)
    weight_send = perwez.SendSocket(perwez_url, "local-weights", broadcast=True)
    while True:
        res = weight_recv.recv()
        weight_send.send(res)


def main(rank, size, perwez_url, rb_addrs):
    mp.set_start_method("spawn", force=True)
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    weights_proxy_proc = mp.Process(
        target=weights_proxy, args=(perwez_url,), daemon=True
    )
    weights_proxy_proc.start()

    worker_procs = []
    num_workers = config["common"]["num_workers"]
    offset = rank * num_workers
    for idx in range(num_workers):
        p = mp.Process(
            target=worker_main,
            args=(
                config,
                offset + idx,
                size * num_workers,
                perwez_url,
                rb_addrs[idx % len(rb_addrs)],
            ),
            daemon=True,
        )
        p.start()
        worker_procs.append(p)

    def graceful_exit(*_):
        global EXITED
        if not EXITED:
            EXITED = True
            print("exiting...")
            for p in worker_procs:
                p.terminate()
                p.join()

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    for p in worker_procs:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rank", type=int, required=True)
    parser.add_argument("-s", "--size", type=int, required=True)
    parser.add_argument("-p", "--perwez_url", required=True)
    parser.add_argument("-rb", "--rb_addrs", required=True, nargs="*")
    args = parser.parse_args()
    main(args.rank, args.size, args.perwez_url, args.rb_addrs)
