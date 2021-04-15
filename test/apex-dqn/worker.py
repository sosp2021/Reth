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
from reth.presets.config import get_solver, get_worker
from reth.utils import getLogger

EXITED = False


def worker_main(config, idx, size):
    perwez_client = perwez.connect()
    weights_watcher = perwez_client.subscribe("local-weights")

    batch_size = config["common"]["rollout_batch_size"]
    eps = 0.4 ** (1 + (idx / (size - 1)) * 7)
    solver = get_solver(config, device="cpu")
    worker = get_worker(
        config, exploration=eps, solver=solver, logger=getLogger(f"worker{idx}")
    )

    while True:
        # load weights
        if not weights_watcher.empty():
            res = weights_watcher.get()
            worker.load_weights(io.BytesIO(res))

        # step
        data = worker.step_batch(batch_size)
        loss = worker.solver.calc_loss(data)
        # format
        s0, a, r, s1, done = data
        s0 = np.asarray(s0, dtype="f4")
        a = np.asarray(a, dtype="i8")
        r = np.asarray(r, dtype="f4")
        s1 = np.asarray(s1, dtype="f4")
        done = np.asarray(done, dtype="f4")
        loss = np.asarray(loss, dtype="f4")
        # upload
        perwez_client.push(
            "data", [s0, a, r, s1, done, loss], ipc=False, compression="lz4"
        )


def weights_proxy():
    c = perwez.connect()
    w = c.subscribe("weights")
    while True:
        res = w.get()
        c.publish("local-weights", res)


def main(rank, size, bootstrap_url):
    mp.set_start_method("spawn", force=True)
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    perwez_proc, _ = perwez.start_server(parent_url=bootstrap_url, bootstrap_timeout=30)

    weights_proxy_proc = mp.Process(target=weights_proxy, daemon=True)
    weights_proxy_proc.start()

    worker_procs = []
    num_workers = config["common"]["num_workers"]
    offset = rank * num_workers
    for idx in range(num_workers):
        p = mp.Process(
            target=worker_main,
            args=(config, offset + idx, size * num_workers),
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
            perwez_proc.terminate()
            perwez_proc.join()

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    for p in worker_procs:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rank", type=int, required=True)
    parser.add_argument("-s", "--size", type=int, required=True)
    parser.add_argument("-b", "--bootstrap_url", required=True)
    args = parser.parse_args()
    main(args.rank, args.size, args.bootstrap_url)
