import os

os.environ["OMP_NUM_THREADS"] = "1"
import io
import os.path as path

import numpy as np
import perwez
import reth_buffer
import torch
import torch.multiprocessing as mp
import yaml
from reth.presets.config import get_solver, get_trainer, get_worker
from reth.utils import getLogger

torch.manual_seed(0)

BUFFER_SHARDS = 1


def worker_main(config, idx):
    rb_clients = [reth_buffer.connect(f"rb{i}") for i in range(BUFFER_SHARDS)]
    perwez_client = perwez.connect()
    weight_watcher = perwez_client.subscribe("weight")

    batch_size = config["common"]["batch_size"]
    num_workers = config["common"]["num_workers"]
    eps = 0.4 ** (1 + (idx / (num_workers - 1)) * 7)
    solver = get_solver(config, device="cpu")
    log_flag = idx >= num_workers + (-num_workers // 3)  # aligned with ray
    worker = get_worker(
        config,
        exploration=eps,
        solver=solver,
        logger=getLogger(f"worker{idx}") if log_flag else None,
    )

    step = 0
    while True:
        step += 1
        rb_client = rb_clients[step % BUFFER_SHARDS]
        # load weights

        if not weight_watcher.empty():
            worker.load_weights(io.BytesIO(weight_watcher.get()))

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
        rb_client.append(s0, a, r, s1, done, weights=loss)


def trainer_main(config):
    rb_clients = [reth_buffer.connect(f"rb{i}") for i in range(BUFFER_SHARDS)]
    perwez_client = perwez.connect()
    trainer = get_trainer(config)
    sync_weights_interval = config["common"]["sync_weights_interval"]
    ts = 0
    while True:
        ts += 1
        if trainer.cur_time > 3600:
            return
        rb_client = rb_clients[ts % BUFFER_SHARDS]
        *data, indices, weights = rb_client.sample()
        loss = trainer.step(data, weights=weights)
        rb_client.update_priorities(np.asarray(indices), np.asarray(loss))

        if ts % sync_weights_interval == 0:
            perwez_client.publish("weight", trainer.save_weights().getbuffer())


def main():
    mp.set_start_method("spawn", force=True)
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # init perwez
    server_proc, _ = perwez.start_server()
    # init reth_buffer
    for i in range(BUFFER_SHARDS):
        reth_buffer.start_server(
            name=f"rb{i}",
            buffer_capacity=config["replay_buffer"]["capacity"] // BUFFER_SHARDS,
            buffer_alpha=config["replay_buffer"]["alpha"],
            buffer_beta=config["replay_buffer"]["beta"],
            batch_size=config["common"]["batch_size"],
        )

    # worker subprocesses
    worker_processes = []
    num_workers = config["common"]["num_workers"]
    for idx in range(num_workers):
        p = mp.Process(
            name=f"apex-worker-{idx}",
            target=worker_main,
            args=(config, idx),
            daemon=True,
        )
        p.start()
        worker_processes.append(p)

    # trainer process should be the main process
    try:
        trainer_main(config)
    finally:
        print("exiting...")
        for p in worker_processes:
            p.terminate()
            p.join()
        server_proc.terminate()
        server_proc.join()


if __name__ == "__main__":
    main()
