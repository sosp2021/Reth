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
from reth_buffer import TorchCudaLoader

torch.manual_seed(0)


def worker_main(config, perwez_url, rb_addr, idx):
    rb_client = reth_buffer.Client(rb_addr)
    weight_recv = perwez.RecvSocket(perwez_url, "weight", broadcast=True)

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
    recv_weights_interval = config["common"]["recv_weights_interval"]

    step = 0
    prev_recv = 0
    try:
        while True:
            step += 1

            # load weights
            interval_flag = (step - prev_recv) * batch_size >= recv_weights_interval
            if interval_flag and not weight_recv.empty():
                worker.load_weights(io.BytesIO(weight_recv.recv()))

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
            rb_client.append([s0, a, r, s1, done], loss)
    except KeyboardInterrupt:
        # silent exit
        pass
        # print(f'worker {idx} terminating...')


def trainer_main(config, perwez_url, rb_addr):
    weight_send = perwez.SendSocket(perwez_url, "weight", broadcast=True)
    rb_client = reth_buffer.Client(rb_addr)
    loader = TorchCudaLoader(rb_addr)
    trainer = get_trainer(config)
    send_weights_interval = config["common"]["send_weights_interval"]

    try:
        ts = 0
        for data, indices, weights in loader:
            ts += 1
            if trainer.cur_time > 3600 * 10:
                return
            loss = trainer.step(data, weights=weights)
            rb_client.update_priorities(np.asarray(indices), np.asarray(loss))

            if ts % send_weights_interval == 0:
                weight_send.send(trainer.save_weights().getbuffer())
    except KeyboardInterrupt:
        # silent exit
        pass
        # print(f'trainer terminating...')


def main():
    mp.set_start_method("spawn", force=True)
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # init perwez
    pwz_proc, pwz_config = perwez.start_server()
    # init reth_buffer
    buffer_proc, rb_addr = reth_buffer.start_per(
        num_sampler_procs=1,
        capacity=config["replay_buffer"]["capacity"],
        batch_size=config["common"]["batch_size"],
        alpha=config["replay_buffer"]["alpha"],
        beta=config["replay_buffer"]["beta"],
    )

    # worker subprocesses
    worker_processes = []
    num_workers = config["common"]["num_workers"]
    for idx in range(num_workers):
        p = mp.Process(
            name=f"apex-worker-{idx}",
            target=worker_main,
            args=(config, pwz_config["url"], rb_addr, idx),
        )
        p.start()
        worker_processes.append(p)

    # trainer process should be the main process
    try:
        trainer_main(config, pwz_config["url"], rb_addr)
    finally:
        print("exiting...")
        for p in worker_processes:
            p.terminate()
            p.join()
        pwz_proc.terminate()
        pwz_proc.join()
        buffer_proc.terminate()
        buffer_proc.join()


if __name__ == "__main__":
    main()
