import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import io
import multiprocessing as mp
import os.path as path
import signal

import numpy as np
import perwez
import reth_buffer
import yaml
from reth.presets.config import get_trainer

BUFFER_SHARDS = 4
DATA_FETCHER_PROCS = 8
EXITED = False


def data_fetcher_main():
    perwez_client = perwez.connect()
    data_watcher = perwez_client.subscribe("data", False)
    rb_clients = [reth_buffer.connect(f"rb{i}") for i in range(BUFFER_SHARDS)]
    step = 0
    while True:
        step += 1
        rb_client = rb_clients[step % BUFFER_SHARDS]
        s0, a, r, s1, done, loss = data_watcher.get()
        rb_client.append(s0, a, r, s1, done, weights=loss)


def trainer_main(config):
    perwez_client = perwez.connect()
    rb_clients = [reth_buffer.connect(f"rb{i}") for i in range(BUFFER_SHARDS)]
    trainer = get_trainer(config)
    sync_weights_interval = config["common"]["sync_weights_interval"]
    ts = 0
    while True:
        ts += 1
        if trainer.cur_time > 3600 * 40:
            return
        rb_client = rb_clients[ts % BUFFER_SHARDS]
        *data, indices, weights = rb_client.sample()
        loss = trainer.step(data, weights=weights)
        rb_client.update_priorities(np.asarray(indices), np.asarray(loss))

        if ts % sync_weights_interval == 0:
            stream = io.BytesIO()
            trainer.save_weights(stream)
            perwez_client.publish("weights", stream.getbuffer(), ipc=False)


def main(port):
    mp.set_start_method("spawn", force=True)
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    perwez_proc, _ = perwez.start_server(port=port)
    rb_procs = []
    for i in range(BUFFER_SHARDS):
        p, _ = reth_buffer.start_server(
            name=f"rb{i}",
            buffer_capacity=config["replay_buffer"]["capacity"] // BUFFER_SHARDS,
            buffer_alpha=config["replay_buffer"]["alpha"],
            buffer_beta=config["replay_buffer"]["beta"],
            batch_size=config["common"]["batch_size"],
        )
        rb_procs.append(p)
    data_fetcher_procs = []
    for _ in range(DATA_FETCHER_PROCS):
        p = mp.Process(target=data_fetcher_main)
        p.start()
        data_fetcher_procs.append(p)

    trainer_proc = mp.Process(target=trainer_main, args=(config,))

    def graceful_exit(*_):
        global EXITED
        if not EXITED:
            EXITED = True
            print("exiting...")
            for p in data_fetcher_procs:
                p.terminate()
                p.join()
            for p in rb_procs:
                p.terminate()
                p.join()
            trainer_proc.terminate()
            trainer_proc.join()
            perwez_proc.terminate()
            perwez_proc.join()

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    trainer_proc.start()
    trainer_proc.join()
    graceful_exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", required=True)
    args = parser.parse_args()
    main(args.port)
