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

EXITED = False


def trainer_main(config, perwez_url, rb_addrs):
    weights_send = perwez.SendSocket(perwez_url, "weights", broadcast=True)
    rb_clients = [reth_buffer.Client(addr) for addr in rb_addrs]
    rb_loaders = [
        reth_buffer.TorchCudaLoader(addr, buffer_size=4, num_procs=2)
        for addr in rb_addrs
    ]
    trainer = get_trainer(config)
    send_weights_interval = config["common"]["send_weights_interval"]
    ts = 0
    while True:
        ts += 1
        if trainer.cur_time > 3600 * 40:
            return
        idx = ts % len(rb_clients)
        data, indices, weights = rb_loaders[idx].sample()
        loss = trainer.step(data, weights=weights)
        rb_clients[idx].update_priorities(np.asarray(indices), np.asarray(loss))

        if ts % send_weights_interval == 0:
            stream = io.BytesIO()
            trainer.save_weights(stream)
            weights_send.send(stream.getbuffer())


def main(perwez_port, rb_ports):
    mp.set_start_method("spawn", force=True)
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    perwez_proc, perwez_config = perwez.start_server(port=perwez_port)
    rb_procs = []
    rb_addrs = []
    for port in rb_ports:
        proc, addr = reth_buffer.start_per(
            capacity=config["replay_buffer"]["capacity"] // len(rb_ports),
            alpha=config["replay_buffer"]["alpha"],
            beta=config["replay_buffer"]["beta"],
            batch_size=config["common"]["batch_size"],
            port=port,
        )
        rb_procs.append(proc)
        rb_addrs.append(addr)

    trainer_proc = mp.Process(
        target=trainer_main, args=(config, perwez_config["url"], rb_addrs)
    )

    def graceful_exit(*_):
        global EXITED
        if not EXITED:
            EXITED = True
            print("exiting...")
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
    parser.add_argument("-p", "--perwez_port", required=True)
    parser.add_argument("-rb", "--rb_ports", required=True, nargs="*")
    args = parser.parse_args()
    main(args.perwez_port, args.rb_ports)
