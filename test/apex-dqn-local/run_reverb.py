import os

os.environ["OMP_NUM_THREADS"] = "1"

import io
import os.path as path

import numpy as np
import reverb
import tensorflow as tf
import torch
import torch.multiprocessing as mp
import yaml

import perwez

from reth.presets.config import get_trainer, get_worker, get_solver
from reth.utils import getLogger

torch.manual_seed(0)

TABLE_NAME = "reverb_test"
PORT = 19133


def worker_main(perwez_url, config, idx):
    reverb_client = reverb.Client(f"localhost:{PORT}")
    reverb_writer = reverb_client.writer(1)
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

    while True:
        # load weights
        if not weight_recv.empty():
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
        for i, _ in enumerate(s0):
            reverb_writer.append([s0[i], a[i], r[i], s1[i], done[i]])
            reverb_writer.create_item(
                table=TABLE_NAME, num_timesteps=1, priority=loss[i]
            )


def _reverb_samples_to_ndarray(samples):
    batch_size = len(samples)
    data0 = samples[0][0].data
    res = []
    for col in data0:
        res.append(np.empty((batch_size, *col.shape), dtype=col.dtype))
    indices = np.empty((batch_size,), dtype=np.uint64)
    weights = np.empty((batch_size,), dtype=np.float64)
    for idx, item in enumerate(samples):
        # data
        for i, data in enumerate(item[0].data):
            res[i][idx] = data
        # indices
        indices[idx] = item[0].info.key
        # weights
        weights[idx] = item[0].info.priority
    return res, indices, weights


def trainer_main_np_client(perwez_url, config):
    weight_send = perwez.SendSocket(perwez_url, "weight", broadcast=True)
    # init reverb
    reverb_client = reverb.Client(f"localhost:{PORT}")

    trainer = get_trainer(config)
    sync_weights_interval = config["common"]["sync_weights_interval"]
    ts = 0
    while True:
        ts += 1

        samples = reverb_client.sample(TABLE_NAME, config["common"]["batch_size"])
        samples = list(samples)
        data, indices, weights = _reverb_samples_to_ndarray(samples)
        weights = (weights / weights.min()) ** (-0.4)

        loss = trainer.step(data, weights=weights)
        reverb_client.mutate_priorities(
            TABLE_NAME, updates=dict(zip(np.asarray(indices), np.asarray(loss)))
        )

        if ts % sync_weights_interval == 0:
            weight_send.send(trainer.save_weights().getbuffer())


def trainer_main_tf_dataset(perwez_url, config):
    weight_send = perwez.SendSocket(perwez_url, "weight", broadcast=True)
    # init reverb
    reverb_client = reverb.Client(f"localhost:{PORT}")
    # reverb dataset
    def _make_dataset(_):
        dataset = reverb.ReplayDataset(
            f"localhost:{PORT}",
            TABLE_NAME,
            max_in_flight_samples_per_worker=config["common"]["batch_size"],
            dtypes=(tf.float32, tf.int64, tf.float32, tf.float32, tf.float32),
            shapes=(
                tf.TensorShape((4, 84, 84)),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape((4, 84, 84)),
                tf.TensorShape([]),
            ),
        )
        dataset = dataset.batch(config["common"]["batch_size"], drop_remainder=True)
        return dataset

    num_parallel_calls = 16
    prefetch_size = 4
    dataset = tf.data.Dataset.range(num_parallel_calls)
    dataset = dataset.interleave(
        map_func=_make_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=False,
    )
    dataset = dataset.prefetch(prefetch_size)
    numpy_iter = dataset.as_numpy_iterator()

    trainer = get_trainer(config)
    sync_weights_interval = config["common"]["sync_weights_interval"]
    ts = 0
    while True:
        ts += 1

        info, data = next(numpy_iter)
        indices = info.key
        weights = info.probability
        weights = (weights / weights.min()) ** (-0.4)

        loss = trainer.step(data, weights=weights)
        reverb_client.mutate_priorities(
            TABLE_NAME, updates=dict(zip(np.asarray(indices), np.asarray(loss)))
        )

        if ts % sync_weights_interval == 0:
            weight_send.send(trainer.save_weights().getbuffer())


def main():
    mp.set_start_method("spawn", force=True)
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # init perwez
    pwz_proc, pwz_config = perwez.start_server()
    # init reverb_server
    reverb.Server(
        tables=[
            reverb.Table(
                name=TABLE_NAME,
                sampler=reverb.selectors.Prioritized(0.6),
                remover=reverb.selectors.Fifo(),
                max_size=config["replay_buffer"]["capacity"],
                rate_limiter=reverb.rate_limiters.MinSize(1000),
            )
        ],
        port=PORT,
    )

    # worker subprocesses
    worker_processes = []
    num_workers = config["common"]["num_workers"]
    for idx in range(num_workers):
        p = mp.Process(
            name=f"apex-worker-{idx}",
            target=worker_main,
            args=(pwz_config["url"], config, idx),
            daemon=True,
        )
        p.start()
        worker_processes.append(p)

    # trainer process should be the main process
    try:
        trainer_main_tf_dataset(pwz_config["url"], config)
    finally:
        print("exiting...")
        for p in worker_processes:
            p.terminate()
            p.join()
        pwz_proc.terminate()
        pwz_proc.join()


if __name__ == "__main__":
    main()
