import multiprocessing as mp
import time

import numpy as np
import torch

import reverb
import tensorflow as tf

import reth_buffer
from reth.algorithm.prioritized_buffer import PRB2

BATCH_SIZE = 64
CAPACITY = 10000
TEST_CNT = 1000

if __name__ == "__main__":
    print("initializing...")
    mp.set_start_method("spawn", force=True)
    data = []
    data.extend(np.random.rand(CAPACITY, 1, 84, 84) * 20 for _ in range(2))
    data.extend(np.random.rand(CAPACITY) * 20 for _ in range(2))

    print("TEST TORCH_BUFFER")
    prb2 = PRB2(CAPACITY, BATCH_SIZE)
    for i in range(CAPACITY):
        prb2.append(*[col[i] for col in data], 1)
    # init
    *items, indices, weights = prb2.sample()
    items = [x.cuda() for x in items]
    print("ready")
    t0 = time.perf_counter()
    for _ in range(TEST_CNT):
        *items, indices, weights = prb2.sample()
        items = [x.cuda() for x in items]
    t1 = time.perf_counter()
    print(TEST_CNT, t1 - t0)
    print("TEST RETH_BUFFER")
    print("reth_zmq")
    buffer_process = reth_buffer.start_server(
        batch_size=BATCH_SIZE, buffer_capacity=CAPACITY, server_type="zmq"
    )
    client = reth_buffer.connect()
    fut = client.append(*data)
    fut.result()
    # init
    *items, indices, weights = client.sample(silent=True)
    cdata = [torch.tensor(x, device="cuda") for x in items]
    print("ready")
    t0 = time.perf_counter()
    for _ in range(TEST_CNT):
        *items, indices, weights = client.sample(silent=True)
        cdata = [torch.tensor(x, device="cuda") for x in items]
    t1 = time.perf_counter()
    print(TEST_CNT, t1 - t0)
    buffer_process.terminate()

    print("TEST REVERB")
    print("initializing...")
    reverb_server = reverb.Server(
        tables=[
            reverb.Table(
                name="req",
                sampler=reverb.selectors.Prioritized(0.6),
                remover=reverb.selectors.Fifo(),
                max_size=CAPACITY,
                rate_limiter=reverb.rate_limiters.MinSize(100),
            )
        ],
        port=15867,
    )
    client = reverb_server.in_process_client()
    for i in range(CAPACITY):
        client.insert([col[i] for col in data], {"req": np.random.rand()})
    dataset = reverb.ReplayDataset(
        server_address="localhost:15867",
        table="req",
        dtypes=(tf.float64, tf.float64, tf.float64, tf.float64),
        shapes=(
            tf.TensorShape([1, 84, 84]),
            tf.TensorShape([1, 84, 84]),
            tf.TensorShape([]),
            tf.TensorShape([]),
        ),
        max_in_flight_samples_per_worker=10,
    )
    dataset = dataset.batch(BATCH_SIZE)
    print("ready")
    t0 = time.perf_counter()
    for sample in dataset.take(TEST_CNT):
        pass
    t1 = time.perf_counter()
    print(TEST_CNT, t1 - t0)
