import multiprocessing as mp
import time

import numpy as np

import reth_buffer

CAPACITY = 100000
BATCH_SIZE = 32


def test_basic():
    rb_proc, addr = reth_buffer.start_server(CAPACITY, BATCH_SIZE)
    try:
        client = reth_buffer.Client(addr)
        loader = reth_buffer.NumpyLoader(addr)
        # 1. append
        data = [np.random.rand(1000, 4, 84), np.random.rand(1000), np.arange(1000)]
        buf_weights = np.random.rand(1000) + 1
        for start in range(0, 1000, 100):
            batch = [x[start : start + 100] for x in data]
            client.append(batch, buf_weights[start : start + 100])
        # 2. sample

        for _ in range(10):
            batch, indices, weights = loader.sample()
            for i, idx in enumerate(indices):
                assert batch[2][i] == idx
                assert batch[1][i] == data[1][idx]
        # 3. update priorities
        buf_weights = np.random.rand(1000) + 10
        client.update_priorities(np.arange(1000), buf_weights)
    finally:
        rb_proc.terminate()
        rb_proc.join()


def _worker(addr):
    client = reth_buffer.Client(addr)
    for _ in range(10):
        print("worker", _, flush=True)
        s0 = np.random.rand(BATCH_SIZE, 4, 84, 84)
        s1 = np.random.rand(BATCH_SIZE, 4, 84, 84)
        a = np.random.randint(0, 8, BATCH_SIZE)
        r = np.random.rand(BATCH_SIZE)
        done = np.random.rand(BATCH_SIZE)
        weights = np.random.rand(BATCH_SIZE) + 1
        client.append([s0, a, r, s1, done], weights)
        time.sleep(0.1)


def _trainer(addr):
    client = reth_buffer.Client(addr)
    loader = reth_buffer.NumpyLoader(addr)
    for _ in range(10):
        print(_, flush=True)
        data, indices, weights = loader.sample()
        assert len(data) == 5
        time.sleep(0.1)
        client.update_priorities(indices, np.random.rand(len(indices)))


def test_mp():
    mp.set_start_method("spawn", force=True)
    buf_proc, addr = reth_buffer.start_per(
        CAPACITY, BATCH_SIZE, sample_start=BATCH_SIZE
    )
    procs = []
    try:
        for _ in range(4):
            proc = mp.Process(target=_worker, args=(addr,))
            proc.start()
            procs.append(proc)
        for _ in range(4):
            proc = mp.Process(target=_trainer, args=(addr,))
            proc.start()
            procs.append(proc)
        for p in procs:
            p.join()
    finally:
        for p in procs:
            p.terminate()
            p.join()
        buf_proc.terminate()
        buf_proc.join()
