import multiprocessing as mp
import time

import numpy as np

import reth_buffer

WORKER_BATCH_SIZE = 128
WORKER_CNT = 8
TRAINER_CNT = 8
BATCH_SIZE = 64


def dummy_worker(idx):
    client = reth_buffer.connect()
    buffer = [[] for _ in range(5)]
    t0 = time.perf_counter()
    ts = 0
    while True:
        ts += 1
        if ts % 1000 == 0:
            t1 = time.perf_counter()
            print(f"worker {idx}, step: {ts}, time: {t1 - t0}")
        s0 = np.random.rand(4, 84, 84)
        s1 = np.random.rand(4, 84, 84)
        a = np.random.randint(8)
        r = np.random.rand()
        done = np.random.rand()
        result = (s0, a, r, s1, done)
        for i, val in enumerate(result):
            buffer[i].append(val)

        if len(buffer[0]) == WORKER_BATCH_SIZE:
            x = [np.asarray(x, dtype="f8") for x in buffer]
            client.append(*x, np.random.rand(len(buffer[0])) * 20)
            buffer = [[] for _ in range(5)]
        time.sleep(0.01)


def dummy_trainer(idx):
    client = reth_buffer.connect()
    ts = 0
    t0 = time.perf_counter()
    while True:
        ts += 1
        if ts % 1000 == 0:
            t1 = time.perf_counter()
            print(f"trainer {idx}, step: {ts}, time: {t1 - t0}")
        *data, indices, weights = client.sample()
        time.sleep(0.01)
        client.update_priorities(indices, np.random.rand(len(indices)) * 20)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    buffer_process = reth_buffer.start_server(batch_size=BATCH_SIZE)
    workers = []
    for i in range(WORKER_CNT):
        proc = mp.Process(target=dummy_worker, args=(i,), daemon=True)
        proc.start()
        workers.append(proc)
    trainers = []
    for i in range(TRAINER_CNT):
        proc = mp.Process(target=dummy_trainer, args=(i,), daemon=True)
        proc.start()
        trainers.append(proc)
    buffer_process.join()
