import multiprocessing as mp
import secrets
import time

import reth
import perwez

ENV_NAME = "QbertNoFrameskip-v4"
WORKER_BATCH_SIZE = 64
WORKER_CNT = 8


def _trainer(server_url):
    data_recv = perwez.RecvSocket(server_url, "data", broadcast=False)
    weight_send = perwez.SendSocket(server_url, "weight", broadcast=True)

    cnt = 0
    while True:
        latest_res = data_recv.recv()
        shapes = [x.shape for x in latest_res]
        cnt += 1
        if cnt % 5 == 0:
            print(f"recv data: {cnt}, shape: {shapes}")
            weight = secrets.token_bytes(50 * 1024 * 1024)
            weight_send.send(weight)
            time.sleep(0.333)


def _worker(idx, server_url):
    weight_recv = perwez.RecvSocket(server_url, "weight", broadcast=True)
    data_send = perwez.SendSocket(server_url, "data", broadcast=False)

    env = reth.env.make(ENV_NAME)
    buffer = reth.buffer.NumpyBuffer(WORKER_BATCH_SIZE, circular=False)

    w_cnt = 0
    d_cnt = 0
    s0 = env.reset().astype("f4")
    while w_cnt < 10:
        if not weight_recv.empty():
            weight_recv.recv()
            w_cnt += 1
        s1, r, done, _ = env.step(env.action_space.sample())
        s1 = s1.astype("f4")
        buffer.append((s0, r, done, s1))

        if buffer.size == buffer.capacity:
            data_send.send(buffer.data)
            d_cnt += 1
            buffer.clear()
            print(f"worker{idx}: recv weights {w_cnt}, send data {d_cnt}")

        if done:
            s0 = env.reset().astype("f4")
        else:
            s0 = s1


def test_pubsub_full():
    try:
        server_proc, config = perwez.start_server()
        procs = []
        trainer_proc = mp.Process(target=_trainer, args=(config["url"],))
        trainer_proc.start()
        for idx in range(WORKER_CNT):
            proc = mp.Process(target=_worker, args=(idx, config["url"]))
            proc.start()
            procs.append(proc)
        for p in procs:
            p.join()
    finally:
        trainer_proc.terminate()
        trainer_proc.join()
        for p in procs:
            p.terminate()
            p.join()
        server_proc.terminate()
        server_proc.join()
