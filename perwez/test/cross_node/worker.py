import atexit
import multiprocessing as mp
import os

import reth
import perwez

ENV_NAME = "QbertNoFrameskip-v4"
WORKER_BATCH_SIZE = 512
WORKER_CNT = 20
DATA_COMPRESSION = "lz4"
PARENT_IP = os.environ.get("PAI_HOST_IP_trainer_0")


def _worker(worker_idx):
    pwz = perwez.connect("default")
    weight_watcher = pwz.subscribe("weight", True)

    env = reth.env.make(ENV_NAME)
    buffer = reth.buffer.NumpyBuffer(WORKER_BATCH_SIZE, circular=False)

    w_cnt = 0
    d_cnt = 0
    s0 = env.reset().astype("f4")

    while True:
        if not weight_watcher.empty():
            weight_watcher.get()
            w_cnt += 1
        s1, r, done, _ = env.step(env.action_space.sample())
        s1 = s1.astype("f4")
        buffer.append((s0, r, done, s1))

        if buffer.size == buffer.capacity:
            pwz.push("data", buffer.data, ipc=False, compression=DATA_COMPRESSION)
            d_cnt += 1
            buffer.clear()
            if w_cnt % 1 == 0:
                print(f"worker{worker_idx}: recv weights {w_cnt}, send data {d_cnt}")

        if done:
            s0 = env.reset().astype("f4")
        else:
            s0 = s1


def _exit(proc_list):
    for proc in proc_list:
        proc.terminate()
        proc.join()


def main():
    pwz_proc, _ = perwez.start_server(
        name="default", parent_url=f"http://{PARENT_IP}:12333"
    )
    processes = [pwz_proc]
    for idx in range(WORKER_CNT):
        proc = mp.Process(target=_worker, args=(idx,))
        proc.start()
        processes.append(proc)

    atexit.register(_exit, processes)

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()
