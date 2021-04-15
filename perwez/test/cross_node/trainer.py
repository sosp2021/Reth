import atexit
import secrets
import time
import multiprocessing as mp

import perwez


def _data_fetcher():
    pwz = perwez.connect("default")
    data_watcher = pwz.subscribe("data")
    cnt = 0
    while True:
        latest_res = data_watcher.get()
        shapes = [x.shape for x in latest_res]
        cnt += 1
        if cnt % 1 == 0:
            print(f"recv data: {cnt}, shape: {shapes}")


def _weight_producer():
    pwz = perwez.connect("default")

    while True:
        weight = secrets.token_bytes(50 * 1024 * 1024)
        pwz.publish("weight", weight, ipc=False)
        time.sleep(0.333)


def _exit(proc_list):
    for proc in proc_list:
        proc.terminate()
        proc.join()


if __name__ == "__main__":
    pwz_proc, _ = perwez.start_server(port=12333)
    processes = [pwz_proc]

    p1 = mp.Process(target=_data_fetcher)
    p1.start()
    processes.append(p1)
    p2 = mp.Process(target=_weight_producer)
    p2.start()
    processes.append(p2)

    atexit.register(_exit, processes)

    for p in processes:
        p.join()
