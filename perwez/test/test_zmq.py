import secrets
import sys
import time
import multiprocessing as mp

import pytest
import zmq

import perwez
from perwez.utils import DEFAULT_HWM

size = 1024 * 1024


def test_zmq_push_block():
    proc, _ = perwez.start_server("default")
    topic = "asd"
    c0 = perwez.connect("default")
    c1 = perwez.connect("default")
    c2 = perwez.connect("default")
    data = secrets.token_bytes(size)

    print("EAGAIN when no client")
    try:
        c0.push(topic, data, ipc=False, noblock=True)
    except zmq.ZMQError:
        pass
    else:
        assert False

    print("c1 join")
    w1 = c1.subscribe(topic)
    for i in range(DEFAULT_HWM):
        c0.push(topic, data, ipc=False)
    print(f"{DEFAULT_HWM} items pushed")
    print("hwm block: push more items")
    for i in range(DEFAULT_HWM * 3):
        try:
            c0.push(topic, data, ipc=False, noblock=True)
        except zmq.ZMQError:
            break
    print(f"{i}/{DEFAULT_HWM * 3} more items in queue")

    res = w1.get(timeout=3)
    assert res == data

    print("c2 join")
    w2 = c2.subscribe(topic)
    time.sleep(1)
    print("hwm block: push more items")
    for i in range(DEFAULT_HWM * 3):
        try:
            c0.push(topic, data, ipc=False, noblock=True)
        except zmq.ZMQError:
            break
    print(f"{i}/{DEFAULT_HWM * 3} more items in queue")
    res = w2.get(timeout=15)
    assert res == data
    print("finished")
    c0.close()
    c1.close()
    c2.close()
    proc.terminate()
    proc.join()


def _producer(topic, idx):
    data = secrets.token_bytes(size)
    c = perwez.connect("default")
    for i in range(20):
        c.push(topic, [f"{idx}-{i}".encode(), data], ipc=False)
    c.close(linger=-1)


def _consumer(topic, idx, sem):
    c = perwez.connect("default")
    w = c.subscribe(topic)
    while True:
        res = w.get()
        print(idx, res[0])
        sem.release()
    c.close(linger=0)


def test_zmq_push_mp():
    mp.set_start_method("spawn", force=True)
    time.sleep(1)
    proc, _ = perwez.start_server("default")
    topic = "asd"

    # consumer first
    print("consumer first")
    processes = []
    sem = mp.Semaphore(0)
    for i in range(3):
        t = mp.Process(target=_consumer, args=(topic, i, sem))
        t.start()
        processes.append(t)
    for i in range(1):
        t = mp.Process(target=_producer, args=(topic, i))
        t.start()
        processes.append(t)
    for _ in range(20):
        sem.acquire()
    for t in processes:
        t.terminate()
        t.join()

    # producer first
    print("producer first")
    processes = []
    sem = mp.Semaphore(0)
    for i in range(1):
        t = mp.Process(target=_producer, args=(topic, i))
        t.start()
        processes.append(t)
    for i in range(3):
        t = mp.Process(target=_consumer, args=(topic, i, sem))
        t.start()
        processes.append(t)
    for _ in range(20):
        sem.acquire()
    for t in processes:
        t.terminate()
        t.join()

    # N to N
    print("N to N")
    processes = []
    sem = mp.Semaphore(0)
    for i in range(4):
        t = mp.Process(target=_producer, args=(topic, i))
        t.start()
        processes.append(t)
    for i in range(5):
        t = mp.Process(target=_consumer, args=(topic, i, sem))
        t.start()
        processes.append(t)
    for _ in range(4 * 20):
        sem.acquire()
    for t in processes:
        t.terminate()
        t.join()

    proc.terminate()
    proc.join()


if __name__ == "__main__":
    pytest.main(sys.argv)
