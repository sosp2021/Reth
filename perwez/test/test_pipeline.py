import secrets
import sys
import multiprocessing as mp

import pytest
import zmq

import perwez

size = 1024 * 1024
topic = "asd"


def _producer_block(data, url, hwm, public):
    w = perwez.SendSocket(url, topic, zmq.PUSH, hwm=hwm, public=public)
    while True:
        cnt = 0
        w.poll()
        while not w.full():
            cnt += 1
            w.send(data)
        print(f"producer: sent {cnt}", flush=True)


def _consumer_block(data, url, hwm, public, step_event, fin_event):
    r = perwez.RecvSocket(url, topic, zmq.PULL, hwm=hwm, public=public)
    res = r.recv()
    assert res == data
    step_event.set()
    fin_event.wait()


@pytest.mark.parametrize("server_public", [True, False])
def test_pipeline_block(server_public):
    proc, config = perwez.start_server()
    data = secrets.token_bytes(size)
    hwm = 5
    w1 = None
    r1 = None
    r2 = None
    try:
        w1 = mp.Process(
            target=_producer_block,
            args=(data, config["url"], hwm, server_public),
        )
        w1.start()
        fin_event = mp.Event()
        r1_event = mp.Event()
        r1 = mp.Process(
            target=_consumer_block,
            args=(
                data,
                config["url"],
                hwm,
                not server_public,
                r1_event,
                fin_event,
            ),
        )
        r1.start()
        print("r1 join")
        r1_event.wait()
        print("r1 recv", flush=True)
        r2_event = mp.Event()
        r2 = mp.Process(
            target=_consumer_block,
            args=(
                data,
                config["url"],
                hwm,
                not server_public,
                r2_event,
                fin_event,
            ),
        )
        r2.start()
        print("r2 join")
        r2_event.wait()
        print("r2 recv", flush=True)
        fin_event.set()
    finally:
        if w1:
            w1.terminate()
            w1.join()
        if r1:
            r1.terminate()
            r1.join()
        if r2:
            r2.terminate()
            r2.join()
        print("term", flush=True)
        proc.terminate()
        proc.join()


def _producer(url, idx):
    data = secrets.token_bytes(size)
    c = perwez.SendSocket(url, topic, zmq.PUSH, hwm=5)
    i = 0
    while True:
        i += 1
        print("send", f"{idx}-{i}")
        c.send([f"{idx}-{i}".encode(), data])


def _consumer(url, idx):
    c = perwez.RecvSocket(url, topic, zmq.PULL, hwm=5)
    for _ in range(10):
        res = c.recv()
        print(idx, res[0])


def test_pipeline_consumer_first():
    processes = []
    consumers = []
    try:
        proc, config = perwez.start_server()
        url = config["url"]
        for i in range(3):
            t = mp.Process(target=_consumer, args=(url, i))
            t.start()
            processes.append(t)
            consumers.append(t)
        for i in range(1):
            t = mp.Process(target=_producer, args=(url, i))
            t.start()
            processes.append(t)
        for p in consumers:
            p.join()
    finally:
        for t in processes:
            t.terminate()
            t.join()
        proc.terminate()
        proc.join()


def test_pipeline_producer_first():
    processes = []
    consumers = []
    try:
        proc, config = perwez.start_server()
        url = config["url"]
        for i in range(1):
            t = mp.Process(target=_producer, args=(url, i))
            t.start()
            processes.append(t)
        for i in range(3):
            t = mp.Process(target=_consumer, args=(url, i))
            t.start()
            processes.append(t)
            consumers.append(t)
        for p in consumers:
            p.join()
    finally:
        for t in processes:
            t.terminate()
            t.join()
        proc.terminate()
        proc.join()


def test_pipeline_nxn():
    processes = []
    consumers = []
    try:
        proc, config = perwez.start_server()
        url = config["url"]
        for i in range(3):
            t = mp.Process(target=_producer, args=(url, i))
            t.start()
            processes.append(t)
        for i in range(3):
            t = mp.Process(target=_consumer, args=(url, i))
            t.start()
            processes.append(t)
            consumers.append(t)
        for p in consumers:
            p.join()
    finally:
        for t in processes:
            t.terminate()
            t.join()
        proc.terminate()
        proc.join()


if __name__ == "__main__":
    pytest.main(sys.argv)
