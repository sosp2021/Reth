import os
import threading
import uuid

import zmq

from .pack import deserialize_data
from ..utils import DEFAULT_HWM, INPROC_INFO_ADDR, INPROC_WATCHER_PREFIX


class WatcherZmqLoop:
    def __init__(
        self, ctx, topic, watcher_sock_id, sub_conflate=True, init_endpoints=None
    ):
        self.ctx = ctx
        self.topic = topic

        self.endpoints = init_endpoints if init_endpoints is not None else []

        self.socks = {}
        for sock_type in [zmq.PULL, zmq.SUB]:
            self.socks[sock_type] = self.ctx.socket(sock_type)
            self.socks[sock_type].set_hwm(DEFAULT_HWM)
            if sock_type == zmq.SUB:
                self.socks[sock_type].subscribe("")
                if sub_conflate:
                    self.socks[sock_type].setsockopt(zmq.CONFLATE, 1)
        self.info_sock = self.ctx.socket(zmq.SUB)
        self.info_sock.set_hwm(DEFAULT_HWM)
        self.info_sock.subscribe("")
        self.info_sock.connect(INPROC_INFO_ADDR)

        self.sock_type_map = {"zmq_push": zmq.PULL, "zmq_pub": zmq.SUB}

        self.dst_sock = self.ctx.socket(zmq.PUSH)
        self.dst_sock.set_hwm(1)
        self.dst_sock.connect(watcher_sock_id)

        for info in self.endpoints:
            self._connect(info["addr"], info["type"])

    def _disconnect(self, url, endpoint_type):
        sock_type = self.sock_type_map[endpoint_type]
        self.socks[sock_type].disconnect(url)

    def _connect(self, url, endpoint_type):
        sock_type = self.sock_type_map[endpoint_type]
        self.socks[sock_type].connect(url)

    def _update_info(self, endpoints):
        old_endpoints = {x["addr"]: x for x in self.endpoints}
        endpoints = {x["addr"]: x for x in endpoints}
        intersection = set(old_endpoints.keys()) & set(endpoints.keys())
        for url, info in old_endpoints.items():
            if url not in intersection:
                self._disconnect(url, info["type"])
        for url, info in endpoints.items():
            if url not in intersection:
                self._connect(url, info["type"])

    def run(self):
        while True:
            rlist, *_ = zmq.select([*self.socks.values(), self.info_sock], [], [])
            if self.info_sock in rlist:
                info = self.info_sock.recv_json()
                self._update_info(info.get(self.topic, []))
            for sock in self.socks.values():
                if sock in rlist:
                    self.dst_sock.send(sock.recv())


class Watcher:
    def __init__(self, ctx, topic, sub_conflate=True, init_endpoints=None):
        self.id = f"{os.getpid()}-{topic}-{str(uuid.uuid4())[:6]}"
        self.ctx = ctx
        self.topic = topic

        self.watcher_sock_id = INPROC_WATCHER_PREFIX + self.id
        self.watcher_sock = self.ctx.socket(zmq.PULL)
        self.watcher_sock.set_hwm(1)
        self.watcher_sock.bind(self.watcher_sock_id)

        init_ev = threading.Event()
        self.worker = None
        self.worker_thread = threading.Thread(
            target=self._worker_thread,
            args=(
                init_ev,
                ctx,
                topic,
                self.watcher_sock_id,
                sub_conflate,
                init_endpoints,
            ),
            daemon=True,
        )
        self.worker_thread.start()
        init_ev.wait()

    def _worker_thread(
        self, init_ev, ctx, topic, watcher_sock_id, sub_conflate, init_endpoints
    ):
        self.worker = WatcherZmqLoop(
            ctx, topic, watcher_sock_id, sub_conflate, init_endpoints
        )
        init_ev.set()
        self.worker.run()

    def empty(self):
        return self.watcher_sock.poll(0) == 0

    def get(self, timeout=None, return_meta=False):
        if timeout is not None:
            ev = self.watcher_sock.poll(timeout * 1000)
            if ev == 0:
                raise TimeoutError("perwez watcher get timeout")
        res = self.watcher_sock.recv()
        return deserialize_data(res, return_meta=return_meta)
