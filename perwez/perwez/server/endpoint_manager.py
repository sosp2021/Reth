import time
import threading

import zmq


class EndpointManager:
    def __init__(self, broadcast_addr):
        # addr -> endpoint
        self.by_addr = {}

        # push socket
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(broadcast_addr)

        self._lock = threading.Lock()

    def _remove(self, addr):
        if addr not in self.by_addr:
            return
        info = self.by_addr[addr]
        del self.by_addr[addr]
        return info

    def _check_ttl(self, info_list):
        current_time = time.time()
        expired = [
            info["addr"] for info in info_list if info["expire_time"] < current_time
        ]
        for addr in expired:
            self._remove(addr)

    def _on_changed(self):
        self._check_ttl(self.by_addr.values())
        self.sock.send_json(list(self.by_addr.values()))

    def add(self, info):
        current_time = time.time()
        info["expire_time"] = info["ttl"] + current_time
        addr = info["addr"]
        with self._lock:
            self.by_addr[addr] = info
            self._on_changed()

    def remove(self, addr):
        with self._lock:
            info = self._remove(addr)
            if info is not None:
                self._on_changed()

    def query(self):
        with self._lock:
            self._check_ttl(self.by_addr.values())
            return list(self.by_addr.values())
