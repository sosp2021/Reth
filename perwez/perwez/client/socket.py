import time
from urllib.parse import urljoin

import portpicker
import requests
import zmq
from loguru import logger

from .pack import serialize, deserialize
from .utils import send_heartbeat, ZMQ_IN, ZMQ_REVERSE
from ..utils import (
    DEFAULT_ZMQ_IO_THREADS,
    get_server_config,
    HEARTBEAT_INTERVAL,
    HEARTBEAT_TIMEOUT,
)


class SendSocket:
    def __init__(
        self,
        server_url,
        topic,
        broadcast=None,
        sock_type=None,
        ctx=None,
        conflate=None,
        hwm=5,
        public=True,
    ):
        if ctx is None:
            ctx = zmq.Context.instance(DEFAULT_ZMQ_IO_THREADS)

        if sock_type is None:
            assert broadcast is not None, "broadcast or sock_type should be specified"
            sock_type = zmq.PUB if broadcast else zmq.PUSH

        assert sock_type in [zmq.PUB, zmq.PUSH]

        if conflate is None:
            conflate = sock_type == zmq.PUB

        if public:
            self.socket = _PublicSocket(
                server_url, topic, sock_type, ctx, conflate, hwm
            )
        else:
            self.socket = _PrivateSocket(
                server_url, topic, sock_type, ctx, conflate, hwm
            )

    def __del__(self):
        self.close()

    def close(self, linger=0):
        if hasattr(self, "socket"):
            try:
                self.socket.close(linger)
            except requests.exceptions.ConnectionError:
                pass

    def poll(self, timeout=None):
        return self.socket.poll(timeout)

    def full(self):
        return self.socket.poll(0) == 0

    def send(self, data, timeout=None, compress=False):
        self.socket.send(data, timeout, compress)


class RecvSocket:
    def __init__(
        self,
        server_url,
        topic,
        broadcast=None,
        sock_type=None,
        ctx=None,
        conflate=None,
        hwm=5,
        public=False,
    ):
        if ctx is None:
            ctx = zmq.Context.instance(DEFAULT_ZMQ_IO_THREADS)

        if sock_type is None:
            assert broadcast is not None, "broadcast or sock_type should be specified"
            sock_type = zmq.SUB if broadcast else zmq.PULL

        assert sock_type in [zmq.SUB, zmq.PULL]

        if conflate is None:
            conflate = sock_type == zmq.SUB

        if public:
            self.socket = _PublicSocket(
                server_url, topic, sock_type, ctx, conflate, hwm
            )
        else:
            self.socket = _PrivateSocket(
                server_url, topic, sock_type, ctx, conflate, hwm
            )

    def __del__(self):
        self.close()

    def close(self, linger=0):
        if hasattr(self, "socket"):
            try:
                self.socket.close(linger)
            except requests.exceptions.ConnectionError:
                pass

    def poll(self, timeout=None):
        return self.socket.poll(timeout)

    def empty(self):
        return self.socket.poll(0) == 0

    def recv(self, timeout=None):
        return self.socket.recv(timeout)


class _PublicSocket:
    def __init__(
        self,
        server_url,
        topic,
        sock_type,
        ctx,
        conflate,
        hwm,
    ):
        self.server_url = server_url
        self.topic = topic
        self.sock_type = sock_type

        self._closed = False
        self.wrapper = _SocketWrapper(sock_type, ctx, conflate, hwm)

        server_config = get_server_config(self.server_url)
        local_ip = server_config["remote_ip"]
        self.poll_flag = zmq.POLLIN if ZMQ_IN[self.sock_type] else zmq.POLLOUT

        port = portpicker.pick_unused_port()
        self.addr = f"tcp://{local_ip}:{port}"
        self.wrapper.sock.bind(self.addr)

        self.last_heartbeat = 0
        # heartbeat payload
        self.payload = {
            "type": self.sock_type,
            "addr": self.addr,
            "topic": self.topic,
            "host": local_ip,
            "ttl": HEARTBEAT_TIMEOUT,
        }
        self._send_heartbeat()

    def _send_heartbeat(self):
        if time.monotonic() - self.last_heartbeat > HEARTBEAT_INTERVAL:
            try:
                send_heartbeat(self.server_url, self.payload, timeout=5)
                self.last_heartbeat = time.monotonic()
            except Exception as e:
                logger.error(f"Failed to send heartbeat. addr: {self.addr}, error: {e}")

    def __del__(self):
        self.close()

    def close(self, linger=0):
        if self._closed:
            return
        self._closed = True

        res = requests.delete(
            urljoin(self.server_url, "/endpoints"), json={"addr": self.addr}
        )
        res.raise_for_status()
        self.wrapper.sock.close(linger)

    def poll(self, timeout=None):
        return self.wrapper.sock.poll(timeout=timeout, flags=self.poll_flag)

    def send(self, data, timeout=None, compress=False):
        self._send_heartbeat()
        res = self.poll(timeout)
        if res == 0:
            raise TimeoutError("perwez send timeout")
        self.wrapper.send(data, compress)

    def recv(self, timeout=None):
        self._send_heartbeat()
        res = self.poll(timeout)
        if res == 0:
            raise TimeoutError("perwez recv timeout")
        return self.wrapper.recv()


class _PrivateSocket:
    def __init__(
        self,
        server_url,
        topic,
        sock_type,
        ctx,
        conflate,
        hwm,
    ):
        self.server_url = server_url
        self.topic = topic
        self.sock_type = sock_type

        self._closed = False
        self.wrapper = _SocketWrapper(sock_type, ctx, conflate, hwm)

        server_config = get_server_config(self.server_url)
        self.info_sock = ctx.socket(zmq.SUB)
        self.info_sock.subscribe(b"")
        self.info_sock.setsockopt(zmq.CONFLATE, 1)
        self.info_sock.connect(server_config["zmq_addr"])

        # addr -> endpoint info
        self.info_dict = {}
        self._info_initialized = False

        self.poller = zmq.Poller()
        self.poller.register(self.info_sock, zmq.POLLIN)
        self.poll_flag = zmq.POLLIN if ZMQ_IN[self.sock_type] else zmq.POLLOUT
        self.poller.register(self.wrapper.sock, self.poll_flag)

    def _init_endpoints(self):
        self._info_initialized = True
        res = requests.get(urljoin(self.server_url, "/endpoints"))
        res.raise_for_status()
        self._update_endpoints(res.json())

    def _update_endpoints(self, ep_list):
        ep_list = [
            info
            for info in ep_list
            if info["type"] == ZMQ_REVERSE[self.sock_type]
            and info["topic"] == self.topic
        ]
        new_info_dict = {info["addr"]: info for info in ep_list}
        # disconnect
        for addr in self.info_dict:
            if addr not in new_info_dict:
                self.wrapper.sock.disconnect(addr)
        # connect
        for addr in new_info_dict:
            if addr not in self.info_dict:
                self.wrapper.sock.connect(addr)
        self.info_dict = new_info_dict

    def __del__(self):
        self.close()

    def close(self, linger=0):
        if self._closed:
            return
        self._closed = True

        self.info_sock.close(linger)
        self.wrapper.sock.close(linger)

    def poll(self, timeout=None):
        if not self._info_initialized:
            self._init_endpoints()
        remain_time = timeout
        while True:
            ts = time.perf_counter()
            events = dict(self.poller.poll(remain_time))
            if len(events) == 0:
                return 0

            if self.info_sock in events:
                res = self.info_sock.recv_json()
                self._update_endpoints(res)

            if self.wrapper.sock in events:
                return self.poll_flag

            if remain_time is not None:
                duration = time.perf_counter() - ts
                remain_time = max(0, remain_time - duration)

    def send(self, data, timeout=None, compress=False):
        res = self.poll(timeout)
        if res == 0:
            raise TimeoutError("perwez send timeout")
        self.wrapper.send(data, compress)

    def recv(self, timeout=None):
        res = self.poll(timeout)
        if res == 0:
            raise TimeoutError("perwez recv timeout")
        return self.wrapper.recv()


class _SocketWrapper:
    def __init__(
        self,
        sock_type,
        ctx,
        conflate,
        hwm,
    ):
        self.sock = ctx.socket(sock_type)
        self.sock.setsockopt(zmq.IMMEDIATE, 1)

        if sock_type == zmq.SUB:
            self.sock.subscribe(b"")

        if conflate:
            self.sock.setsockopt(zmq.CONFLATE, 1)
        else:
            self.sock.set_hwm(hwm)

    def send(self, data, compress=False):
        packed = serialize(data, compress=compress)
        self.sock.send(packed)

    def recv(self):
        bin_data = self.sock.recv()
        res = deserialize(bin_data)
        return res
