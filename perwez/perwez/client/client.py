import json
import os
import threading
import uuid
import weakref
from urllib.parse import urljoin

import msgpack
import portpicker
import socketio
import zmq
from loguru import logger

from .pack import serialize_data
from .watcher import Watcher
from ..utils import (
    DEFAULT_HWM,
    DEFAULT_ZMQ_IO_THREADS,
    INPROC_INFO_ADDR,
    get_config_path,
    ping_server,
)

ENDPOINT_TYPE_MAP = {
    zmq.PUB: "zmq_pub",
    zmq.PUSH: "zmq_push",
}


class PublishManager:
    def __init__(self, ctx, sio_client, server_config):
        self.ctx = ctx
        self.sio_client = sio_client
        self.local_ip = server_config["ip"]

        self.root_dir = os.path.join(
            server_config["root_dir"], f"{os.getpid()}-socket-{str(uuid.uuid4())}"
        )
        os.makedirs(self.root_dir, exist_ok=True)

        self.socks = {}

        self.endpoints = []

        self.packer = msgpack.Packer(autoreset=False)
        self.packer_lock = threading.Lock()

    def close(self, linger=-1):
        for sock in self.socks.values():
            sock.close(linger=linger)

    def register_all(self):
        self.sio_client.emit("register", self.endpoints)

    def _create_socket(self, sock_type, topic, ipc=True, conflate=False):
        key = (sock_type, topic, ipc, conflate)
        if key in self.socks:
            return

        if ipc:
            path = os.path.join(self.root_dir, f"{os.getpid()}-{topic}-{sock_type}-{conflate}")
            addr = f"ipc://{path}"
        else:
            port = portpicker.pick_unused_port()
            addr = f"tcp://{self.local_ip}:{port}"

        self.socks[key] = self.ctx.socket(sock_type)
        self.socks[key].set_hwm(DEFAULT_HWM)
        self.socks[key].bind(addr)
        if conflate:
            self.socks[key].setsockopt(zmq.CONFLATE, 1)

        item = {
            "type": ENDPOINT_TYPE_MAP[sock_type],
            "addr": addr,
            "topic": topic,
            "host": self.local_ip,
        }
        self.endpoints.append(item)
        self.register_all()

    def send(
        self,
        sock_type,
        topic,
        data,
        ipc=True,
        noblock=False,
        conflate=False,
        compression=None,
        **compression_args,
    ):
        assert sock_type in [zmq.PUB, zmq.PUSH]
        self._create_socket(sock_type, topic, ipc, conflate)

        key = (sock_type, topic, ipc, conflate)
        with self.packer_lock:
            with serialize_data(
                data, compression, self.packer, **compression_args
            ) as packed_data:
                flags = zmq.NOBLOCK if noblock else 0
                self.socks[key].send(packed_data, flags=flags)


class Client:
    def __init__(self, server_name):
        self.ctx = zmq.Context(DEFAULT_ZMQ_IO_THREADS)
        self.endpoint_info = {}
        self.watchers = weakref.WeakSet()

        config_path = get_config_path(server_name)
        assert os.path.exists(config_path)
        with open(config_path, "r") as f:
            self.server_config = json.load(f)

        # check connection
        try:
            pid = self.server_config["pid"]
            url = self.server_config["url"]
            ping_server(urljoin(url, "/echo"), pid)
        except BaseException:
            logger.exception(
                f"perwez server can not be accessed. name {server_name}, url {url}, pid {pid}"
            )
            raise

        self.sio_client = socketio.Client()
        self.sio_client.on("connect", self._on_connect)
        self.sio_client.on("update", self._on_info_updated)

        self.pub_manager = PublishManager(self.ctx, self.sio_client, self.server_config)
        init_ev = threading.Event()
        self._info_sock = None
        self._info_lock = threading.Lock()
        self._sio_thread = threading.Thread(
            target=self._sio_poller, args=(init_ev,), daemon=True
        )
        self._sio_thread.start()
        init_ev.wait()

    def close(self, linger=-1):
        self.pub_manager.close(linger)
        if self.sio_client.connected:
            self.sio_client.disconnect()
        # pylint: disable=protected-access
        if self.sio_client._reconnect_task is not None:
            self.sio_client._reconnect_abort.set()
        # pylint: enable=protected-access

    def _sio_poller(self, init_ev):
        self._info_sock = self.ctx.socket(zmq.PUB)
        self._info_sock.setsockopt(zmq.CONFLATE, 1)
        self._info_sock.bind(INPROC_INFO_ADDR)
        self.sio_client.connect(self.server_config["url"], transports="websocket")
        init_ev.set()
        self.sio_client.wait()

    def _on_connect(self):
        self.pub_manager.register_all()

    def _on_info_updated(self, endpoints):
        new_info = {}
        for x in endpoints:
            new_info.setdefault(x["topic"], [])
            new_info[x["topic"]].append(x)
        with self._info_lock:
            self.endpoint_info = new_info
            self._info_sock.send_json(new_info)

    def publish(
        self, topic, data, ipc=True, noblock=False, conflate=True, compression=None, **compression_args
    ):
        self.pub_manager.send(
            zmq.PUB, topic, data, ipc, noblock, conflate, compression, **compression_args
        )

    def push(
        self, topic, data, ipc=True, noblock=False, conflate=False, compression=None, **compression_args
    ):
        self.pub_manager.send(
            zmq.PUSH, topic, data, ipc, noblock, conflate, compression, **compression_args
        )

    def subscribe(self, topic, conflate):
        with self._info_lock:
            watcher = Watcher(
                self.ctx, topic, conflate, self.endpoint_info.get(topic)
            )
        self.watchers.add(watcher)
        return watcher
