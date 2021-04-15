import os

import zmq
from aiohttp import web
from loguru import logger
from portpicker import pick_unused_port

from .endpoint_manager import EndpointManager
from .routes import register_routes
from ..utils import get_local_ip


class Server:
    def __init__(self, host="0.0.0.0", port=None, zmq_port=None):
        self.host = host
        if port is None:
            port = pick_unused_port()
        self.port = port
        if zmq_port is None:
            zmq_port = pick_unused_port()
        self.zmq_port = zmq_port

        # local ip & url
        if host == "0.0.0.0":
            self.ip = get_local_ip()
        else:
            self.ip = host

        self.url = f"http://{self.ip}:{self.port}"
        self.zmq_addr = f"tcp://{self.ip}:{self.zmq_port}"

        self.config = {
            "pid": os.getpid(),
            "host": self.host,
            "port": self.port,
            "url": self.url,
            "zmq_addr": self.zmq_addr,
            "ip": self.ip,
        }

        self.app = web.Application()
        self.app["config"] = self.config
        self.app["endpoint_manager"] = EndpointManager(self.zmq_addr)
        register_routes(self.app)

        self.app.on_cleanup.append(self.cleanup)

        logger.info(f"Perwez server started at {self.url}")

    def start(self):
        web.run_app(
            self.app,
            host=self.host,
            port=self.port,
            print=False,
            shutdown_timeout=False,
        )

    async def cleanup(self, _):
        ctx = zmq.Context.instance()
        ctx.destroy(linger=0)
        logger.info(f"perwez server {self.url} exited.")
