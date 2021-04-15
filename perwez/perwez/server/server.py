import json
import os
import re
import shutil
from urllib.parse import urljoin

import psutil
import requests
import socketio
from aiohttp import web
from loguru import logger
from portalocker import Lock as FileLock
from portpicker import pick_unused_port

from .routes import register_routes
from .sio_name_server import SioNameServer
from ..utils import get_local_ip, get_lock_path, get_root_dir


class Server:
    def __init__(
        self,
        name,
        host="0.0.0.0",
        port=None,
        parent_url=None,
    ):
        self.name = name
        self.host = host
        if port is None:
            port = pick_unused_port()
        self.port = port
        self.parent_url = parent_url
        self._closed = False

        self.root_dir = get_root_dir(self.name)
        os.makedirs(self.root_dir, exist_ok=True)
        self.lock_path = get_lock_path(self.name)

        # normalize parent_url
        if parent_url is not None:
            if not re.match(r"([a-zA-Z]{2,20}):\/\/", parent_url):
                parent_url = f"http://{parent_url}"

        # local ip & url
        if parent_url is None:
            self.ip = get_local_ip()
            self.url = f"http://{self.ip}:{self.port}"
        else:
            res = requests.get(urljoin(parent_url, "/echo"))
            self.ip = res.json()["remote_ip"]
            self.url = f"http://{self.ip}:{self.port}"

        self.config = {
            "pid": os.getpid(),
            "name": self.name,
            "port": self.port,
            "parent_url": self.parent_url,
            "root_dir": self.root_dir,
            "ip": self.ip,
            "url": self.url,
        }

        # config file
        with FileLock(self.lock_path):
            config_path = os.path.join(self.root_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as config_file:
                    data = json.load(config_file)
                # check exists process
                pid = data.get("pid")
                if pid is not None and psutil.pid_exists(pid):
                    raise Exception(
                        f"perwez_server with name '{name}' is already started. pid: {pid}"
                    )

                logger.warning(
                    f"Old server with name '{name}', pid {pid} not found. Remove old server temp dir"
                )
                shutil.rmtree(self.root_dir)
            os.makedirs(self.root_dir, exist_ok=True)
            # create config
            with open(config_path, "w") as config_file:
                json.dump(self.config, config_file)

        self.app = web.Application()
        self.app["config"] = self.config
        register_routes(self.app)

        # socket io
        self.sio = socketio.AsyncServer(async_mode="aiohttp")
        self.sio.attach(self.app)
        self.sio_name_server = SioNameServer(self.sio, parent_url)
        self.app["sio_ns"] = self.sio_name_server

        logger.info(f"Perwez server started at {self.url}")

    def start(self):
        web.run_app(
            self.app,
            host=self.host,
            port=self.port,
            print=False,
            shutdown_timeout=False,
        )

    def close(self):
        if self._closed:
            return
        print("perwez server exiting...")
        self._closed = True
        shutil.rmtree(self.root_dir)
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)
