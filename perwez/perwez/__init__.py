import atexit
import json
import multiprocessing as mp
import os
import signal
import sys
from urllib.parse import urljoin

import psutil
import requests
from portpicker import pick_unused_port

from .client.client import Client
from .server.server import Server
from .utils import ROOT_DIR_PREFIX, ping_server


def connect(name="default"):
    return Client(name)


def _start(**kwargs):
    server = Server(**kwargs)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    atexit.register(server.close)
    server.start()


def start_server(
    name="default", host="0.0.0.0", port=None, parent_url=None, bootstrap_timeout=3
):
    if port is None:
        port = pick_unused_port()
    if parent_url is not None:
        ping_server(urljoin(parent_url, "/echo"), timeout=bootstrap_timeout)
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_start,
        kwargs={
            "name": name,
            "host": host,
            "port": port,
            "parent_url": parent_url,
        },
        daemon=True,
    )
    process.start()
    ping_server(f"http://localhost:{port}/echo", process.pid)
    return process, port
