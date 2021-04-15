import multiprocessing as mp

from portpicker import pick_unused_port

from .client.async_socket import AsyncRecvSocket, AsyncSendSocket
from .client.socket import RecvSocket, SendSocket
from .server.server import Server
from .utils import ping_url


def _start(**kwargs):
    server = Server(**kwargs)
    server.start()


def start_server(host="0.0.0.0", port=None):
    if port is None:
        port = pick_unused_port()
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_start,
        kwargs={
            "host": host,
            "port": port,
        },
        daemon=True,
    )
    process.start()
    config = ping_url(f"http://localhost:{port}/echo", process.pid)
    return process, config
