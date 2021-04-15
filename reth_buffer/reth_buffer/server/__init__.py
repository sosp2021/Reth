import gc
import json
import os
import shutil
import signal
import sys

import psutil
from portalocker import Lock as FileLock
from portpicker import pick_unused_port

from .buffer_service import BufferService
from .zmq_server import start_zmq_server
from ..utils import ROOT_DIR_PREFIX

EXITED = False


def start(
    server_type="zmq",
    name="default",
    port=None,
    root_dir=None,
    batch_size=64,
    buffer_config=None,
    sample_max_threads=4,
):
    if port is None:
        port = pick_unused_port()
    if root_dir is None:
        root_dir = ROOT_DIR_PREFIX + name
    os.makedirs(root_dir, exist_ok=True)

    # init config
    config = {"pid": os.getpid(), "name": name, "port": port, "type": server_type}
    with FileLock(os.path.join(root_dir, "lock")):
        config_path = os.path.join(root_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                data = json.load(config_file)
            pid = data.get("pid")
            if pid is not None:
                if psutil.pid_exists(pid):
                    raise Exception(f"reth_buffer with name {name} is already started")
            os.remove(config_path)

        with open(config_path, "w") as config_file:
            json.dump(config, config_file)
    # graceful exit
    def _graceful_exit(*args):
        global EXITED
        if not EXITED:
            EXITED = True
            print("[Graceful Exit] deleting temp files...")
            gc.collect()
            shutil.rmtree(root_dir)
            print("[Graceful Exit] fin")
            sys.exit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    buffer_service = BufferService(
        root_dir, batch_size, buffer_config, sample_max_threads
    )
    if server_type == "zmq":
        start_zmq_server(port, buffer_service)
    else:
        raise Exception("Invalid server type")
