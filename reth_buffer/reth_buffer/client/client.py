import json
import os

import numpy as np


from .sample_fetcher import SampleFetcher
from .utils import get_zmq_socket
from ..utils import ROOT_DIR_PREFIX
from ..utils.thread import DaemonExecutor
from ..utils.shm_buffer import ShmBuffer


class Client:
    def __init__(self, name, sample_preload=None, sample_preload_workers=4):
        self._name = name
        self._root_dir = ROOT_DIR_PREFIX + name
        with open(os.path.join(self._root_dir, "config.json"), "r") as config_file:
            self._config = json.load(config_file)

        if sample_preload is None:
            try:
                import torch

                if torch.cuda.is_available():
                    sample_preload = "torch-cuda-ex"
                else:
                    sample_preload = "numpy"
            except ImportError:
                sample_preload = "numpy"

        self._sample_fetcher = None
        self._sample_fetcher_args = {
            "preload": sample_preload,
            "preload_workers": sample_preload_workers,
        }
        self._send_executor = DaemonExecutor(1)
        self._send_socket = get_zmq_socket(self._config["port"])
        self._shm_buffer = ShmBuffer(root_dir=self._root_dir)

    def _executor_send(self, req, *data):
        req["shm_buffer"] = self._shm_buffer.path
        if data:
            struct = self._shm_buffer.write(*data)
            req["struct"] = struct
        self._send_socket.send_json(req)
        res = self._send_socket.recv_json()
        if "error" in res:
            print(f"[Error] req: {json.dumps(req)}, res: {json.dumps(res)}")

    def append(self, *data, weights=None):
        if weights is None:
            weights = np.array([])
        future = self._send_executor.submit(
            self._executor_send, {"type": "append"}, *data, weights
        )
        return future

    def sample(self, silent=False):
        if self._sample_fetcher is None:
            self._sample_fetcher = SampleFetcher(
                self._config["port"],
                root_dir=self._root_dir,
                **self._sample_fetcher_args,
            )

        return self._sample_fetcher.sample(silent)

    def update_priorities(self, indices, weights):
        future = self._send_executor.submit(
            self._executor_send, {"type": "update_priorities"}, indices, weights
        )
        return future
