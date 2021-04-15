import json
import threading
import time
from queue import Queue

from .utils import get_zmq_socket
from ..server.buffer import ExternalMemmapBuffer
from ..utils import QUEUE_SIZE_PER_THREAD
from ..utils.shm_buffer import ShmBuffer
from ..utils.thread import start_thread

try:
    import torch
except ImportError:
    pass


class _TorchExPreloader:
    def __init__(self, buffer, batch_size, swap_size):
        self._tensors = [torch.as_tensor(col) for col in buffer.buffers]
        self._pin_memory_swap = [
            [
                torch.empty(
                    (batch_size, *col.shape[1:]), dtype=col.dtype, pin_memory=True
                )
                for col in self._tensors
            ]
            for _ in range(swap_size)
        ]
        self._cur = 0

    def select(self, indices):
        self._cur = (self._cur + 1) % len(self._pin_memory_swap)
        item = self._pin_memory_swap[self._cur]
        res = []
        for i, col in enumerate(item):
            torch.index_select(self._tensors[i], 0, torch.as_tensor(indices), out=col)
            res.append(col.cuda(non_blocking=True))
        return res


class SampleFetcher:
    def __init__(self, port, root_dir, preload, preload_workers=4):
        self.buffer = None
        self.shm_buffer = ShmBuffer(root_dir=root_dir)

        self._queue_size = QUEUE_SIZE_PER_THREAD
        self.res_queue = Queue(self._queue_size)
        self._res_queue_empty_count = 0

        self._indices_queue = Queue()
        self.preload = preload
        self._preload_workers = [
            start_thread(self._preload_worker, daemon=True)
            for _ in range(preload_workers)
        ]
        # extra
        self._ex_preloader = None

        self._fetch_event = threading.Event()
        self._fetcher = start_thread(
            self._fetch_worker, args=(get_zmq_socket(port),), daemon=True
        )

        # init
        threading.Thread(
            target=self._init_buffer, args=(get_zmq_socket(port),), daemon=True
        ).run()

    def _init_buffer(self, socket):
        while True:
            socket.send_json({"type": "echo"})
            res = socket.recv_json()
            if res["buffer"]["sample_start"]:
                self.buffer = ExternalMemmapBuffer(res["buffer"]["root_dir"])
                self._fetch_event.set()
                break
            time.sleep(1)

    def _preload_worker(self):
        while True:
            indices, weights = self._indices_queue.get()
            with self.buffer.lock_sh():
                if self.preload == "numpy":
                    data = [x.copy() for x in self.buffer.select(indices)]
                elif self.preload == "torch-cuda":
                    data = self.buffer.select(indices)
                    weights = (
                        torch.as_tensor(weights).pin_memory().cuda(non_blocking=True)
                    )
                    data = [
                        torch.as_tensor(x).pin_memory().cuda(non_blocking=True)
                        for x in data
                    ]
                elif self.preload == "torch-pin":
                    data = self.buffer.select(indices)
                    data = [torch.as_tensor(x).pin_memory() for x in data]
                elif self.preload == "torch-cuda-ex":
                    if self._ex_preloader is None:
                        self._ex_preloader = _TorchExPreloader(
                            self.buffer, len(indices), self._queue_size + 5
                        )

                    data = self._ex_preloader.select(indices)
                    weights = (
                        torch.as_tensor(weights).pin_memory().cuda(non_blocking=True)
                    )
                else:
                    raise Exception(f"Invalid preload method {self.preload}")
            self.res_queue.put((*data, indices, weights))

    def _fetch_worker(self, socket):
        while True:
            self._fetch_event.wait()
            req = {
                "type": "sample",
                "shm_buffer": self.shm_buffer.path,
                "size": QUEUE_SIZE_PER_THREAD,
            }
            socket.send_json(req)
            res = socket.recv_json()
            if "error" in res:
                print(f"[Error] req: {json.dumps(req)}, res: {json.dumps(res)}")
                return

            res_data = self.shm_buffer.read(res["struct"])
            assert len(res_data) % 2 == 0
            for i in range(len(res_data) // 2):
                indices = res_data[i * 2].copy()
                weights = res_data[i * 2 + 1].copy()
                self._indices_queue.put((indices, weights))
            if self.res_queue.qsize() + self._indices_queue.qsize() >= self._queue_size:
                self._fetch_event.clear()

    def sample(self, silent=False):
        if self.res_queue.empty() and not silent:
            self._res_queue_empty_count += 1
            if self._res_queue_empty_count > 10:
                print("[Warning] client is waiting for sample_fetcher")
                self._res_queue_empty_count = 0
        *data, indices, weights = self.res_queue.get()
        if self.res_queue.qsize() + self._indices_queue.qsize() < self._queue_size:
            self._fetch_event.set()
        return (*data, indices, weights)
