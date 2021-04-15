import glob
import os
import signal

import zmq
import torch
import torch.multiprocessing as mp

from ..utils import init_lmdb, get_meta
from ..utils.pack import deserialize


def _rm_cuda_ipc_buffer():
    pid = os.getpid()
    files = glob.glob(f"/dev/shm/cuda.shm.*.{hex(pid)[2:]}*")
    for path in files:
        os.remove(path)


def _torch_cuda_worker(
    in_idx_queue, res_queue, tensor_buffer, lmdb_path, sampler_addrs
):

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.set_hwm(1)
    # sock.setsockopt(zmq.CONFLATE, 1)
    for addr in sampler_addrs:
        sock.connect(addr)
    lmdb_env = None
    stream = torch.cuda.Stream()
    try:
        while True:
            tensor_idx = in_idx_queue.get()
            indices, weights = deserialize(sock.recv())
            if lmdb_env is None:
                lmdb_env = init_lmdb(lmdb_path)
                # adopt map size change
                lmdb_env.set_mapsize(0)
            cuda_row, cuda_weight = tensor_buffer[tensor_idx]

            # init buffer
            pin_buffer = []
            for item in cuda_row:
                pin_buffer.append(
                    torch.empty(item.shape, dtype=item.dtype, pin_memory=True)
                )

            # adopt map size change
            lmdb_env.set_mapsize(0)
            with lmdb_env.begin(buffers=True) as txn:
                for row_i, idx in enumerate(indices):
                    row = txn.get(str(idx).encode())
                    row = deserialize(row)
                    for col_i, item in enumerate(row):
                        pin_buffer[col_i][row_i] = torch.from_numpy(item)
            with torch.cuda.stream(stream):
                event = torch.cuda.Event(interprocess=True)
                for i, pin_item in enumerate(pin_buffer):
                    cuda_row[i].copy_(pin_item, non_blocking=True)
                cuda_weight.copy_(torch.from_numpy(weights), non_blocking=True)
                event.record()
            res_queue.put((tensor_idx, indices, event.ipc_handle()))
    except KeyboardInterrupt:
        # silent exit
        _rm_cuda_ipc_buffer()


class TorchCudaLoader:
    def __init__(self, meta_addr, topic="default", buffer_size=8, num_procs=6):
        self.ctx = zmq.Context.instance()
        self.meta = get_meta(meta_addr)
        self.buffer_size = buffer_size
        self.num_procs = num_procs
        self.addrs = self.meta["sampler_info"][topic]["addrs"]
        self.lmdb_path = self.meta["lmdb_path"]

        self.mp_ctx = mp.get_context("spawn")

        self.workers = []
        self.tensor_buffer = []
        self.in_idx_queue = self.mp_ctx.SimpleQueue()
        for idx in range(buffer_size):
            self.in_idx_queue.put(idx)
        self.res_queue = self.mp_ctx.SimpleQueue()

        self.last_idx = None
        self._init_buffer()
        self._init_worker()
        # signal handler
        signal.signal(signal.SIGTERM, lambda *_: self.close())
        signal.signal(signal.SIGINT, lambda *_: self.close())

    def _init_buffer(self):

        sock = self.ctx.socket(zmq.PULL)
        sock.set_hwm(1)
        for addr in self.addrs:
            sock.connect(addr)
        res = sock.recv()
        sock.close(linger=0)
        indices, weights = deserialize(res)
        batch_size = len(indices)
        lmdb_env = init_lmdb(self.lmdb_path)
        lmdb_env.set_mapsize(0)
        with lmdb_env.begin(buffers=True) as txn:
            row = txn.get(str(indices[0]).encode())
            row = deserialize(row)
            struct = []
            for item in row:
                tensor_item = torch.from_numpy(item)
                struct.append((tensor_item.dtype, (batch_size, *tensor_item.shape)))
            tensor_weights = torch.from_numpy(weights)
            for _ in range(self.buffer_size):
                data = []
                for dtype, shape in struct:
                    data.append(torch.empty(shape, dtype=dtype, device="cuda"))
                weights_buf = torch.empty_like(tensor_weights, device="cuda")
                self.tensor_buffer.append((data, weights_buf))

    def _init_worker(self):
        for _ in range(self.num_procs):
            proc = self.mp_ctx.Process(
                target=_torch_cuda_worker,
                args=(
                    self.in_idx_queue,
                    self.res_queue,
                    self.tensor_buffer,
                    self.lmdb_path,
                    self.addrs,
                ),
                daemon=True,
            )
            proc.start()
            self.workers.append(proc)

    def close(self):
        _rm_cuda_ipc_buffer()
        for proc in self.workers:
            proc.terminate()
            proc.join()

    def __del__(self):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    def sample(self):
        if self.last_idx is not None:
            self.in_idx_queue.put(self.last_idx)
            self.last_idx = None
        tensor_idx, indices, event_handle = self.res_queue.get()
        event = torch.cuda.Event.from_ipc_handle(
            torch.cuda.current_device(), event_handle
        )
        event.wait()
        self.last_idx = tensor_idx
        data, weights = self.tensor_buffer[tensor_idx]
        return data, indices, weights
