import numpy as np
import zmq

from ..utils import init_lmdb, get_meta
from ..utils.pack import deserialize


class NumpyLoader:
    def __init__(self, meta_addr, topic="default"):
        self.ctx = zmq.Context.instance()
        self.meta = get_meta(meta_addr)
        self.sampler_info = self.meta["sampler_info"][topic]
        self.lmdb_env = None

        self.sample_sock = self.ctx.socket(zmq.PULL)
        self.sample_sock.set_hwm(1)
        # self.sample_sock.setsockopt(zmq.CONFLATE, 1)
        for addr in self.sampler_info["addrs"]:
            self.sample_sock.connect(addr)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    def sample(self):
        indices, weights = self.get_indices()
        batch_size = len(indices)
        data = None
        if self.lmdb_env is None:
            self.lmdb_env = init_lmdb(self.meta["lmdb_path"])
            # adopt map size
            self.lmdb_env.set_mapsize(0)
        with self.lmdb_env.begin(buffers=True) as txn:
            for i, idx in enumerate(indices):
                row = txn.get(str(idx).encode())
                row = deserialize(row)
                # init data
                if data is None:
                    assert isinstance(row, (list, tuple))
                    for item in row:
                        assert isinstance(item, np.ndarray)
                    data = []
                    for item in row:
                        data.append(
                            np.empty((batch_size, *item.shape), dtype=item.dtype)
                        )
                for col_idx, item in enumerate(row):
                    data[col_idx][i] = item
        return data, indices, weights

    def get_indices(self):
        res = self.sample_sock.recv()
        indices, weights = deserialize(res)
        return indices, weights
