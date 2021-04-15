import numpy as np
import zmq

from ..utils import ZMQ_DEFAULT_HWM, get_meta
from ..utils.pack import serialize


class Client:
    def __init__(self, meta_addr):
        self.ctx = zmq.Context.instance()
        self.meta = get_meta(meta_addr)

        self.append_sock = self.ctx.socket(zmq.PUSH)
        self.append_sock.set_hwm(ZMQ_DEFAULT_HWM)
        self.append_sock.connect(self.meta["append_addr"])

        self.update_sock = self.ctx.socket(zmq.PUSH)
        self.update_sock.set_hwm(ZMQ_DEFAULT_HWM)
        self.update_sock.connect(self.meta["update_addr"])

    def append(self, data, weights, compress=False):
        assert isinstance(data, (list, tuple))
        for col in data:
            assert isinstance(col, np.ndarray)
            assert len(col) == len(weights)
        batch_size = len(weights)
        # col to row
        res = []
        for i in range(batch_size):
            row = []
            for col in data:
                row.append(col[i, ...])
            res.append(serialize(row, compress=False))

        self.append_sock.send(serialize([res, weights], compress=compress))

    def update_priorities(self, indices, weights, step=True):
        assert len(indices) == len(weights)
        self.update_sock.send(serialize([indices, weights, step]))
