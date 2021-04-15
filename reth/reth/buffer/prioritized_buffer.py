import numpy as np

from .buffer import NumpyBuffer
from .sumtree import NumbaSumTree
from ..utils import Schedule


class PrioritizedBuffer:
    def __init__(
        self,
        capacity=50000,
        alpha=0.6,
        beta=0.4,
        struct=None,
    ):
        self.alpha = Schedule.from_str(alpha)
        self._alpha_str = alpha
        self.beta = Schedule.from_str(beta)
        self._beta_str = beta
        self._eps = 1e-6

        self.buffer = NumpyBuffer(capacity, struct=struct)
        self.sumtree = NumbaSumTree(capacity)

    def _normalize_weights(self, weights):
        weights = np.asarray(weights)
        return (weights + self._eps) ** self.alpha.value()

    def append(self, data, weight=None):
        if weight is None:
            weight = 1
        index = self.buffer.append(data)
        self.sumtree.update(np.asarray(index), np.asarray(weight))

    def append_batch(self, data, weights=None):
        if weights is None:
            weights = np.ones(len(data[0]))
        else:
            assert len(weights) == len(data[0])
            weights = self._normalize_weights(weights)
        indices = self.buffer.append_batch(data)
        self.sumtree.update(indices, weights)

    def sample(self, batch_size):
        assert batch_size <= self.size
        indices, weights = self.sumtree.sample(batch_size)

        self.alpha.step()
        self.beta.step()
        weights = (weights / self.sumtree.min()) ** (-self.beta.value())
        data = self.buffer.select(indices)

        return data, indices, weights

    def update_priorities(self, indices, weights):
        self.sumtree.update(indices, self._normalize_weights(weights))

    def clear(self):
        self.alpha = Schedule.from_str(self._alpha_str)
        self.beta = Schedule.from_str(self._beta_str)
        self.buffer.clear()
        self.sumtree.clear()

    @property
    def capacity(self):
        return self.buffer.capacity

    @property
    def size(self):
        return self.buffer.size

    @property
    def struct(self):
        return self.buffer.struct
