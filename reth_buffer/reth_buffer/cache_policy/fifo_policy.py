import numpy as np

from .base_policy import BasePolicy


class FIFOPolicy(BasePolicy):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tail = 0

    def get_indices(self, batch_size):
        assert batch_size <= self.capacity
        res = np.zeros(batch_size, dtype="i4")
        for i in range(batch_size):
            res[i] = self.tail
            self.tail = (self.tail + 1) % self.capacity

        return res
