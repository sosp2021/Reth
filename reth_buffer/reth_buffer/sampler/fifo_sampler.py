from collections import deque

import numpy as np

from .base_sampler import BaseSampler


class FIFOSampler(BaseSampler):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def clear(self):
        self.buffer.clear()

    def ready_sample(self, batch_size):
        return len(self.buffer) > batch_size

    def sample(self, batch_size):
        res_indices = []
        res_weights = []
        for _ in range(batch_size):
            idx, weight = self.buffer.pop()
            res_indices.append(idx)
            res_weights.append(weight)
        return res_indices, res_weights

    def update(self, indices, weights):
        for i, idx in enumerate(indices):
            self.buffer.appendleft((idx, weights[i]))
