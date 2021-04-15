import numpy as np

from .base_sampler import BaseSampler


class UniformSampler(BaseSampler):
    def __init__(self, capacity):
        self.indices = np.zeros(capacity, dtype="i8")
        self.capacity = capacity
        self.tail = 0

    def sample(self, batch_size):
        sample = np.random.choice(self.tail, batch_size)
        return self.indices[sample], np.ones(batch_size, dtype="i8")

    def update(self, indices, weights):
        if self.tail == self.capacity:
            return
        for idx in indices:
            self.indices[self.tail] = idx
            self.tail = min(self.tail + 1, self.capacity)

    def clear(self):
        self.indices = np.zeros(self.capacity, dtype="i8")
        self.tail = 0
