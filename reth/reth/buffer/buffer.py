import numpy as np


class NumpyBuffer:
    def __init__(self, capacity, struct=None, circular=True):
        self._capacity = capacity
        self._struct = struct
        self.circular = circular

        self._buffers = None
        self._size = 0
        self._tail = -1

        if self._struct is not None:
            self._create_buffer()

    def _create_buffer(self):
        assert self._struct is not None
        self._buffers = [
            np.empty((self.capacity, *shape), dtype) for dtype, shape in self._struct
        ]

    def _detect_struct(self, trans):
        self._struct = []
        for col in trans:
            try:
                data = np.asarray(col)
                self._struct.append((data.dtype.name, data.shape))
            except Exception as e:
                raise Exception("Invalid transition data") from e
        return self._struct

    def resize(self, new_capacity):
        assert new_capacity > self.size
        self._capacity = new_capacity
        if self.struct is not None:
            old_buffers = self._buffers
            self._create_buffer()
            for i, new_buf in enumerate(self._buffers):
                new_buf[: self.size] = old_buffers[i][: self.size]

    def append(self, trans):
        if self._struct is None:
            self._detect_struct(trans)
            self._create_buffer()

        if not self.circular:
            assert self.size < self.capacity

        self._size = min(self.size + 1, self.capacity)
        self._tail = (self._tail + 1) % self.capacity
        for i, item in enumerate(trans):
            self._buffers[i][self._tail] = item

        return self._tail

    def append_batch(self, trans):
        if self._struct is None:
            self._detect_struct([col[0] for col in trans])
            self._create_buffer()

        batch_size = len(trans[0])
        if not self.circular:
            assert self.size + batch_size <= self.capacity
        else:
            assert batch_size <= self.capacity

        self._size = min(self.size + batch_size, self.capacity)
        self._tail = (self._tail + 1) % self.capacity
        len1 = min(batch_size, self.capacity - self._tail)
        len2 = batch_size - len1
        for i, trans_col in enumerate(trans):
            temp_col = np.asarray(trans_col)
            self._buffers[i][self._tail : self._tail + len1] = temp_col[:len1]
            if len2 > 0:
                self._buffers[i][:len2] = temp_col[len1:]
        if len2 == 0:
            indices = np.arange(self._tail, self._tail + len1)
            self._tail = self._tail + len1 - 1
        else:
            indices = np.concatenate(
                (np.arange(self._tail, self._tail + len1), np.arange(len2))
            )
            self._tail = len2 - 1

        return indices

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size)
        return self.select(indices)

    def select(self, indices):
        return [buffer[indices] for buffer in self._buffers]

    def clear(self):
        self._size = 0
        self._tail = -1

    @property
    def data(self):
        return [buffer[: self.size] for buffer in self._buffers]

    @property
    def capacity(self):
        return self._capacity

    @property
    def struct(self):
        return self._struct

    @property
    def size(self):
        return self._size


class DynamicSizeBuffer(NumpyBuffer):
    def __init__(self, init_capacity=64, struct=None):
        super().__init__(init_capacity, struct, False)

    def append(self, trans):
        if self.size == self.capacity:
            self.resize(2 * self.capacity)
        super().append(trans)

    def append_batch(self, trans):
        batch_size = len(trans[0])
        target_capacity = self.capacity
        while batch_size + self.size > target_capacity:
            target_capacity *= 2
        self.resize(target_capacity)
        super().append_batch(trans)
