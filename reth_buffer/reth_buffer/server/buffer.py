import glob
import os

import numba
import numpy as np

from portalocker import Lock as FileLock, LOCK_EX, LOCK_SH


class _RLock:
    def __init__(self, outer_lock_path, inner_lock_path):
        self.outer_lock = FileLock(outer_lock_path, timeout=120, flags=LOCK_SH)
        self.inner_lock = FileLock(inner_lock_path, timeout=120, flags=LOCK_SH)

    def acquire(self):
        self.outer_lock.acquire()
        self.inner_lock.acquire()
        self.outer_lock.release()

    def release(self):
        self.inner_lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, type_, value, tb):
        self.release()


class _WLock:
    def __init__(self, outer_lock_path, inner_lock_path):
        self.outer_lock = FileLock(outer_lock_path, timeout=120, flags=LOCK_EX)
        self.inner_lock = FileLock(inner_lock_path, timeout=120, flags=LOCK_EX)

    def acquire(self):
        self.outer_lock.acquire()
        self.inner_lock.acquire()

    def release(self):
        self.inner_lock.release()
        self.outer_lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, type_, value, tb):
        self.release()


class BaseBuffer:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self._buffer_dir = os.path.join(root_dir, "buffer")
        self._lock_path = os.path.join(self._buffer_dir, "lock")
        self._lock2_path = os.path.join(self._buffer_dir, "lock2")
        os.makedirs(self._buffer_dir, exist_ok=True)

    def lock_sh(self):
        return _RLock(
            self._lock_path,
            self._lock2_path,
        )

    def lock_ex(self):
        return _WLock(
            self._lock_path,
            self._lock2_path,
        )


@numba.njit(nogil=True)
def _numba_select(buffers, indices):
    res = numba.typed.List()
    for buffer in buffers:
        res.append(buffer[indices])
    return res


class NumbaMemmapBuffer(BaseBuffer):
    def __init__(self, capacity, root_dir, struct=None):
        super().__init__(root_dir)
        self.capacity = capacity
        self._struct = struct

        self._buffers = None
        self._size = 0
        self._tail = -1

        if self._struct is not None:
            self._create_buffer()

    def _create_buffer(self):
        assert self._struct is not None
        self._buffers = numba.typed.List()
        for i, (dtype, shape) in enumerate(self._struct):
            buf = np.lib.format.open_memmap(
                os.path.join(self._buffer_dir, f"buffer-{i}.npy"),
                mode="w+",
                dtype=dtype,
                shape=(self.capacity, *shape),
            )
            byte_view = buf.view(dtype=np.uint8)
            byte_view.shape = (self.capacity, -1)
            self._buffers.append(byte_view)

    def _detect_struct(self, trans):
        self._struct = []
        for col in trans:
            try:
                data = np.asarray(col[0])
                self._struct.append((data.dtype.name, data.shape))
            except Exception:
                raise Exception("Invalid transition data")
        return self._struct

    def append(self, *trans):
        with self.lock_ex():
            if self._struct is None:
                self._detect_struct(trans)
                self._create_buffer()

            batch_size = len(trans[0])
            assert batch_size <= self.capacity
            self._size = min(self.size + batch_size, self.capacity)
            self._tail = (self.tail + 1) % self.capacity
            len1 = min(batch_size, self.capacity - self.tail)
            len2 = batch_size - len1
            for i, trans_col in enumerate(trans):
                temp_col = (
                    np.asarray(trans_col).view(dtype=np.uint8).reshape((batch_size, -1))
                )
                self._buffers[i][self.tail : self.tail + len1] = temp_col[:len1]
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

    def _reshape(self, data):
        return [
            data[i].view(dtype=dtype).reshape((-1, *shape))
            for i, (dtype, shape) in enumerate(self.struct)
        ]

    def select(self, indices):
        raw_data = _numba_select(self._buffers, indices)
        return self._reshape(raw_data)

    def clear(self):
        self._size = 0
        self._tail = -1

    @property
    def struct(self):
        return self._struct

    @property
    def tail(self):
        return self._tail

    @property
    def size(self):
        return self._size


class MemmapBuffer(BaseBuffer):
    def __init__(self, capacity, root_dir, struct=None):
        super().__init__(root_dir)
        self.capacity = capacity
        self._struct = struct

        self._buffers = None
        self._size = 0
        self._tail = -1

        if self._struct is not None:
            self._create_buffer()

    def _create_buffer(self):
        assert self._struct is not None
        self._buffers = [
            np.lib.format.open_memmap(
                os.path.join(self._buffer_dir, f"buffer-{i}.npy"),
                mode="w+",
                dtype=dtype,
                shape=(self.capacity, *shape),
            )
            for i, (dtype, shape) in enumerate(self._struct)
        ]

    def _detect_struct(self, trans):
        self._struct = []
        for col in trans:
            try:
                data = np.asarray(col[0])
                self._struct.append((data.dtype.name, data.shape))
            except Exception:
                raise Exception("Invalid transition data")
        return self._struct

    def append(self, *trans):
        with self.lock_ex():
            if self._struct is None:
                self._detect_struct(trans)
                self._create_buffer()

            batch_size = len(trans[0])
            assert batch_size <= self.capacity

            self._size = min(self.size + batch_size, self.capacity)
            self._tail = (self.tail + 1) % self.capacity
            len1 = min(batch_size, self.capacity - self.tail)
            len2 = batch_size - len1
            for i, trans_col in enumerate(trans):
                temp_col = np.asarray(trans_col)
                self._buffers[i][self.tail : self.tail + len1] = temp_col[:len1]
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

    def select(self, indices):
        return [buffer[indices] for buffer in self._buffers]

    def clear(self):
        self._size = 0
        self._tail = -1

    @property
    def struct(self):
        return self._struct

    @property
    def tail(self):
        return self._tail

    @property
    def size(self):
        return self._size


class ExternalMemmapBuffer(BaseBuffer):
    def __init__(self, root_dir):
        super().__init__(root_dir)

        files = sorted(glob.glob(os.path.join(self._buffer_dir, "buffer-*.npy")))
        self._buffers = [np.lib.format.open_memmap(path, mode="r+") for path in files]
        self._capacity = len(self._buffers[0])
        self._struct = [(x.dtype, x.shape[1:]) for x in self._buffers]

    def select(self, indices):
        return [buffer[indices] for buffer in self._buffers]

    @property
    def buffers(self):
        return self._buffers

    @property
    def struct(self):
        return self._struct

    @property
    def size(self):
        return self._capacity
