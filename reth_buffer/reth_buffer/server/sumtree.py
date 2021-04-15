import os

import numba
import numpy as np

from readerwriterlock import rwlock


@numba.njit(nogil=True)
def _numba_maintain_node(tree, idx):
    capacity, _sum, _min, _val = tree
    left = idx * 2 + 1
    right = idx * 2 + 2
    sum_res = _val[idx]
    min_res = _val[idx] if _val[idx] != 0 else 1
    if left < capacity:
        sum_res += _sum[left]
        if _min[left] != 0:
            min_res = min(min_res, _min[left])
    if right < capacity:
        sum_res += _sum[right]
        if _min[right] != 0:
            min_res = min(min_res, _min[right])
    _sum[idx] = sum_res
    _min[idx] = min_res


@numba.njit(nogil=True)
def _numba_maintain(tree, idx):
    cur = idx
    while True:
        _numba_maintain_node(tree, cur)
        if cur == 0:
            break
        cur = (cur - 1) // 2


@numba.njit(nogil=True)
def _numba_update(tree, indices, weights):
    assert len(indices) == len(weights)
    capacity, _sum, _min, _val = tree
    for i, idx in enumerate(indices):
        val = weights[i]
        assert idx < capacity and idx >= 0
        assert val != 0
        _val[idx] = val
        _numba_maintain(tree, idx)


@numba.njit(nogil=True)
def _numba_find_index(tree, weight):
    capacity, _sum, _min, _val = tree
    # assert weight < _sum[0]
    cur = 0
    while True:
        left = cur * 2 + 1
        right = cur * 2 + 2
        # left
        if left < capacity:
            if weight < _sum[left]:
                cur = left
                continue
            else:
                weight -= _sum[left]
        # center
        if weight < _val[cur] + 1e-5:
            return cur
        else:
            weight -= _val[cur]
        # right
        if right >= capacity:
            return cur
        # assert right < self.capacity
        cur = right


@numba.njit(nogil=True)
def _numba_sample(tree, batch_size):
    _, _sum, _min, _val = tree
    assert batch_size > 0
    segment = _sum[0] / batch_size
    targets = (np.arange(batch_size) + np.random.random_sample(batch_size)) * segment
    res = np.zeros(batch_size, dtype=np.int64)
    for i, weight in enumerate(targets):
        res[i] = _numba_find_index(tree, weight)
    return res, _val[res]


class NumbaSumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self._sum = np.zeros(capacity, dtype=np.float64)
        self._min = np.zeros(capacity, dtype=np.float64)
        self._val = np.zeros(capacity, dtype=np.float64)
        self._lock = rwlock.RWLockWrite()

    @property
    def _tree(self):
        return (self.capacity, self._sum, self._min, self._val)

    def clear(self):
        with self._lock.gen_wlock():
            self._sum.fill(0)
            self._min.fill(0)
            self._val.fill(0)

    def update(self, indices, weights):
        with self._lock.gen_wlock():
            _numba_update(self._tree, indices, weights)

    def sum(self):
        return self._sum[0]

    def min(self):
        return self._min[0] if self._min[0] != 0 else 1

    def find_index(self, weight):
        with self._lock.gen_rlock():
            return _numba_find_index(self._tree, weight)

    def sample(self, batch_size):
        with self._lock.gen_rlock():
            return _numba_sample(self._tree, batch_size)


class NumbaMemmapSumTree:
    def __init__(self, capacity, root_dir):
        self.capacity = capacity
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir

        self._sum = np.memmap(
            os.path.join(self.root_dir, "sumtree-sum.dat"),
            mode="w+",
            dtype=np.float64,
            shape=capacity,
        )
        self._sum.fill(0)
        self._min = np.memmap(
            os.path.join(self.root_dir, "sumtree-min.dat"),
            mode="w+",
            dtype=np.float64,
            shape=capacity,
        )
        self._min.fill(0)
        self._val = np.memmap(
            os.path.join(self.root_dir, "sumtree-val.dat"),
            mode="w+",
            dtype=np.float64,
            shape=capacity,
        )
        self._val.fill(0)
        self._lock = rwlock.RWLockWrite()

    @property
    def _tree(self):
        return (self.capacity, self._sum, self._min, self._val)

    def clear(self):
        with self._lock.gen_wlock():
            self._sum.fill(0)
            self._min.fill(0)
            self._val.fill(0)

    def update(self, indices, weights):
        with self._lock.gen_wlock():
            _numba_update(self._tree, indices, weights)

    def sum(self):
        return self._sum[0]

    def min(self):
        return self._min[0]

    def find_index(self, weight):
        with self._lock.gen_rlock():
            return _numba_find_index(self._tree, weight)

    def sample(self, batch_size):
        with self._lock.gen_rlock():
            return _numba_sample(self._tree, batch_size)
