import numba
import numpy as np


@numba.njit
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


@numba.njit
def _numba_maintain(tree, idx):
    cur = idx
    while True:
        _numba_maintain_node(tree, cur)
        if cur == 0:
            break
        cur = (cur - 1) // 2


@numba.njit
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


@numba.njit
def _numba_update(tree, indices, weights):
    assert len(indices) == len(weights)
    _, _sum, _min, _val = tree
    for i, idx in enumerate(indices):
        _val[idx] = weights[i]
        _numba_maintain(tree, idx)


@numba.njit
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

    @property
    def _tree(self):
        return (
            self.capacity,
            self._sum,
            self._min,
            self._val,
        )

    def clear(self):
        self._sum.fill(0)
        self._min.fill(0)
        self._val.fill(0)

    def update(self, indices, weights):
        _numba_update(self._tree, indices, weights)

    def sum(self):
        return self._sum[0]

    def min(self):
        return self._min[0] if self._sum[0] != 0 else 1

    def sample(self, batch_size):
        return _numba_sample(self._tree, batch_size)
