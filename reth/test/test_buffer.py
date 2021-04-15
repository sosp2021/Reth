import pytest

import numpy as np
from reth.buffer import NumpyBuffer, DynamicSizeBuffer, PrioritizedBuffer


def _generate_data(size):
    obs_space = (4, 84, 84)
    num_actions = 10
    result = []
    # idx
    result.append(np.arange(size, dtype="i8"))
    # s0
    result.append(np.random.rand(size, *obs_space).astype("f4"))
    # a
    result.append(np.random.randint(0, num_actions, size, dtype="i8"))
    # r
    result.append(np.random.rand(size).astype("f4"))
    # s1
    result.append(np.random.rand(size, *obs_space).astype("f4"))
    # done
    result.append(np.random.randint(0, 2, size).astype("f4"))

    return result


def test_buffer_detect_struct():
    capacity = 40
    buf = NumpyBuffer(capacity)
    data = _generate_data(20)
    buf.append_batch(data)
    assert buf.size == 20
    for i, (dtype, shape) in enumerate(buf.struct):
        assert np.dtype(dtype) == np.dtype(data[i].dtype)
        assert (20, *shape) == data[i].shape


def test_buffer_circular_append():
    capacity = 40
    buf = NumpyBuffer(capacity, circular=False)
    data = _generate_data(20)
    buf.append_batch(data)
    assert buf.size == 20
    buf.append_batch(data)
    assert buf.size == 40

    with pytest.raises(Exception):
        buf.append_batch(data)

    with pytest.raises(Exception):
        buf.append([col[0] for col in data])

    buf.clear()
    buf.append([col[0] for col in data])
    assert buf.size == 1


def test_buffer_resize():
    capacity = 40
    buf = NumpyBuffer(capacity, circular=False)
    data = _generate_data(capacity)
    buf.append_batch(data)

    with pytest.raises(Exception):
        buf.append_batch(data)

    buf.resize(capacity * 2)
    buf.append_batch(data)
    assert buf.size == capacity * 2
    for i, col in enumerate(buf.data):
        assert np.array_equal(col[:capacity], data[i])
        assert np.array_equal(col[capacity:], data[i])


def test_dynamic_size_buffer():
    buf = DynamicSizeBuffer(20)
    data = _generate_data(100)
    buf.append_batch(data)
    assert buf.size == 100
    assert buf.capacity >= 100

    buf = DynamicSizeBuffer(20)
    data = _generate_data(20)
    buf.append_batch(data)
    buf.append([col[0] for col in data])
    assert buf.size == 21
    assert buf.capacity >= 21


def test_buffer_random_sample():
    capacity = 20
    sample_size = 1000
    buf = DynamicSizeBuffer(capacity)
    data = _generate_data(capacity)
    buf.append_batch(data)
    res = buf.sample(sample_size)
    # type
    for i, col in enumerate(res):
        for j, line in enumerate(col):
            idx = res[0][j]
            assert np.array_equal(line, data[i][idx])
    # distribution
    cnt = [0 for _ in range(capacity)]
    for idx in res[0]:
        cnt[idx] += 1
    mean_cnt = sample_size / capacity
    l1_loss = 0
    for x in cnt:
        l1_loss += abs(x - mean_cnt) / mean_cnt
    assert l1_loss / capacity < 0.2


def test_per():
    capacity = 1000
    sample_size = 64
    buf = PrioritizedBuffer(capacity)
    data = _generate_data(capacity)
    weights = np.random.rand(capacity)
    buf.append_batch(data, weights=weights)
    res, indices, res_weights = buf.sample(sample_size)
    # check data
    for i, col in enumerate(res):
        for j, line in enumerate(col):
            idx = res[0][j]
            assert idx == indices[j]
            assert np.array_equal(line, data[i][idx])
    # check weight
    out_weights = (weights + 1e-6) ** buf.alpha.value()
    out_weights = (out_weights / min(out_weights)) ** (-buf.beta.value())
    for i, w in enumerate(res_weights):
        idx = indices[i]
        assert abs(out_weights[idx] - w) < 1e-3


def test_per_distribution():
    capacity = 100
    sample_size = 64
    sample_cnt = 100
    buf = PrioritizedBuffer(capacity)
    data = _generate_data(capacity)
    weights = np.random.rand(capacity)
    buf.append_batch(data, weights=weights)

    # distribution
    def _test_distribution(buf, weights):
        cnt = [0 for _ in range(capacity)]
        for _ in range(sample_cnt):
            _, indices, _ = buf.sample(sample_size)
            for idx in indices:
                cnt[idx] += 1

        sum_cnt = sum(cnt)
        out_weights = (weights + 1e-6) ** buf.alpha.value()
        sum_weights = sum(out_weights)

        l1_loss = 0
        for i, x in enumerate(cnt):
            a = x / sum_cnt
            b = out_weights[i] / sum_weights
            l1_loss += abs(a - b) / b
        assert l1_loss / capacity < 0.1

    _test_distribution(buf, weights)
    # update_priorities
    weights = np.random.rand(capacity)
    buf.update_priorities(np.arange(capacity), weights)
    _test_distribution(buf, weights)
