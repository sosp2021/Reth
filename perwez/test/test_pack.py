import secrets

import numpy as np

from perwez.client.pack import serialize, deserialize


def test_pack_value():
    data = secrets.token_bytes(1024 * 1024)
    res = deserialize(serialize(data))
    assert res == data
    shape = (20, 30, 40)
    dtype = np.dtype("f4")
    data = np.random.rand(*shape).astype(dtype)
    res = deserialize(serialize(data))
    assert isinstance(res, np.ndarray)
    assert res.shape == shape
    assert res.dtype == dtype


def test_pack_list():
    data = [secrets.token_bytes(1024 * 1024) for _ in range(5)]
    res = deserialize(serialize(data))
    assert isinstance(res, list)
    for i, x in enumerate(res):
        assert x == data[i]
    shape = (20, 30, 40)
    dtype = np.dtype("f4")
    data = [np.random.rand(*shape).astype(dtype) for _ in range(5)]
    res = deserialize(serialize(data))
    assert isinstance(res, list)
    for i, x in enumerate(res):
        assert np.array_equal(x, data[i])


def test_pack_compress():
    shape = (20, 30, 40)
    dtype = np.dtype("f4")
    data = np.ones(shape, dtype=dtype)
    res = serialize(data, compress=True)
    assert memoryview(res).nbytes < data.nbytes
    res = deserialize(res)
    assert np.array_equal(res, data)
