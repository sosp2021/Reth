import secrets

import numpy as np

from perwez.client.pack import serialize_data, deserialize_data


def test_pack_value():
    data = secrets.token_bytes(1024 * 1024)
    with serialize_data(data) as packed:
        res = deserialize_data(packed)
        assert res == data
    shape = (20, 30, 40)
    dtype = np.dtype("f4")
    data = np.random.rand(*shape).astype(dtype)
    with serialize_data(data) as packed:
        res = deserialize_data(packed)
        assert isinstance(res, np.ndarray)
        assert res.shape == shape
        assert res.dtype == dtype


def test_pack_list():
    data = [secrets.token_bytes(1024 * 1024) for _ in range(5)]
    with serialize_data(data) as packed:
        res = deserialize_data(packed)
        assert isinstance(res, list)
        for i, x in enumerate(res):
            assert x == data[i]
    shape = (20, 30, 40)
    dtype = np.dtype("f4")
    data = [np.random.rand(*shape).astype(dtype) for _ in range(5)]
    with serialize_data(data) as packed:
        res = deserialize_data(packed)
        assert isinstance(res, list)
        for i, x in enumerate(res):
            assert np.array_equal(x, data[i])


def test_pack_compress():
    data = b"".join([b"asd" for _ in range(100)])
    with serialize_data(data, compression="lz4") as packed:
        assert len(packed) < len(data)
        res = deserialize_data(packed)
        assert res == data
    shape = (20, 30, 40)
    dtype = np.dtype("f4")
    data = np.ones(shape, dtype=dtype)
    with serialize_data(data, compression="lz4") as packed:
        assert len(packed) < data.nbytes
        res = deserialize_data(packed)
        assert np.array_equal(res, data)
