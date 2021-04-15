import time
from contextlib import contextmanager

import lz4.frame
import msgpack
import numpy as np


@contextmanager
def serialize_data(data, compression=None, packer=None, **compression_args):
    flag_value = False
    if not isinstance(data, (list, tuple)):
        flag_value = True
        data = [data]

    # generate data
    res = []
    for item in data:
        if isinstance(item, np.ndarray):
            assert not item.dtype.hasobject
            res.append(
                {
                    **np.lib.format.header_data_from_array_1_0(item),
                    "data": item.data,
                    "size": item.nbytes,
                    "type": "numpy",
                    "ts": time.monotonic(),
                }
            )
        else:
            view = memoryview(item)
            res.append(
                {
                    "data": view,
                    "size": len(view),
                    "type": "bytes",
                    "ts": time.monotonic(),
                }
            )
    if flag_value:
        res[0]["value_type"] = True

    # compression
    if compression == "lz4":
        for item in res:
            item["compression"] = "lz4"
            item["data"] = lz4.frame.compress(
                item["data"], return_bytearray=True, **compression_args
            )
    else:
        assert compression is None

    # pack
    if packer is None:
        packer = msgpack.Packer(autoreset=False)
    packer.reset()
    packer.pack(res)
    try:
        yield packer.getbuffer()
    finally:
        packer.reset()


def deserialize_data(data, return_meta=False):
    data = msgpack.unpackb(data)
    # compression
    for item in data:
        if "compression" in item:
            assert item["compression"] == "lz4"
            item["data"] = lz4.frame.decompress(item["data"], return_bytearray=True)
            del item["compression"]
    # check size
    for item in data:
        assert len(item["data"]) == item["size"]
    # numpy
    # https://github.com/numpy/numpy/blob/d7a75e8e8fefc433cf6e5305807d5f3180954273/numpy/lib/format.py#L751
    for item in data:
        if item.get("type") == "numpy":
            dtype = np.lib.format.descr_to_dtype(item["descr"])
            shape = item["shape"]
            fortran_order = item["fortran_order"]
            array = np.frombuffer(item["data"], dtype=dtype)
            if fortran_order:
                array.shape = shape[::-1]
                array = array.transpose()
            else:
                array.shape = shape
            item["data"] = array

    if return_meta:
        res = data
    else:
        res = [x["data"] for x in data]

    if len(data) == 1 and data[0].get("value_type"):
        return res[0]
    else:
        return res
