import time

import lz4.frame
import msgpack
import numpy as np
from loguru import logger

KEY_COMPRESS = "reth_compress_953531b51ab8"
KEY_NUMPY = "reth_numpy_356b759ef2b3"
KEY_BYTES = "reth_bytes_a32b00c47e72"


class MessageBody:
    def __init__(self):
        self.offset = 0
        self.data = []

    def append(self, data):
        view = memoryview(data)
        res = (self.offset, view.nbytes)
        self.offset += view.nbytes
        self.data.append(view)
        return res


def _serialize_obj(cur, body):
    if isinstance(cur, (list, tuple)):
        res = []
        for item in cur:
            item_res = _serialize_obj(item, body)
            res.append(item_res)
        return res
    elif isinstance(cur, dict):
        res = {}
        for key, val in cur.items():
            val_res = _serialize_obj(val, body)
            res[key] = val_res
        return res
    elif isinstance(cur, np.ndarray):
        res = {
            KEY_NUMPY: True,
            "header": np.lib.format.header_data_from_array_1_0(cur),
            "raw_size": cur.nbytes,
        }
        offset, length = body.append(cur.data)
        res["offset"] = offset
        res["length"] = length
        return res
    elif isinstance(cur, (bytes, bytearray, memoryview)):
        view = memoryview(cur)
        res = {KEY_BYTES: True, "raw_size": view.nbytes}
        offset, length = body.append(cur)
        res["offset"] = offset
        res["length"] = length
        return res
    else:
        return cur


def serialize(data, compress=False):
    body = MessageBody()
    meta = _serialize_obj(data, body)
    if compress:
        data_buf = bytearray(body.offset)
        offset = 0
        for item in body.data:
            data_buf[offset : offset + item.nbytes] = item
            offset += item.nbytes
        compressed_buf = lz4.frame.compress(data_buf, return_bytearray=True)
        header = {
            KEY_COMPRESS: True,
            "meta": meta,
            "body_len": len(compressed_buf),
            "time": time.time(),
        }

        header = msgpack.packb(header)
        header_len = len(header)
        buf_len = 4 + header_len + len(compressed_buf)
        buf = bytearray(buf_len)
        buf[:4] = header_len.to_bytes(4, "big")
        buf[4 : 4 + header_len] = header
        buf[4 + header_len :] = compressed_buf
        return buf
    else:
        header = {"meta": meta, "body_len": body.offset, "time": time.time()}
        header = msgpack.packb(header)
        header_len = len(header)
        buf_len = 4 + header_len + body.offset
        buf = bytearray(buf_len)
        buf[:4] = header_len.to_bytes(4, "big")
        buf[4 : 4 + header_len] = header
        offset = 4 + header_len
        for item in body.data:
            buf[offset : offset + item.nbytes] = item
            offset += item.nbytes

        return buf


# https://github.com/numpy/numpy/blob/d7a75e8e8fefc433cf6e5305807d5f3180954273/numpy/lib/format.py#L751
def rebuild_ndarray(data, header):
    dtype = np.lib.format.descr_to_dtype(header["descr"])
    shape = header["shape"]
    fortran_order = header["fortran_order"]
    array = np.frombuffer(data, dtype=dtype)
    if fortran_order:
        array.shape = shape[::-1]
        array = array.transpose()
    else:
        array.shape = shape
    return array


def _deserialize_obj(cur, body_view):
    if isinstance(cur, (list, tuple)):
        res = []
        for item in cur:
            res.append(_deserialize_obj(item, body_view))
        return res
    elif isinstance(cur, dict):
        if KEY_NUMPY in cur:
            data = body_view[cur["offset"] : cur["offset"] + cur["length"]]
            header = cur["header"]
            res = rebuild_ndarray(data, header)
            assert res.nbytes == cur["raw_size"]
            return res
        elif KEY_BYTES in cur:
            data = body_view[cur["offset"] : cur["offset"] + cur["length"]]
            assert memoryview(data).nbytes == cur["raw_size"]
            return data
        else:
            res = {}
            for key, val in cur.items():
                res[key] = _deserialize_obj(val, body_view)
            return res
    else:
        return cur


def read_header(data):
    view = memoryview(data)
    header_len = int.from_bytes(view[:4], "big")
    header = msgpack.unpackb(view[4 : 4 + header_len])
    return header


def deserialize(data, verbose=False):
    view = memoryview(data)
    header_len = int.from_bytes(view[:4], "big")
    header = msgpack.unpackb(view[4 : 4 + header_len])
    body_view = view[4 + header_len :]
    assert body_view.nbytes == header["body_len"]
    if header.get(KEY_COMPRESS):
        lz4_res = lz4.frame.decompress(body_view, return_bytearray=True)
        body_view = memoryview(lz4_res)
    meta = header["meta"]
    res = _deserialize_obj(meta, body_view)
    if verbose:
        logger.info(
            f"latency: {time.time() - header['time']}, len: {header['body_len']}",
            flush=True,
        )
    return res
