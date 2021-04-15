import gc
import os
import tempfile

import numpy as np


class ShmBuffer:
    def __init__(self, path=None, root_dir=None, init_size=1024 * 1024, mode="w+"):
        assert mode in ["r+", "w+"]
        self.mode = mode
        if path is not None:
            self.path = path
        else:
            assert root_dir is not None
            self.path = tempfile.mktemp(
                dir=root_dir, prefix="shm_buffer_", suffix=".dat"
            )
        self.mmap = np.memmap(self.path, dtype=np.uint8, shape=(init_size,), mode=mode)

    def __del__(self):
        if self.mode == "w+":
            del self.mmap
            gc.collect()
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass

    def read(self, struct):
        # reload if resized
        totbytes = 0
        for dtype, shape in struct:
            totbytes += np.dtype((dtype, shape)).itemsize
        if totbytes > self.mmap.nbytes:
            self.mmap = np.memmap(
                self.path, dtype=np.uint8, shape=(totbytes,), mode="r+"
            )

        offset = 0
        res = []
        for dtype, shape in struct:
            itemsize = np.dtype((dtype, shape)).itemsize
            item = self.mmap[offset : offset + itemsize].view(dtype)
            item.shape = shape
            res.append(item)
            offset += itemsize
        return res

    def write(self, *data):
        totbytes = 0
        for x in data:
            assert isinstance(x, np.ndarray)
            totbytes += x.nbytes

        # resize
        if totbytes > len(self.mmap):
            self.mmap = np.memmap(
                self.path, dtype=np.uint8, shape=(totbytes,), mode="r+"
            )

        offset = 0
        for x in data:
            view = x.view(dtype=np.uint8)
            view.shape = -1
            self.mmap[offset : offset + len(view)] = view
            offset += len(view)

        return [(x.dtype.str, x.shape) for x in data]
