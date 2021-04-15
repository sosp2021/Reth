import queue

from .prioritized_buffer import PrioritizedBuffer
from ..utils import QUEUE_SIZE_PER_THREAD
from ..utils.thread import start_thread


class BufferService:
    def __init__(self, root_dir, batch_size, buffer_config=None, sample_max_threads=4):
        if buffer_config is None:
            buffer_config = {}
        self.batch_size = batch_size
        self.buffer = PrioritizedBuffer(root_dir=root_dir, **buffer_config)
        self.sample_queue = queue.Queue(sample_max_threads * QUEUE_SIZE_PER_THREAD)
        self.samplers = [start_thread(self._worker_sample, daemon=True)]
        self.sample_max_threads = sample_max_threads
        self.sample_fail_count = 0
        self.sample_add_worker_threshold = 10

    def _worker_sample(self):
        while True:
            self.buffer.sample_start_event.wait()
            indices, weights = self.buffer.sample(self.batch_size)
            self.sample_queue.put((indices, weights))

    def _sample_fail(self):
        self.sample_fail_count += 1
        if (
            self.sample_fail_count > self.sample_add_worker_threshold
            and len(self.samplers) < self.sample_max_threads
        ):
            self.samplers.append(start_thread(self._worker_sample, daemon=True))

    def append(self, *data, weights=None):
        self.buffer.append(*data, weights=weights)

    def sample(self):
        if self.sample_queue.empty():
            self._sample_fail()
        return self.sample_queue.get()

    def batch_sample(self, max_size=5):
        res = []
        for _ in range(max_size):
            try:
                item = self.sample_queue.get_nowait()
                res.append(item)
            except queue.Empty:
                break
        if len(res) == 0:
            self._sample_fail()

        return res

    def update_priorities(self, indices, weights):
        self.buffer.update_priorities(indices, weights)

    def is_sample_started(self):
        return self.buffer.sample_start_event.is_set()
