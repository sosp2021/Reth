import traceback
from concurrent.futures import Executor, Future
from queue import Queue
from threading import Thread


def start_thread(fn, args=(), daemon=True, **thread_args):
    t = Thread(target=fn, args=args, daemon=daemon, **thread_args)
    t.start()
    return t


class DaemonExecutor(Executor):
    def __init__(self, max_workers=1):
        self.queue = Queue(5)
        self.threads = []
        for idx in range(max_workers):
            thread = Thread(
                target=self._worker, daemon=True, name=f"DaemonExecutor_{idx}"
            )
            thread.start()
            self.threads.append(thread)

    def _worker(self):
        while True:
            fut, fn, args, kwargs = self.queue.get()
            if not fut.set_running_or_notify_cancel():
                return
            try:
                res = fn(*args, **kwargs)
            except BaseException as exc:
                traceback.print_exc()
                fut.set_exception(exc)
            else:
                fut.set_result(res)

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        self.queue.put((fut, fn, args, kwargs))
        return fut
