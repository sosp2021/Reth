import math
from collections import deque


class NStepAdder:
    def __init__(self, gamma, step=3):
        self.step = step
        self.gamma = gamma
        self._buffer = deque(maxlen=step)

    def push(self, s0, a, r, s1, done, *extra_args):
        res = None
        if self._buffer.maxlen == len(self._buffer):
            res = self._buffer.pop()
        t_gamma = self.gamma
        for item in self._buffer:
            # done check
            if item[4]:
                break
            # r
            item[2] += t_gamma * r
            t_gamma *= self.gamma
            # s1
            item[3] = s1

        row = [s0, a, r, s1, done, *extra_args]
        self._buffer.appendleft(row)
        return res
