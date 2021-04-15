import time

import numpy as np

from ..algorithm import Algorithm
from ..utils import Interval, getLogger


class Trainer:
    def __init__(
        self,
        solver,
        logger=getLogger("trainer"),
        print_interval=300,
    ):
        assert isinstance(solver, Algorithm)
        self.solver = solver
        self.logger = logger

        self.cur_step = 0
        self.start_time = None

        self.on_step_end = []
        self.on_step_end.append(Interval(self.print, int(print_interval)))

        self.recent_errors = []

    @property
    def cur_time(self):
        if self.start_time is None:
            return 0
        return time.monotonic() - self.start_time

    def evaluate(self, env):
        s0 = env.reset()
        reward_sum = 0
        while True:
            a = self.solver.act(s0)
            s1, r, done, _ = env.step(a)
            reward_sum += r
            s0 = s1
            if done:
                break
        return reward_sum

    def print(self):
        if self.logger:
            self.logger.info(
                f"train_cnt: {self.cur_step}, mean_error: {np.mean(self.recent_errors):.3f}, time: {self.cur_time:.2f}"
            )
        self.recent_errors = []

    def load_weights(self, stream):
        self.solver.load_weights(stream)

    def save_weights(self, stream=None):
        return self.solver.save_weights(stream)

    def step(self, batch, **kwargs):
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.cur_step += 1

        td_error = self.solver.update(batch, **kwargs)

        # self.recent_errors.append(td_error[0])
        mean_error = np.mean(np.asarray(td_error))
        self.recent_errors.append(mean_error)

        for func in self.on_step_end:
            func()

        return td_error
