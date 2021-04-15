import time

import gym
import numpy as np

from ..algorithm import Algorithm
from ..buffer import DynamicSizeBuffer, NumpyBuffer
from ..utils import Interval, getLogger
from ..utils.exploration import BaseExploration, RandomExploration, OUNoiseExploration


class Worker:
    def __init__(
        self,
        env,
        solver,
        logger=getLogger("worker"),
        exploration=None,
        print_interval=500,
    ):
        assert isinstance(env, gym.Env)
        self.env = env
        assert isinstance(solver, Algorithm)
        self.solver = solver
        self.logger = logger

        # local variable
        self.s0 = self.env.reset()
        ## counter
        self.cur_step = 0
        self.cur_episode = 1
        self._start_time = None
        ## rewards
        self.recent_rewards = []
        # interval
        self.on_episode_end = []
        self.on_step_end = []
        if print_interval is not None:
            self.add_callback(self.print, print_interval)
        # exploration
        if exploration is None:
            self.exploration = None
        elif isinstance(exploration, (dict, int, str, float)):
            self._init_exploration_from_config(exploration)
        else:
            assert isinstance(exploration, BaseExploration)
            self.exploration = exploration

    def _init_exploration_from_config(self, config):
        if isinstance(config, (int, str, float)):
            self.exploration = RandomExploration(
                self.solver, self.env.action_space, epsilon=config
            )
            return

        name = config["name"].lower()
        kwargs = {k: config[k] for k in config if k != "name"}
        if name == "random":
            self.exploration = RandomExploration(
                self.solver, self.env.action_space, **kwargs
            )
        elif name == "ounoise":
            self.exploration = OUNoiseExploration(
                self.solver, self.env.action_space, **kwargs
            )
            self.on_episode_end.append(self.exploration.reset)
        else:
            raise Exception("Invalid exploration name")

    @property
    def cur_time(self):
        if self._start_time is None:
            return 0
        return time.monotonic() - self._start_time

    def add_callback(self, cb, interval):
        num, unit = self._parse_interval(interval)
        i = Interval(cb, num)
        if unit == "ts":
            self.on_step_end.append(i)
        elif unit == "e":
            self.on_episode_end.append(i)
        else:
            raise Exception(f"Invalid interval {interval}")

    def _parse_interval(self, interval):
        try:
            if isinstance(interval, int):
                return interval, "ts"
            elif isinstance(interval, str):
                if interval.endswith("ts"):
                    return int(interval[:-2]), "ts"
                elif interval.endswith("e"):
                    return int(interval[:-1]), "e"
        except ValueError:
            pass
        raise Exception(
            f"Invalid interval input {interval}, valid units: ts (timestep), e (episode)"
        )

    def load_weights(self, stream):
        self.solver.load_weights(stream)

    def save_weights(self, stream=None):
        return self.solver.save_weights(stream)

    def print(self):
        if self.logger:
            self.logger.info(
                f"ts: {self.cur_step}, episode: {self.cur_episode}, mean_reward: {np.mean(self.recent_rewards):.3f}, time: {self.cur_time:.2f}"
            )
        self.recent_rewards = []

    def evaluate(self, env=None):
        if env is None:
            env = self.env
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

    def step(self, action=None):
        if self._start_time is None:
            self._start_time = time.monotonic()

        self.cur_step += 1
        # act
        if action is None:
            if self.exploration is None:
                action = self.solver.act(self.s0)
            else:
                action = self.exploration.act(self.s0)
        # env
        s1, r, done, info = self.env.step(action)
        result = (self.s0, action, r, s1, done)
        self.s0 = s1

        if done:
            self.s0 = self.env.reset()
            if "episode" in info:
                self.recent_rewards.append(info["episode"]["r"])
            for cb in self.on_episode_end:
                cb()
            self.cur_episode += 1

        for cb in self.on_step_end:
            cb()

        return result

    def step_batch(self, batch_size):
        buffer = NumpyBuffer(batch_size, circular=False)

        for _ in range(batch_size):
            res = self.step()
            buffer.append(res)

        return buffer.data

    def step_episode(self):
        buffer = DynamicSizeBuffer(64)
        while True:
            res = self.step()
            buffer.append(res)
            *_, done = res
            if done:
                break
        return buffer.data
