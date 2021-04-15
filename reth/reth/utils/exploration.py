from abc import abstractmethod

import gym
import numpy as np

from .noise import OUNoise
from .schedule import Schedule


class BaseExploration:
    def __init__(self):
        pass

    @abstractmethod
    def act(self, state):
        pass


class RandomExploration(BaseExploration):
    def __init__(self, solver, action_space, epsilon=0):
        super().__init__()
        self.solver = solver
        self.action_space = action_space
        self.schedule = Schedule.from_str(epsilon)

    def act(self, state):
        eps = self.schedule.step()
        if np.random.rand() < eps:
            return self.action_space.sample()
        else:
            return self.solver.act(state)


class OUNoiseExploration(BaseExploration):
    def __init__(self, solver, action_space, epsilon, **noise_kwargs):
        super().__init__()
        self.solver = solver
        assert isinstance(action_space, gym.spaces.Box)
        assert len(action_space.shape) == 1
        self.action_space = action_space

        self.noise = OUNoise(self.action_space.shape[0], **noise_kwargs)
        self.schedule = Schedule.from_str(epsilon)

    def act(self, state):
        action = self.solver.act(state)
        eps = self.schedule.step()
        action += eps * self.noise.noise()
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def reset(self):
        self.noise.reset()
