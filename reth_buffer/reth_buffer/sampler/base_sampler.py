from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def ready_sample(self, batch_size):
        return True

    def on_step(self):
        pass

    @abstractmethod
    def sample(self, batch_size):
        ...

    @abstractmethod
    def update(self, indices, weights):
        ...

    @abstractmethod
    def clear(self):
        ...
