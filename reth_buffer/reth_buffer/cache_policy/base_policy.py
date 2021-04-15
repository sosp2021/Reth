from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def get_indices(self, batch_size):
        ...
