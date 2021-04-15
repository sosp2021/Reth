from abc import abstractmethod

import torch


class Algorithm:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        print("Init model.")

    @abstractmethod
    def update(self, batch, weights=None):
        raise NotImplementedError

    @abstractmethod
    def act(self, state):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, stream):
        raise NotImplementedError

    @abstractmethod
    def save_weights(self, stream=None):
        raise NotImplementedError
