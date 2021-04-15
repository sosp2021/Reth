from .base_sampler import BaseSampler
from ..utils import NumbaSumTree, Schedule


class PERSampler(BaseSampler):
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.sumtree = NumbaSumTree(capacity)

        self.alpha = Schedule.from_str(alpha)
        self._alpha_str = alpha
        self.beta = Schedule.from_str(beta)
        self._beta_str = beta

        self._eps = 1e-6

    def _normalize_weights(self, weights):
        return (weights + self._eps) ** self.alpha.value()

    def clear(self):
        self.sumtree.clear()
        self.alpha = Schedule.from_str(self._alpha_str)
        self.beta = Schedule.from_str(self._beta_str)

    def sample(self, batch_size):
        indices, weights = self.sumtree.sample(batch_size)
        weights = (weights / self.sumtree.min()) ** (-self.beta.value())

        return indices, weights

    def on_step(self):
        self.alpha.step()
        self.beta.step()

    def update(self, indices, weights):
        self.sumtree.update(indices, self._normalize_weights(weights))
