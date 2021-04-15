import io

import gym
import numpy as np
import torch

from .pg_model import generate_pg_default_models
from ..algorithm import Algorithm
from ..util import ensure_tensor


class PGSolver(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        models=None,
        learning_rate=5e-5,
        device=None,
    ):

        super().__init__(device)

        if models is None:
            models = generate_pg_default_models(
                observation_space, action_space, learning_rate
            )

        assert models["policy_model"]
        assert models["optimizer"]
        self.policy = models["policy_model"]
        self.optimizer = models["optimizer"]
        self.policy.to(self.device, non_blocking=True)

        assert isinstance(action_space, gym.spaces.Discrete)
        self.num_actions = action_space.n

    def update(self, batch, weights=None):
        s0, a, r = batch

        s0 = ensure_tensor(s0, torch.float, self.device)
        a = ensure_tensor(a, torch.long, self.device)
        r = ensure_tensor(r, torch.float, self.device)

        self.optimizer.zero_grad()

        log_probs = torch.log(self.policy(s0))
        selected_log_probs = r * log_probs[np.arange(len(a)), a]
        loss = -selected_log_probs.mean()
        out_loss = selected_log_probs.detach().cpu().abs()

        loss.backward()
        self.optimizer.step()

        return out_loss

    def act(self, state):
        state = ensure_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

        probs = self.policy(state).detach().cpu().numpy()
        action = np.argmax(probs, axis=1)

        return action.item()

    def load_weights(self, stream):
        states = torch.load(stream, map_location=self.device)
        self.policy.load_state_dict(states)
        self.policy.to(self.device, non_blocking=True)

    def save_weights(self, stream=None):
        if stream is None:
            stream = io.BytesIO()
        torch.save(self.policy.state_dict(), stream)
        return stream
