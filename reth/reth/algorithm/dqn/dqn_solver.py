import io

import gym
import torch
import torch.nn.functional as F

from reth.utils import Interval

from .dqn_model import generate_dqn_default_models
from .. import Algorithm
from ..util import ensure_tensor


class DQNSolver(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        models=None,
        gamma=0.99,
        clip_value=40,
        double_q=True,
        dueling=True,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        update_target_interval=150,
        device=None,
        n_step=1,
    ):
        super().__init__(device)

        if models is None:
            models = generate_dqn_default_models(
                observation_space,
                action_space,
                dueling=dueling,
                learning_rate=learning_rate,
                adam_epsilon=adam_epsilon,
            )

        assert models["q_network"]
        assert models["target_q_network"]
        assert models["optimizer"]
        self.q_network = models["q_network"]
        self.target_q_network = models["target_q_network"]
        self.optimizer = models["optimizer"]

        self.q_network.to(self.device, non_blocking=True)
        self.target_q_network.to(self.device, non_blocking=True)
        self.update_target()

        assert isinstance(action_space, gym.spaces.Discrete)
        self.num_actions = action_space.n
        self.clip_value = clip_value
        self.double_q = double_q
        self.gamma = gamma
        self.n_step = n_step
        if update_target_interval is not None:
            self._update_target_interval = Interval(
                self.update_target, update_target_interval
            )
        else:
            self._update_target_interval = None

    def update_target(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _calc_td_error(self, batch):
        s0, a, r, s1, done = batch

        s0 = ensure_tensor(s0, torch.float, self.device)
        a = ensure_tensor(a, torch.long, self.device)
        r = ensure_tensor(r, torch.float, self.device)
        s1 = ensure_tensor(s1, torch.float, self.device)
        done = ensure_tensor(done, torch.float, self.device)

        q_values = self.q_network(s0)
        next_q_values = self.target_q_network(s1)
        one_hot_selection = F.one_hot(a, self.num_actions).float()

        q_values = torch.sum(q_values * one_hot_selection, 1)

        if self.double_q:
            next_q_values_online = self.q_network(s1)
            max_next_q_values_action = torch.argmax(next_q_values_online, 1)
            best_one_hot_selection = F.one_hot(
                max_next_q_values_action, self.num_actions
            ).float()
            next_q_best = torch.sum(next_q_values * best_one_hot_selection, 1)
        else:
            best_one_hot_selection = F.one_hot(
                torch.argmax(next_q_values, 1), self.num_actions
            ).float()
            next_q_best = torch.sum(next_q_values * best_one_hot_selection, 1)

        expected_q_value = r + (self.gamma ** self.n_step) * next_q_best * (1 - done)
        td_error = q_values - expected_q_value.detach()
        return td_error

    def calc_loss(self, batch):
        td_error = self._calc_td_error(batch)
        return td_error.detach().cpu().abs()

    def update(self, batch, weights=None):
        if weights is not None:
            weights = ensure_tensor(weights, torch.float, self.device)

        td_error = self._calc_td_error(batch)
        out_td_error = td_error.detach().cpu().abs()

        # compute the error (potentially clipped)
        loss = F.smooth_l1_loss(td_error, torch.zeros_like(td_error), reduction="none")
        if weights is not None:
            loss *= weights
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_value >= 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clip_value)
        self.optimizer.step()

        if self._update_target_interval is not None:
            self._update_target_interval()
        return out_td_error

    def act(self, state):
        state = ensure_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

        q_value = self.q_network.forward(state)
        action_index = torch.argmax(q_value, dim=1)
        return action_index.item()

    def load_weights(self, stream):
        states = torch.load(stream, map_location=self.device)
        self.q_network.load_state_dict(states)
        self.q_network.to(self.device, non_blocking=True)
        self.update_target()

    def save_weights(self, stream=None):
        if stream is None:
            stream = io.BytesIO()
        torch.save(self.q_network.state_dict(), stream)
        return stream
