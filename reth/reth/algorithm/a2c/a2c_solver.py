import io

import gym
import numpy as np
import torch
import torch.nn.functional as F

from .a2c_model import generate_a2c_default_models
from ..algorithm import Algorithm
from ..util import transform_to_onehot, ensure_tensor


class A2CSolver(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        models=None,
        clip_value=40,
        learning_rate=0.01,
        hidden_size=64,
        device=None,
    ):
        super().__init__(device)

        if models is None:
            models = generate_a2c_default_models(
                observation_space,
                action_space,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
            )

        assert models["actor_network"]
        assert models["value_network"]
        assert models["actor_optimizer"]
        assert models["value_optimizer"]
        self.actor_network = models["actor_network"]
        self.value_network = models["value_network"]
        self.actor_optimizer = models["actor_optimizer"]
        self.value_optimizer = models["value_optimizer"]

        self.actor_network.to(self.device, non_blocking=True)
        self.value_network.to(self.device, non_blocking=True)

        assert isinstance(action_space, gym.spaces.Discrete)
        self.num_actions = action_space.n
        self.clip_value = clip_value

    def update(self, batch, weights=None):
        states, actions, rewards = batch
        actions = transform_to_onehot(actions, self.num_actions)

        actions = ensure_tensor(actions, dtype=torch.long, device=self.device)
        states = ensure_tensor(states, dtype=torch.long, device=self.device)
        rewards = ensure_tensor(rewards, dtype=torch.long, device=self.device)

        # train actor network
        self.actor_optimizer.zero_grad()
        log_softmax_actions = self.actor_network(states)
        vs = self.value_network(states).detach()
        qs = rewards
        advantages = qs - vs
        actor_network_loss = -torch.mean(
            torch.sum(log_softmax_actions * actions, 1) * advantages
        )
        actor_network_loss.backward()
        if self.clip_value >= 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_network.parameters(), self.clip_value
            )
        self.actor_optimizer.step()

        # train value network
        self.value_optimizer.zero_grad()
        target_values = qs
        values = self.value_network(states).squeeze()
        output = (target_values - values).detach().cpu().abs()
        value_network_loss = F.mse_loss(values, target_values)
        value_network_loss.backward()
        if self.clip_value >= 0:
            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(), self.clip_value
            )
        self.value_optimizer.step()

        return output

    def act(self, state):
        state = ensure_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

        log_softmax_action = self.actor_network(state)
        softmax_action = torch.exp(log_softmax_action)
        probs = softmax_action.cpu().data.numpy()
        return np.argmax(probs, axis=1).item()

    def load_weights(self, stream):
        states = torch.load(stream, map_location=self.device)
        self.actor_network.load_state_dict(states)
        self.actor_network.to(self.device, non_blocking=True)

    def save_weights(self, stream=None):
        if stream is None:
            stream = io.BytesIO()
        torch.save(self.actor_network.state_dict(), stream)
        return stream
