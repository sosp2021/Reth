import io

import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .ppo_model import generate_discrite_ppo_model
from ..algorithm import Algorithm
from ..util import ensure_tensor

# from ..util import transform_to_onehot


class PPOSolver(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        models=None,
        k_epochs=4,
        eps_clip=0.2,
        learning_rate=0.01,
        hidden_size=64,
        device=None,
    ):

        super().__init__(device)
        if models is None:
            models = generate_discrite_ppo_model(
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
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip

    def _evaluate_policy(self, state, action):
        action_probs = self.actor_network(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_network(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def update(self, batch, weights=None):
        states, actions, rewards, logprobs = batch
        # actions = transform_to_onehot(actions, self.config.action_space)

        actions = ensure_tensor(actions, torch.long, self.device)
        states = ensure_tensor(states, torch.float, self.device)
        rewards = ensure_tensor(rewards, torch.float, self.device)
        logprobs = ensure_tensor(logprobs, torch.float, self.device)

        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            new_logprobs, state_values, dist_entropy = self._evaluate_policy(
                states, actions
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(new_logprobs - logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * F.mse_loss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.actor_optimizer.zero_grad()
            self.value_network.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.value_optimizer.step()

        return loss.detach().cpu().abs()

    def act(self, state):
        state = ensure_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

        action_probs = self.actor_network(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        logprobs = dist.log_prob(action)
        return action.item(), logprobs.item()

    def sync_weights(self, src_ppo):
        assert isinstance(src_ppo, PPOSolver)
        self.actor_network.load_state_dict(src_ppo.actor_network.state_dict())
        self.value_network.load_state_dict(src_ppo.value_network.state_dict())

    def load_weights(self, stream):
        states = torch.load(stream, map_location=self.device)
        self.actor_network.load_state_dict(states)
        self.actor_network.to(self.device, non_blocking=True)

    def save_weights(self, stream=None):
        if stream is None:
            stream = io.BytesIO()
        torch.save(self.actor_network.state_dict(), stream)
        return stream
