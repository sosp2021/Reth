import io

import gym
import torch

from .ddpg_model import generate_ddpg_model
from ..algorithm import Algorithm
from ..util import soft_update, ensure_tensor


class DDPGSolver(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        models=None,
        gamma=0.99,
        tau=0.001,
        learning_rate=None,
        learning_rate_actor=0.0001,
        learning_rate_critic=0.001,
        is_image=None,
        device=None,
    ):
        super().__init__(device)
        if learning_rate_actor is None:
            learning_rate_actor = learning_rate
        if learning_rate_critic is None:
            learning_rate_critic = learning_rate
        if models is None:
            models = generate_ddpg_model(
                observation_space,
                action_space,
                learning_rate_actor,
                learning_rate_critic,
                is_image,
            )

        assert models["actor"]
        assert models["actor_target"]
        assert models["actor_optimizer"]
        assert models["critic"]
        assert models["critic_target"]
        assert models["critic_optimizer"]
        self.actor = models["actor"]
        self.actor_target = models["actor_target"]
        self.actor_optimizer = models["actor_optimizer"]
        self.critic = models["critic"]
        self.critic_target = models["critic_target"]
        self.critic_optimizer = models["critic_optimizer"]

        self.actor.to(self.device, non_blocking=True)
        self.actor_target.to(self.device, non_blocking=True)
        self.critic.to(self.device, non_blocking=True)
        self.critic_target.to(self.device, non_blocking=True)

        self.update_target()

        assert isinstance(action_space, gym.spaces.Box)
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau

    def update_target(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update(self, batch, weights=None):
        s0, a, r, s1, done = batch

        s0 = ensure_tensor(s0, torch.float, self.device)
        a = ensure_tensor(a, torch.long, self.device)
        r = ensure_tensor(r, torch.float, self.device)
        s1 = ensure_tensor(s1, torch.float, self.device)
        done = ensure_tensor(done, torch.float, self.device)
        if weights is not None:
            weights = ensure_tensor(weights, torch.float, self.device)

        a1 = self.actor_target(s1).detach()
        target_q = self.critic_target(s1, a1).detach()
        y_expected = r[:, None] + (1 - done[:, None]) * self.gamma * target_q
        y_predicted = self.critic.forward(s0, a)

        # critic gradient
        critic_loss = y_predicted - y_expected.detach()
        critic_loss = critic_loss.squeeze()
        out_loss = critic_loss.detach().cpu().abs()
        critic_loss = critic_loss.pow(2)
        if weights is not None:
            critic_loss *= weights
        critic_loss = critic_loss.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor gradient
        pred_a = self.actor.forward(s0)
        loss_actor = (-self.critic.forward(s0, pred_a)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Notice that we only have gradient updates for actor and critic, not target
        # actor_optimizer.step() and critic_optimizer.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return out_loss

    def act(self, state):
        state = ensure_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

        action = self.actor(state).detach()
        action = action.squeeze(0).cpu().numpy()
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def load_weights(self, stream):
        states = torch.load(stream, map_location=self.device)
        self.actor.load_state_dict(states)
        self.actor.to(self.device, non_blocking=True)
        self.update_target()

    def save_weights(self, stream=None):
        if stream is None:
            stream = io.BytesIO()
        torch.save(self.actor.state_dict(), stream)
        return stream
