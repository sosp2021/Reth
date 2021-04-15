import gym
import numpy as np
import torch
import torch.nn as nn


class ConvActor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_unit=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.features_size(), hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, action_shape),
        )
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.features(state)
        x = self.fc1(x)
        action = self.tanh(x)
        return action


class ConvCritic(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_unit=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.features_size() + action_shape, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, 1),
        )

    def forward(self, state, action):
        """
        return critic Q(s,a)
        :param state: state [n, state_dim] (n is batch_size)
        :param action: action [n, action_dim]
        :return: Q(s,a) [n, 1]
        """
        s1 = self.features(state)
        x = torch.cat((s1, action), dim=1)
        x = self.fc1(x)
        return x


def fanin_init(size, fanin=None):
    """
    weight initializer known from https://arxiv.org/abs/1502.01852
    :param size:
    :param fanin:
    :return:
    """
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, eps=0.03):
        super().__init__()
        h1 = 400
        h2 = 300
        self.fc1 = nn.Linear(obs_shape[0], h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, action_shape[0])
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, eps=0.03):
        super().__init__()

        h1 = 400
        h2 = 300
        self.fc1 = nn.Linear(obs_shape[0], h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1 + action_shape[0], h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        s1 = self.relu(self.fc1(state))
        x = torch.cat((s1, action), dim=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_ddpg_model(
    observation_space,
    action_space,
    learning_rate_actor,
    learning_rate_critic,
    is_image=None,
):
    assert isinstance(action_space, gym.spaces.Box)
    action_shape = action_space.shape
    assert isinstance(observation_space, gym.spaces.Box)
    obs_shape = observation_space.shape

    if is_image is None:
        is_image = len(obs_shape) == 3

    if is_image:
        actor = ConvActor(obs_shape, action_shape)
        actor_target = ConvActor(obs_shape, action_shape)
        critic = ConvCritic(obs_shape, action_shape)
        critic_target = ConvCritic(obs_shape, action_shape)
    else:
        actor = Actor(obs_shape, action_shape)
        actor_target = Actor(obs_shape, action_shape)
        critic = Critic(obs_shape, action_shape)
        critic_target = Critic(obs_shape, action_shape)

    actor_optimizer = torch.optim.Adam(actor.parameters(), learning_rate_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), learning_rate_critic)

    models = {
        "actor": actor,
        "actor_target": actor_target,
        "actor_optimizer": actor_optimizer,
        "critic": critic,
        "critic_target": critic_target,
        "critic_optimizer": critic_optimizer,
    }
    return models
