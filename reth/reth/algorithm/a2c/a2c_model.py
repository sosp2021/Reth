import torch
import torch.nn as nn
import torch.nn.functional as F

import gym


class ActorNetwork(nn.Module):
    def __init__(self, obs_shape, output_shape, hidden_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim=-1)
        return out


class ValueNetwork(nn.Module):
    def __init__(self, obs_shape, output_shape, hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def generate_a2c_default_models(
    observation_space, action_space, hidden_size=64, learning_rate=0.01
):
    assert isinstance(action_space, gym.spaces.Discrete)
    num_actions = action_space.n
    assert isinstance(observation_space, gym.spaces.Box)
    obs_shape = observation_space.shape

    value_network = ValueNetwork(obs_shape, 1, hidden_size)
    value_network_optimizer = torch.optim.Adam(
        value_network.parameters(), lr=learning_rate
    )

    # init actor network
    actor_network = ActorNetwork(obs_shape, num_actions, hidden_size)
    actor_network_optimizer = torch.optim.Adam(
        actor_network.parameters(), lr=learning_rate
    )

    return {
        "value_network": value_network,
        "value_optimizer": value_network_optimizer,
        "actor_network": actor_network,
        "actor_optimizer": actor_network_optimizer,
    }
