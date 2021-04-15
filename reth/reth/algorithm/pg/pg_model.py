import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, obs_shape, output_size):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(obs_shape[0], 32)
        self.fc2 = nn.Linear(32, 20)
        self.fc3 = nn.Linear(20, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


def generate_pg_default_models(observation_space, action_space, learning_rate=0.001):
    assert isinstance(action_space, gym.spaces.Discrete)
    num_actions = action_space.n
    assert isinstance(observation_space, gym.spaces.Box)
    obs_shape = observation_space.shape
    model = PolicyNet(obs_shape, num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return {"policy_model": model, "optimizer": optimizer}
