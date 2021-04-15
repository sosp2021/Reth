import gym
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions, dueling=True, hidden_unit=256):
        super().__init__()

        self.input_shape = obs_shape
        self.num_actions = num_actions
        self.dueling = dueling

        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        if not dueling:
            self.fc = nn.Sequential(
                nn.Linear(self.features_size(), 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
            )
        else:
            self.fc_adv = nn.Sequential(
                nn.Linear(self.features_size(), hidden_unit),
                nn.ReLU(),
                nn.Linear(hidden_unit, self.num_actions),
            )

            self.fc_value = nn.Sequential(
                nn.Linear(self.features_size(), hidden_unit),
                nn.ReLU(),
                nn.Linear(hidden_unit, 1),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        if self.dueling:
            adv = self.fc_adv(x)
            value = self.fc_value(x)
            adv_avg = torch.mean(adv, dim=1, keepdim=True)
            q = value + adv - adv_avg
        else:
            q = self.fc(x)
        return q

    def features_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class MLP_DQNNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.nn(x)


def generate_dqn_network(observation_space, action_space, dueling=True):
    assert isinstance(action_space, gym.spaces.Discrete)
    num_actions = action_space.n
    assert isinstance(observation_space, gym.spaces.Box)
    obs_shape = observation_space.shape

    if len(obs_shape) == 3:
        return DQNNetwork(obs_shape, num_actions, dueling=dueling)
    else:
        return MLP_DQNNetwork(obs_shape, num_actions)


def generate_dqn_default_models(
    observation_space, action_space, dueling=True, learning_rate=5e-5, adam_epsilon=1e-8
):
    q_network = generate_dqn_network(observation_space, action_space, dueling=dueling)
    target_q_network = generate_dqn_network(
        observation_space, action_space, dueling=dueling
    )
    return {
        "q_network": q_network,
        "target_q_network": target_q_network,
        "optimizer": torch.optim.Adam(
            q_network.parameters(), lr=learning_rate, eps=adam_epsilon
        ),
    }
