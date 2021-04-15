import gym
import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_latent_var):
        super().__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(input_shape[0], n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, output_shape),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        out = self.action_layer(x)
        return out


class ValueNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_latent_var):
        super().__init__()

        # actor
        self.value_layer = nn.Sequential(
            nn.Linear(input_shape[0], n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, output_shape),
        )

    def forward(self, x):
        out = self.value_layer(x)
        return out


def generate_discrite_ppo_model(
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
