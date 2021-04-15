import numpy as np
import torch


def ensure_tensor(arr, dtype, device, non_blocking=True):
    device = torch.device(device)
    if torch.is_tensor(arr):
        return arr.to(dtype=dtype, device=device, non_blocking=non_blocking)
    else:
        return torch.as_tensor(arr, dtype=dtype).to(
            device=device, non_blocking=non_blocking
        )


def soft_update(target, source, tau=0.001):
    """
    update target by target = tau * source + (1 - tau) * target
    :param target: Target network
    :param source: source network
    :param tau: 0 < tau << 1
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def calculate_discount_rewards_with_dones(r, done, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        if done[t]:
            discounted_r[t] = 0
        else:
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add

    discounted_r -= discounted_r.mean()
    discounted_r /= discounted_r.std() + 1e-7
    return discounted_r


def calculate_discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean()
    discounted_r /= discounted_r.std() + 1e-7
    return discounted_r


def transform_to_onehot(actions, action_space):
    result = []
    for action in actions:
        result.append([int(k == action) for k in range(action_space)])
    return result
