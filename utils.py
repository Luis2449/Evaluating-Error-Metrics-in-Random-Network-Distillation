from config import *
import numpy as np

import torch
from math import inf

# if default_config['TrainMethod'] in ['PPO', 'ICM', 'RND']:
#     num_step = int(ppo_config['NumStep'])
# else:
#     num_step = int(default_config['NumStep'])

use_gae = default_config.getboolean('UseGAE')
lam = float(default_config['Lambda'])
train_method = default_config['TrainMethod']


def torch_load(path, map_location=None, weights_only=True):
    """Load PyTorch files across versions before/after the 2.6 weights_only default change."""
    kwargs = {}
    if map_location is not None:
        kwargs['map_location'] = map_location
    try:
        return torch.load(path, weights_only=weights_only, **kwargs)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, **kwargs)


def make_train_data(reward, done, value, gamma, num_step, num_worker):
    """Compute rollout returns and advantages for PPO.

    The caller chooses whether ``done`` represents episodic external rewards or
    the always-continuing intrinsic reward stream. ``UseGAE`` and ``Lambda`` are
    read from the shared experiment configuration.
    """
    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * lam * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

            # For Actor
        adv = discounted_return - value[:, :-1]

    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        # For Actor
        adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


class RunningMeanStd(object):
    """Online mean/variance estimator used for RND normalization statistics."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    """Discounted reward filter used before updating intrinsic-reward RMS."""

    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def global_grad_norm_(parameters, norm_type=2):
    r"""Compute the total gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are not modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor whose gradients will be measured
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm
