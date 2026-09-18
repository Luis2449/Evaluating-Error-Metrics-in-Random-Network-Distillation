import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, RNDModel
from utils import global_grad_norm_


class RNDAgent(object):
    """PPO agent with Random Network Distillation as the intrinsic reward source.

    The policy/value network, optimizer settings, rollout interface, and RND
    architecture are shared across loss conditions.
    """

    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            rnd_loss='mse'):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # Normalize common configuration spellings so the experiment scripts can
        # use readable loss names without changing the code path.
        self.rnd_loss = rnd_loss.lower().replace('-', '_').replace('^', '')
        # Fixed constants used by the alternative RND prediction losses. Keeping
        # them here makes their experimental role explicit and shared by reward
        # computation and predictor training.
        self.rnd_loss_delta = 1.0
        self.rnd_dead_zone_eps = 0.05
        self.rnd_loss_power = 4.0
        if self.rnd_loss not in ('mse', 'huber', 'reverse_huber', 'cauchy', 'dead_zone_high_power', 'e4'):
            raise ValueError(
                "RNDLoss must be 'mse', 'huber', 'reverse_huber', 'cauchy', "
                "'dead_zone_high_power', or 'e4', got '{}'".format(rnd_loss))

        self.rnd = RNDModel(input_size, output_size)
        # Only the actor-critic and RND predictor are optimized. The random RND
        # target network is fixed in RNDModel and is intentionally not included.
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()),
                                    lr=learning_rate)
        self.rnd = self.rnd.to(self.device)

        self.model = self.model.to(self.device)

    def rnd_prediction_error(self, prediction, target, reduction_dim, mse_scale=0.5):
        """Return the selected RND prediction error reduced over feature axes.

        mse is the RND baseline. The default scale of ``0.5`` matches the
        upstream intrinsic-reward convention ``sum(e^2) / 2``. Predictor
        training passes ``mse_scale=1.0`` and then divides by feature count, so
        MSE training uses a feature mean, while reward computation uses a feature
        sum. The non-MSE losses follow the same reduction pattern.

        reverse_huber is implemented as a fixed-threshold BerHu-style loss:
        L1 below delta and quadratic above it. cauchy uses the robust
        log loss with the same delta scale, and e4 is the fourth-power
        prediction error.
        """
        if self.rnd_loss == 'huber':
            return F.huber_loss(
                prediction,
                target,
                reduction='none',
                delta=self.rnd_loss_delta).sum(reduction_dim)

        error = prediction - target
        abs_error = torch.abs(error)
        if self.rnd_loss == 'reverse_huber':
            delta = self.rnd_loss_delta
            loss = torch.where(
                abs_error <= delta,
                abs_error,
                (error.pow(2) + delta ** 2) / (2 * delta))
        elif self.rnd_loss == 'cauchy':
            delta = self.rnd_loss_delta
            loss = (delta ** 2 / 2) * torch.log1p((error / delta).pow(2))
        elif self.rnd_loss == 'dead_zone_high_power':
            loss = torch.clamp(abs_error - self.rnd_dead_zone_eps, min=0).pow(self.rnd_loss_power)
        elif self.rnd_loss == 'e4':
            loss = error.pow(4)
        else:
            loss = error.pow(2) * mse_scale

        return loss.sum(reduction_dim)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return action, value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, next_obs):
        """Compute RND novelty from target/predictor disagreement.

        ``next_obs`` is the RND observation stream produced by the environment
        workers. In the noisy-observation experiment this may differ from the
        policy observation stream when ``ObsNoiseTarget=rnd_only``.
        """
        next_obs = torch.FloatTensor(next_obs).to(self.device)

        # The fixed target and trainable predictor are compared here; the
        # resulting per-sample prediction error is used directly as intrinsic
        # reward by the training and evaluation loops.
        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = self.rnd_prediction_error(predict_next_feature, target_next_feature, 1)

        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        """Run PPO updates and the RND predictor update on one rollout batch."""
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)


        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]


                # RND predictor training uses the same selected prediction loss
                # as intrinsic reward, but averages across the feature vector
                # before applying the original stochastic update mask.
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])

                forward_loss = self.rnd_prediction_error(
                    predict_next_state_feature,
                    target_next_state_feature.detach(),
                    -1,
                    mse_scale=1.0)
                forward_loss = forward_loss / predict_next_state_feature.size(-1)
                # Match the upstream RND update schedule: only a fixed
                # proportion of rollout samples train the predictor each epoch.
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))


                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                loss.backward()
                parameters = list(self.model.parameters()) + list(self.rnd.predictor.parameters())
                if self.clip_grad_norm > 0:
                    clip_grad_norm_(parameters, self.clip_grad_norm)
                else:
                    global_grad_norm_(parameters)
                self.optimizer.step()
