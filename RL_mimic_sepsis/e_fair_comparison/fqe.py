"""Fitted Q Evaluation utilities.

This module implements a lightweight, self-contained version of Fitted Q Evaluation
(FQE) tailored to the discrete BCQ agents used in this repository. The goal is to
provide a fast, numerically stable routine that can be re-used in model-selection
and evaluation scripts without depending on external libraries.

Usage outline:

    from RL_mimic_sepsis.e_fair_comparison.fqe import (
        load_transition_dataset_t2,
        fitted_q_evaluation,
    )

    transitions = load_transition_dataset_t2(data_path, t_step)
    transitions.reward = remap_rewards(transitions.reward, reward_args)
    episodes = load_dataset_t2(data_path, t_step)
    episodes.reward = remap_rewards(episodes.reward, reward_args)

    fqe_value, info = fitted_q_evaluation(
        policy_model,
        transition_buffer=transitions,
        eval_episode_buffer=episodes,
        discount=policy_model.discount,
        eps=0.01,
        device=torch.device('cpu'),
    )

The helper mirrors the BCQ masking logic (threshold over estimated behavior
probabilities) so that the value estimate matches the deployed policy.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from RL_mimic_sepsis.d_BCQ.src.data import EpisodicBuffer, SASRBuffer
from RL_mimic_sepsis.d_BCQ.src.model import BCQ_Net
from RL_mimic_sepsis.utils.timestep_util import get_horizon, get_state_dim


@dataclass
class FQEConfig:
    """Configuration for Fitted Q-Evaluation."""
    batch_size = 512
    num_epochs = 30
    lr = 1e-3
    weight_decay = 1e-4
    hidden_dim = 256
    tau = 0.05
    eps = 0.1

    num_workers = 0


def bcq_policy_distribution(model, states, behavior_pibs, eps):
    """
    Return the epsilon-soft action distribution.
    
    This function computes the policy distribution of a Batch-Constrained Q-learning
    (BCQ) model. It applies a threshold to the behavior policy probabilities to
    mask unlikely actions, computes greedy actions over the masked Q-values, and
    then creates an epsilon-soft version of the resulting policy.
    """
    device = states.device
    with torch.no_grad():
        if hasattr(model.Q, 'πb'):
            q_values, log_pibs, _ = model.Q(states)
            behavior = log_pibs.exp()
            behavior = behavior.to(device=device, dtype=q_values.dtype)
        
        else:
            q_values = model.Q(states)
            behavior = behavior_pibs.to(device=device, dtype=q_values.dtype)
            
    behavior = torch.clone(behavior)
    threshold = float(getattr(model, 'threshold', 0.0))

    mask = (behavior > threshold).to(q_values.dtype)
    masked_q = mask * q_values + (1.0 - mask) * torch.finfo(q_values.dtype).min
    greedy_action = masked_q.argmax(dim=1, keepdim=True)

    policy = eps * behavior
    policy.scatter_add_(1, greedy_action, torch.ones_like(greedy_action, dtype=policy.dtype) * (1.0 - eps))
    return policy


class FQEEstimator(nn.Module):
    """
    A simple Q-network wrapper for fitted Q-evaluation.
    
    This class encapsulates a BCQ_Net, which serves as the Q-function approximator
    for the FQE process.
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        """Initializes the FQE estimator network."""
        super().__init__()
        self.q_network = BCQ_Net(state_dim, action_dim, hidden_dim)

    def forward(self, states):
        """Performs a forward pass through the Q-network."""
        return self.q_network(states)


class FQETrainer:
    """
    Manages the training process for the FQE Q-estimator.
    
    This class handles the Bellman updates, target network synchronization,
    and optimization for learning the Q-function of a fixed policy.
    """
    def __init__(
        self,
        policy_model,
        state_dim,
        num_actions,
        discount,
        config,
        device,
    ):
        """Initializes the FQE trainer."""
        super().__init__()
        self.device = device
        self.discount = float(discount)
        self.policy_model = policy_model.to(device)
        self.policy_model.eval()
        self.config = config

        self.estimator = FQEEstimator(state_dim, num_actions, config.hidden_dim).to(device)
        self.target_estimator = copy.deepcopy(self.estimator).to(device)
        for param in self.target_estimator.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.estimator.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    def _soft_update(self):
        """Performs a soft update of the target network parameters."""
        tau = self.config.tau
        with torch.no_grad():
            for param, target_param in zip(self.estimator.parameters(), self.target_estimator.parameters()):
                target_param.data.mul_(1 - tau).add_(tau * param.data)

    def _value_target(self, next_states, next_behavior, rewards, not_dones):
        """Computes the Bellman target for a batch of transitions."""
        with torch.no_grad():
            next_policy = bcq_policy_distribution(self.policy_model, next_states, next_behavior, self.config.eps)
            target_q = self.target_estimator(next_states)
            v_next = (next_policy * target_q).sum(dim=1, keepdim=True)
            return rewards + not_dones * self.discount * v_next

    def update(self, batch):
        """Performs a single training update on a batch of data."""
        states, actions, next_states, rewards, not_dones, _, next_pibs = batch
        states = states.to(self.device).float()
        actions = actions.to(self.device).long()
        next_states = next_states.to(self.device).float()
        rewards = rewards.to(self.device).float()
        not_dones = not_dones.to(self.device).float()
        next_behavior = next_pibs.to(self.device).float()

        q_pred = self.estimator(states).gather(1, actions)
        target = self._value_target(next_states, next_behavior, rewards, not_dones)

        loss = F.mse_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self._soft_update()
        return float(loss.item())

    def fit(self, dataloader):
        """Trains the FQE estimator for a configured number of epochs."""
        losses = []
        for _ in range(self.config.num_epochs):
            for batch in dataloader:
                losses.append(self.update(batch))
        return losses

    def estimate_initial_value(self, states, behavior):
        """
        Estimates the policy value over a distribution of initial states.
        
        Returns the mean and standard deviation of the estimated values.
        """
        self.estimator.eval()
        with torch.no_grad():
            states = states.to(self.device).float()
            behavior = behavior.to(self.device).float()
            policy = bcq_policy_distribution(self.policy_model, states, behavior, self.config.eps)
            q = self.estimator(states)
            values = (policy * q).sum(dim=1)
            mean = float(values.mean().item())
            std = float(values.std(unbiased=False).item())
        return mean, std


def load_transition_dataset_t2(data_path, t2, action_space='NormThreshold'):
    """
    Loads a transition dataset for a given timestep.
    
    This helper function initializes and loads a SASRBuffer, which contains
    individual state-action-reward-state transitions.
    """
    state_dim = get_state_dim(t2, action_space)
    num_actions = 25
    buffer = SASRBuffer(state_dim, num_actions)
    buffer.load(data_path)
    return buffer


def fitted_q_evaluation(
    policy_model,
    transition_buffer,
    eval_episode_buffer=None,
    discount=None,
    config=None,
    device=None,
):
    """
    Runs the main Fitted Q Evaluation algorithm.

    This function orchestrates the training of the FQE Q-estimator and computes
    the final policy value estimate based on the initial state distribution.

    Args:
        policy_model: The BCQ policy to evaluate.
        transition_buffer: Transition data (SASRBuffer) for training the estimator.
        eval_episode_buffer: Episodic buffer to define the initial state distribution.
        discount: Discount factor. Defaults to the policy's discount.
        config: FQEConfig with hyperparameters.
        device: The torch device to run on.

    Returns:
        A tuple of (value_estimate, diagnostics_dict).
    """
    if config is None:
        config = FQEConfig()
    if discount is None:
        discount = float(getattr(policy_model, 'discount', 0.99))

    if device is None:
        device = next(policy_model.parameters()).device if any(p.device.type != 'cpu' for p in policy_model.parameters()) else torch.device('cpu')
    device = torch.device(device)

    policy_model = policy_model.to(device)
    policy_model.eval()

    batch_size = min(config.batch_size, len(transition_buffer))
    if batch_size < 1:
        raise ValueError('Transition buffer is empty; cannot run FQE.')

    dataloader = DataLoader(
        transition_buffer,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=device.type == 'cuda',
    )

    trainer = FQETrainer(
        policy_model=policy_model,
        state_dim=transition_buffer.state.shape[1],
        num_actions=transition_buffer.pibs.shape[1],
        discount=discount,
        config=config,
        device=device,
    )
    loss_history = list(trainer.fit(dataloader))

    if eval_episode_buffer is not None:
        init_states = eval_episode_buffer.state[:, 0, :]
        init_behavior = eval_episode_buffer.estm_pibs[:, 0, :]
    else:
        init_states = transition_buffer.state
        init_behavior = transition_buffer.pibs

    value_mean, value_std = trainer.estimate_initial_value(init_states, init_behavior)
    diagnostics = {
        'fqe_value_std': value_std,
        'fqe_final_loss': float(loss_history[-1]) if loss_history else 0.0,
    }
    return value_mean, diagnostics



        # if self.config.grad_clip is not None:
        #     torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), self.config.grad_clip)
            # grad_clip = 5.0