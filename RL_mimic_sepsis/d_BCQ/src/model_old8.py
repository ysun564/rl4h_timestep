import sys
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

sys.path.append('/local/scratch/ysun564/project/OfflineRL_TimeStep')
from RL_mimic_sepsis.utils.timestep_util import get_horizon, get_state_dim

# Problem-specific hyperparameters
timestep = 8

action_space = 'NormThreshold'

state_dim = get_state_dim(timestep, action_space)
horizon = get_horizon(timestep)
num_actions = 25


class BCQ_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        self.q: Learns Q-values.
        """
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.πb = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        q_values = self.q(x)
        p_logits = self.πb(x)
        return q_values, F.log_softmax(p_logits, dim=1), p_logits


class BCQ(pl.LightningModule):
    def __init__(
        self,
        *,
        state_dim,
        num_actions,
        hidden_dim,
        lr,
        weight_decay,
        threshold,
        discount,
        eval_discount,
        polyak_target_update=True,
        target_update_frequency=1,
        tau=0.005,
        target_value_clipping=False,
        Rmin=None,
        Rmax=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.target_value_clipping = target_value_clipping

        self.automatic_optimization = False
        # Initializes a BCQ_Net.
        self.Q = BCQ_Net(state_dim, num_actions, hidden_dim)
        # Copy a network as target network.
        self.Q_target = copy.deepcopy(self.Q)
        self.discount = discount
        self.num_actions = num_actions
        
        # Freeze target network so we don't accidentally train it
        for param in self.Q_target.parameters():
            param.requires_grad = False
        
        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        

        # Threshold for "unlikely" actions
        self.threshold = threshold
        
        # Discount for validation WIS OPE
        self.eval_discount = eval_discount or self.discount
        
        self.Rmin = Rmin
        self.Rmax = Rmax

        # Range of rewards/values for clipping
        if target_value_clipping and (Rmin is not None) and (Rmax is not None):
            self.vmin = self.Rmin / (1 - self.discount)
            self.vmax = self.Rmax / (1 - self.discount)
        else:
            self.vmin = None
            self.vmax = None
        # Number of training iterations
        self.iterations = 0
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BCQ")
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--threshold", type=float, default=0.3)
        parser.add_argument("--discount", type=float, default=0.99)
        parser.add_argument("--eval_discount", type=float, default=None)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument('--target_value_clipping', default=False, action=argparse.BooleanOptionalAction)
        return parent_parser
    
    def forward(self, state):
        return self.Q(state)
    
    def configure_optimizers(self):
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return self.Q_optimizer
    
    def training_step(self, batch, batch_idx):
        """Training BCQ Network.
        
        """
        state, action, next_state, reward, notdone, pibs, next_pibs = batch
        
        # Compute the target Q value.
        with torch.no_grad():
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True).values > self.threshold).float()
            # Use large negative number to mask actions from argmax.
            next_action = (imt * q + (1 - imt) * torch.finfo(q.dtype).min).argmax(axis=1, keepdim=True)

            q, imt, i = self.Q_target(next_state)
            target_Q = reward + notdone * self.discount * q.gather(1, next_action).reshape(-1, 1)
            
            if self.target_value_clipping:
                target_Q = torch.clamp(target_Q, self.vmin, self.vmax)

        # Get current Q estimate.
        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)

        # Compute Q loss.
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))
        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # self.Q_optimizer.zero_grad()
        # self.manual_backward(Q_loss)
        # self.Q_optimizer.step() 

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(Q_loss)
        opt.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())


    def validation_step(self, batch, batch_idx):
        qvalues = self.offline_q_evaluation(batch)
        valid_wis, valid_ess = self.offline_evaluation(batch)
        self.log('iteration', int(self.iterations), prog_bar=True, logger=True)
        self.log('val_qvalues', qvalues, prog_bar=True, logger=True)
        self.log('val_wis', valid_wis, prog_bar=True, logger=True)
        self.log('val_ess', valid_ess, prog_bar=True, logger=True)
        return {
            'iteration': self.iterations,
            'val_qvalues': qvalues,
            'val_wis': valid_wis,
            'val_ess': valid_ess,
        }
    

    def offline_q_evaluation(self, eval_buffer):
        states, _, _, _, _, estm_pibs = eval_buffer
        # Only consider the initial states and behavior actions.
        states = states[:, 0, :]  
        estm_pibs = estm_pibs[:, 0, :]

        q, imt, _ = self.Q(states)
        imt = imt.exp()
        imt = (imt / imt.max(axis=1, keepdim=True).values > self.threshold).float()

        # Use large negative number to mask actions from argmax.
        values = (imt * q + (1. - imt) * torch.finfo(q.dtype).min).max(axis=1).values
        return values.mean().item()

    def offline_evaluation(self, eval_buffer, eps=0.1,
                           analyze: bool = False, clipping: float = 1.438):
        """Offline evaluation function. Returns estimated WIS value and ESS. 
        'analyze': determines whether analyze the IR distribution and draw figure.
        'clipping': automatically distinguish the clipping method (threshold/percentile).
        """
        states, actions, rewards, not_dones, pibs, estm_pibs = eval_buffer
        rewards = rewards[:, :, 0].cpu().numpy()
        n, horizon, _ = states.shape
        # NOTE: eval_discount = 1 by default, so here rewards should be unchanged. 
        discounted_rewards = rewards * (self.eval_discount ** np.arange(horizon))

        ir = np.ones((n, horizon))
        for idx in range(n):
            lng = (not_dones[idx, :, 0].sum() + 1).item()  # All but the final transition has notdone==1.

            # Predict Q-values.
            q, imt, _ = self.Q(states[idx])
            
            # In test sets we cannot know the ground truth of behavior cloning 
            # probability distribution (i.e. pibs). We can only estimate it via
            # the training behavior policy, so use estm_pibs here.
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True).values > self.threshold).float()

            # Select one action.
            a_id = (imt * q + (1. - imt) * torch.finfo(q.dtype).min).argmax(axis=1).cpu().numpy()

            # Soften using training behavior policy.
            pie_soft = np.zeros((horizon, self.num_actions))
            pie_soft += eps * estm_pibs[idx].cpu().numpy() 
            pie_soft[range(horizon), a_id] += (1.0 - eps)

            # Compute importance sampling ratios.
            a_obs = actions[idx, :, 0]
            ir[idx, :lng] = (pie_soft[range(lng), a_obs[:lng].cpu().numpy()] 
                           / pibs[idx, range(lng), a_obs[:lng]].cpu().numpy())
            
            ir[idx, lng:] = 1  # Mask out the padded timesteps

        rho_prefix  = np.cumprod(ir, axis=1) 
        rho_final  = rho_prefix[:, -1]

        # Automaticlly distinguishes the clipping method.
        rho_final_clip = []
        if clipping > 100:
            threshold = clipping
            rho_final_clip = np.clip(rho_final, 0, threshold)

        elif 5.0 <= clipping <= 100.0:
            M = np.percentile(rho_final, clipping)
            rho_prefix_clip = np.minimum(rho_prefix, M)
            rho_final_clip  = np.minimum(rho_final,  M)
            ir_clip = np.minimum(ir, M)

        elif clipping < 5.0 and clipping > 0.0:
            # Calculate clipped importance ratio via a threshold list 
            # based on valid step size for each trajectory.
            # 'clipping' is by default 1.438 (19th root of 1000), 
            # where 19 is the maximum step size for 4-h time step.
            # For 4-h: 'ir': (2785, 19) -> 'valid_step_list': (2785) -> 'threshold_list' -> (2785)
            valid_step_list = []
            for idx in range(n):
                valid_step_list.append((not_dones[idx, :, 0].sum() + 1).item())
            valid_step_array = np.array(valid_step_list)
            threshold_list = clipping ** valid_step_array
            rho_final_clip = np.clip(rho_final, 0, threshold_list)

        elif clipping == 0:
            # Doing nothing.
            rho_final_clip = rho_final

        else:
            rho_final_clip = rho_final
        
        wis_est, wis_ess = self.wis(rho_final_clip, discounted_rewards, True)
    
        return wis_est, wis_ess
        

    def wis(self, weights, discounted_rewards, weighted = True):
        if weighted:
            weights /= weights.sum()
        wis = (weights * discounted_rewards.sum(axis=-1)).sum()
        ess  = 1.0 / (weights**2).sum()
        return wis, ess
        


    
    
