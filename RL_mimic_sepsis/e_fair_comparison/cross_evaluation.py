"""
Cross-Δt Offline Policy Evaluation.

Evaluate a policy trained at  timestep t1 on an dataset with timestep t2. 
The numerator is constructed by:
  1) computing a 25-d epsilon-soft policy at the current t2 state using the t1 policy.
  2) automatically compare t1 and t2, then extend/shirnk the distribution.
  3) aggregating with aggregate_identical_t1_to_t2 to obtain a single 25-d t2 distribution,
     instead of using aggregate_t1_to_t2_25d, to accelerate the inference.

This uses the Expected-Overlap mapping and vaso-max rules implemented in aggregate.py.
Clipping and WIS follow the repository's original OPE logic.

To run:
D:\\Software\\anaconda3\\envs\\rl4h_rep_new\\python.exe -m RL_mimic_sepsis.e_fair_comparison.cross_evaluation
"""

import os
import sys
import time
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from RL_mimic_sepsis.utils.timestep_util import get_state_dim, get_horizon
from RL_mimic_sepsis.d_BCQ.src.model import BCQ
from RL_mimic_sepsis.d_BCQ.src.data import EpisodicBuffer, remap_rewards
from RL_mimic_sepsis.e_fair_comparison.aggregate import aggregate_t1_to_t2_25d, aggregate_identical_t1_to_t2
from RL_mimic_sepsis.e_fair_comparison.mapping_raw import CAPS, sum_interval_for_counts, decide_bucket
from RL_mimic_sepsis.e_fair_comparison.fqe import fitted_q_evaluation, load_transition_dataset_t2


def eps_soft_joint25(state, model, estm_pib25, eps):
    """
    Build an epsilon-soft 25-d distribution at 'state' using the t1 policy.
    It can automatically adjust using pi_b based on model structure.

    Steps:
      - Q(s) from model;
      - IMT mask via estm_pib25 / max > threshold;
      - Greedy joint action under masked Q;
      - Epsilon-soft around greedy with estm_pib25 as the epsilon base.
    """
    with torch.no_grad():
        if hasattr(model.Q, 'πb'):
            q, log_pibs, _ = model.Q(state.unsqueeze(0))
            q = q[0]
            # Use internal behavior policy and ignore estm_pib25 for the mask
            estm = log_pibs.exp()[0]      
        else: 
            q = model.Q(state.unsqueeze(0))[0]
            estm = estm_pib25.to(q.device, dtype=q.dtype)

    threshold = float(getattr(model, 'threshold', 0.0))
    mx = estm.max()
    if mx.item() <= 0:
        # if the estimated behavior is all zeros, fall back to no mask
        imt = torch.ones_like(estm)
    else:
        imt = (estm / mx > threshold).to(q.dtype)

    masked_q = imt * q + (1.0 - imt) * torch.finfo(q.dtype).min
    a_star = int(masked_q.argmax().item())

    # epsilon-soft using estm as the base; normalize at the end
    p = eps * estm.detach().cpu().numpy()
    p[a_star] += (1.0 - eps)
    s = p.sum()
    if s > 0:
        p /= s
    return p  

def wis(weights, discounted_rewards):
    """
    Weighted importance sampling estimate and effective sample size.
    """
    w_sum = weights.sum()
    if w_sum <= 0:
        return 0.0, 0.0
    w = weights / w_sum
    est = (w * discounted_rewards.sum(axis=-1)).sum()
    ess = 1.0 / (w ** 2).sum()
    return est, ess

def ph_wis(final_weights, discounted_rewards, trajectory_lengths):
    """
    Per-horizon weighted importance sampling (PHWIS).

    This groups trajectories by horizon length L and computes a WIS estimate
    within each group, then mixes per-horizon estimates by the empirical
    frequency of horizons.

    Returns a tuple (estimate, effective_sample_size).
    """
    n = len(final_weights)
    if n == 0:
        return 0.0, 0.0

    # Compute per-trajectory discounted return truncated by its horizon.
    returns = np.array([
        discounted_rewards[i, :L].sum() for i, L in enumerate(trajectory_lengths)
    ], dtype=float)

    estimate = 0.0
    unique_L = np.unique(trajectory_lengths)
    for L in unique_L:
        idx = np.where(trajectory_lengths == L)[0]
        w = final_weights[idx]
        r = returns[idx]
        denom = w.sum()
        v_L = 0.0 if denom <= 0 else float((w * r).sum() / denom)
        estimate += (len(idx) / n) * v_L

    ess = 0.0
    denom_ess = float((final_weights ** 2).sum())
    if denom_ess > 0:
        ess = float((final_weights.sum() ** 2) / denom_ess)
    return float(estimate), float(ess)

def evaluate_cross_dt_fqe(model, dataset_t2, data_path_t2, t1, t2, eps=0.01, **kwargs):
    """
    Evaluate a t1 policy on a t2 dataset using Fitted Q-Evaluation.

    This function orchestrates FQE by:
    1. Loading the transition data (SASRBuffer) for the evaluation timestep t2.
    2. Using the episodic data (dataset_t2) for the initial state distribution.
    3. Calling the core `fitted_q_evaluation` routine.
    """
    model.eval()
    device = next(model.parameters()).device

    # FQE requires a transition buffer for training the Q-estimator
    # The episodic buffer's initial states are used for the final value estimate.
    transition_buffer_t2 = load_transition_dataset_t2(data_path_t2, t2)

    # Ensure rewards are consistent between episodic and transition buffers
    reward_args = kwargs.get('reward_args', SimpleNamespace(**{'R_immed': 0.0, 'R_death': 0.0, 'R_disch': 100.0}))
    transition_buffer_t2.reward = remap_rewards(transition_buffer_t2.reward, reward_args)

    # The policy's discount factor is used by default in FQE
    discount = 1

    fqe_value, diagnostics = fitted_q_evaluation(
        policy_model=model,
        transition_buffer=transition_buffer_t2,
        eval_episode_buffer=dataset_t2,
        discount=discount,
        device=device,
    )

    # FQE does not produce an ESS equivalent, so we return 0.0 for API consistency.
    return fqe_value, 0.0


def evaluate_cross_dt_policy_pib(model, dataset_t2, t1, t2, eps=0.01, clipping=1.438, weighting='wis'):
    """
    Evaluate a t1 policy on a t2 dataset using IS/WIS with proper cross-Δt alignment.

    Denominator: dataset_t2.pibs (25-d at each t2 step).
    Numerator: aggregate_t1_to_t2_25d([p25_small]*n, t1, t2, CAPS), where
               p25_small is epsilon-soft t1 policy at the current t2 state.
    """
    model.eval()
    device = next(model.parameters()).device

    states = dataset_t2.state.to(device)       
    actions = dataset_t2.action
    rewards = dataset_t2.reward[:, :, 0].numpy()
    not_dones = dataset_t2.not_done
    pibs = dataset_t2.pibs
    estm_pibs = dataset_t2.estm_pibs
      
    n_episodes, horizon, _ = states.shape

    # Discount schedule (eval_discount=1 if not specified).
    eval_discount = getattr(model, 'eval_discount', None)
    if eval_discount is None:
        eval_discount = getattr(model, 'discount', 1.0)
    if eval_discount is None:
        eval_discount = 1.0

    discounted_rewards = rewards * (eval_discount ** np.arange(horizon))

    # Cross ratio n = t2/t1.
    m, n = 0, 0
    if t1 == t2:
        mode = 'equal'
    elif t2 > t1:
        if t2 % t1 != 0:
            raise ValueError('t2 must be a multiple of t1.')
        mode = 'up'
        n = t2 // t1
    else:
        if t1 % t2 != 0:
            raise ValueError('t1 must be a multiple of t2.')
        mode = 'down'
        m = t1 // t2

    # Step-wise importance ratios at t2 granularity.
    ir = np.ones((n_episodes, horizon), dtype=np.float64)

    for i in range(n_episodes):
        lng = int(not_dones[i, :, 0].sum().item() + 1)

        if mode == 'equal':
            # Standard per-step OPE at same timestep
            for t in range(lng):
                s_t = states[i, t]
                a_obs = int(actions[i, t, 0].item())
                pib_t = pibs[i, t].cpu().numpy()
                est_t = estm_pibs[i, t]

                # 1)
                p25 = eps_soft_joint25(s_t, model, est_t, eps)

                # 2)
                num = float(p25[a_obs])
                den = float(pib_t[a_obs])

                if den <= 0:
                    ir[i, t] = 0.0
                else:
                    ir[i, t] = num / den


        elif mode == 'up':
            for t in range(lng):
                s_t = states[i, t].to(device)                         # (state_dim,)
                a_obs = int(actions[i, t, 0].item())                  # observed coarse action (0..24)
                pib_t = pibs[i, t].cpu().numpy()                      # behavior at t2
                est_t = estm_pibs[i, t]                               # est. behavior (for eps-soft & IMT)

                # 1) small-step epsilon-soft (25-d) from t1 policy at the SAME t2 state
                p25_small = eps_soft_joint25(s_t, model, est_t, eps)  # (25,)

                # 2) aggregate n copies -> one 25-d t2 distribution (numerator)
                p25_list = [p25_small] * n
                
                pie_t2 = aggregate_identical_t1_to_t2(p25_list[0], t1=t1, t2=t2, caps=CAPS)

                # 3) importance ratio at this t2 step
                num = float(pie_t2[a_obs])
                den = float(pib_t[a_obs])
                if den <= 0:
                    ir[i, t] = 0.0
                else:
                    ir[i, t] = num / den

        else:  # mode == 'down'
            # t1 policy on finer t2 data: broadcast big-step distribution across m small steps
            t = 0
            while t < lng:
                s_t = states[i, t]
                est_t = estm_pibs[i, t]

                # 1)
                p25_big = eps_soft_joint25(s_t, model, est_t, eps) 
                
                # NOTE: How to process last step?
                # 2)
                for j in range(m):
                    t_step = t + j
                    if t_step >= lng:
                        break

                    a_obs = int(actions[i, t_step, 0].item())
                    pib_t = pibs[i, t_step].cpu().numpy()

                    num = float(p25_big[a_obs])
                    den = float(pib_t[a_obs])

                    if den <= 0:
                        ir[i, t] = 0.0
                    else:
                        ir[i, t] = num / den

                t += m

        # mask out padded steps (remain 1.0)
        if lng < horizon:
            ir[i, lng:] = 1.0

    # Trajectory weights
    rho_prefix = np.cumprod(ir, axis=1)
    rho_final = rho_prefix[:, -1]

    # Clipping (mirror your original code)
    if clipping > 100.0:
        rho_final_clip = np.clip(rho_final, 0, clipping)

    elif 5.0 <= clipping <= 100.0:
        M = np.percentile(rho_final, clipping)
        rho_final_clip = np.minimum(rho_final, M)

    elif 0.0 < clipping < 5.0:
        valid_steps = (not_dones[:, :, 0].sum(dim=1).numpy() + 1).astype(int)
        threshold_list = np.power(clipping, valid_steps)
        rho_final_clip = np.minimum(rho_final, threshold_list)

    elif clipping == 0.0:
        rho_final_clip = rho_final

    else:
        rho_final_clip = rho_final

    if weighting == 'phwis':
        traj_lens = (not_dones[:, :, 0].sum(dim=1).numpy() + 1).astype(int)
        est, ess = ph_wis(rho_final_clip, discounted_rewards, traj_lens)
    else:
        est, ess = wis(rho_final_clip, discounted_rewards)
    return est, ess

def evaluate_cross_dt_dataset_pib(model, dataset_t2, t1, t2, eps=0.01, clipping=1.438, weighting='wis'):
    """
    Cross-Δt evaluation USING DATASET BEHAVIOR as the epsilon-soft base.

    Denominator: dataset_t2.pibs (25-d at each t2 step).
    Numerator: 
      - t1 < t2 (fine->coarse): 1) take greedy t1 action at t2 state; 
                                2) map n identical fine actions to ONE coarse (IV-bucket, vaso-max);
                                3) epsilon-soft around that coarse action using pib_t (dataset behavior @ t2).
      - t1 > t2 (coarse->fine): 1) take greedy t1 action at the start of each coarse block; 
                                2) broadcast to next m fine steps; 
                                3) epsilon-soft around that fine action using pib_t at each fine step.
      - t1 == t2:               epsilon-soft around greedy action using pib_t each step.

    Returns:
        wis_est, wis_ess
    """
    model.eval()
    device = next(model.parameters()).device

    # ---- unpack dataset tensors (keep identical names to your code) ----
    states   = dataset_t2.state.to(device)          # (N, H, state_dim)
    actions  = dataset_t2.action                    # (N, H, 1)
    rewards  = dataset_t2.reward[:, :, 0].numpy()   # (N, H)
    not_dones= dataset_t2.not_done                  # (N, H, 1) with 1s for valid, then 0s
    pibs     = dataset_t2.pibs                      # (N, H, 25) behavior @ t2
    estm_pibs= dataset_t2.estm_pibs                 # kept for API symmetry (not used here)

    n_episodes, horizon, _ = states.shape

    # ---- discount schedule (match evaluate_cross_dt) ----
    eval_discount = getattr(model, 'eval_discount', None)
    if eval_discount is None:
        eval_discount = getattr(model, 'discount', 1.0)
    if eval_discount is None:
        eval_discount = 1.0
    discounted_rewards = rewards * (eval_discount ** np.arange(horizon))

    # ---- mode selection & cross ratio ----
    if t1 == t2:
        mode = 'equal'
        m = n = 0
    elif t2 > t1:
        if t2 % t1 != 0:
            raise ValueError('t2 must be a multiple of t1.')
        mode = 'up'           # fine -> coarse
        n = t2 // t1
        m = 0
    else:
        if t1 % t2 != 0:
            raise ValueError('t1 must be a multiple of t2.')
        mode = 'down'         # coarse -> fine
        m = t1 // t2
        n = 0

    # ---- helper: greedy joint action index via existing eps_soft_joint25 (eps=0) ----
    # This avoids assumptions about model internals; with eps=0 it should return the pure model policy.
    def _greedy_action_idx(s_t, est_t):
        p25 = eps_soft_joint25(s_t, model, est_t, eps=0.0)  # (25,)
        return int(np.argmax(p25))

    # ---- pre-allocate stepwise importance ratios at t2 granularity ----
    ir = np.ones((n_episodes, horizon), dtype=np.float64)

    # ---- main loop over episodes ----
    for i in range(n_episodes):
        # Number of valid steps (same definition you used)
        lng = int(not_dones[i, :, 0].sum().item() + 1)

        if mode == 'equal':
            # t1 == t2: epsilon-soft around GREEDY action using dataset pib_t
            for t in range(lng):
                s_t   = states[i, t]
                a_obs = int(actions[i, t, 0].item())
                pib_t = pibs[i, t].detach().cpu().numpy()
                est_t = estm_pibs[i, t]  # just to match signature; unused for mixing here

                a_star = _greedy_action_idx(s_t, est_t)
                one_hot = np.zeros_like(pib_t)
                one_hot[a_star] = 1.0

                num_dist = eps * pib_t + (1.0 - eps) * one_hot
                num_dist /= num_dist.sum()

                den = float(pib_t[a_obs])
                num = float(num_dist[a_obs])
                ir[i, t] = 0.0 if den <= 0.0 else (num / den)

        elif mode == 'up':
            # t1 < t2: map n identical fine actions (greedy) -> one coarse action,
            # then epsilon-soft around that coarse action using dataset pib_t (coarse)
            for t in range(lng):
                s_t   = states[i, t]
                a_obs = int(actions[i, t, 0].item())
                pib_t = pibs[i, t].detach().cpu().numpy()
                est_t = estm_pibs[i, t]

                a_star = _greedy_action_idx(s_t, est_t)
                iv_level   = a_star // 5
                vaso_level = a_star % 5

                # counts vector: repeated iv_level n times
                counts = [0, 0, 0, 0, 0]
                counts[iv_level] = n

                # IV expected-overlap mapping over n fine steps
                L, U = sum_interval_for_counts(tuple(counts), cap_small=CAPS[t1], timestep_small=t1)
                bucket_label = decide_bucket(L, U, t2, cap_target=CAPS[t2])  # e.g., 'A3'
                b = int(bucket_label[1])  # IV bucket 0..4

                coarse_action_idx = b * 5 + vaso_level

                rec_coarse = np.zeros_like(pib_t)
                rec_coarse[coarse_action_idx] = 1.0

                num_dist = eps * pib_t + (1.0 - eps) * rec_coarse
                num_dist /= num_dist.sum()

                den = float(pib_t[a_obs])
                num = float(num_dist[a_obs])
                ir[i, t] = 0.0 if den <= 0.0 else (num / den)

        else:
            # mode == 'down': t1 > t2
            # Broadcast greedy coarse action to next m fine steps, epsilon-soft using each fine-step pib_t
            t = 0
            while t < lng:
                s_t   = states[i, t]
                est_t = estm_pibs[i, t]
                a_star = _greedy_action_idx(s_t, est_t)  # coarse policy's greedy joint action

                for j in range(m):
                    t_j = t + j
                    if t_j >= lng:
                        break
                    a_obs = int(actions[i, t_j, 0].item())
                    pib_t = pibs[i, t_j].detach().cpu().numpy()

                    one_hot = np.zeros_like(pib_t)
                    one_hot[a_star] = 1.0

                    num_dist = eps * pib_t + (1.0 - eps) * one_hot
                    num_dist /= num_dist.sum()

                    den = float(pib_t[a_obs])
                    num = float(num_dist[a_obs])
                    ir[i, t_j] = 0.0 if den <= 0.0 else (num / den)

                t += m

        # mask out padded steps
        if lng < horizon:
            ir[i, lng:] = 1.0

    # ---- trajectory weights / clipping / WIS ----
    rho_prefix = np.cumprod(ir, axis=1)
    rho_final  = rho_prefix[:, -1]

    if clipping > 100.0:
        rho_final_clip = np.clip(rho_final, 0, clipping)
    elif 5.0 <= clipping <= 100.0:
        M = np.percentile(rho_final, clipping)
        rho_final_clip = np.minimum(rho_final, M)
    elif 0.0 < clipping < 5.0:
        valid_steps = (not_dones[:, :, 0].sum(dim=1).numpy() + 1).astype(int)
        threshold_list = np.power(clipping, valid_steps)
        rho_final_clip = np.minimum(rho_final, threshold_list)
    elif clipping == 0.0:
        rho_final_clip = rho_final
    else:
        rho_final_clip = rho_final

    if weighting == 'phwis':
        traj_lens = (not_dones[:, :, 0].sum(dim=1).numpy() + 1).astype(int)
        wis_est, wis_ess = ph_wis(rho_final_clip, discounted_rewards, traj_lens)
    else:
        wis_est, wis_ess = wis(rho_final_clip, discounted_rewards)
    return wis_est, wis_ess

def load_dataset_t2(data_path, t2, action_space='NormThreshold'):
    """
    Load t2 dataset (EpisodicBuffer) with pibs and optional estm_pibs.
    """
    state_dim = 128
    horizon = get_horizon(t2)
    num_actions = 25
    buf = EpisodicBuffer(state_dim, num_actions, horizon)
    buf.load(data_path)
    return buf

def get_ckpt_path(timestep):
    '''[t=1h] best with ESS >= 100 & iter>1000 -> ESS=100.3749, WIS=98.0338, tau=0.1, seed=3, iter=100
    [t=2h] best with ESS >= 150 & iter>1000 -> ESS=157.6976, WIS=97.9959, tau=0.75, seed=0, iter=100
    [t=4h] best with ESS >= 200 & iter>1000 -> ESS=257.8325, WIS=97.3369, tau=0.5, seed=2, iter=400
    [t=8h] best with ESS >= 250 & iter>1000 -> ESS=286.9038, WIS=97.4762, tau=0.5, seed=0, iter=9700
    
    
    [t=1h] best with ESS>=50 -> ESS=55.5946, WIS=90.1773, tau=0.5, seed=1, iter=600
    [t=2h] best with ESS>=100 -> ESS=104.3268, WIS=91.8410, tau=0.5, seed=1, iter=200
    [t=4h] best with ESS>=150 -> ESS=159.8254, WIS=94.0218, tau=0.75, seed=4, iter=8300
    [t=8h] best with ESS>=250 -> ESS=300.6269, WIS=96.6350, tau=0.5, seed=2, iter=500
    '''
    action_space = 'NormThreshold'  
    if timestep == 1:
        best_threshold = 0.5
        best_seed = 1
        best_iter = 600
    elif timestep == 2:
        best_threshold = 0.5
        best_seed = 1
        best_iter = 200
    elif timestep == 4:
        best_threshold = 0.75
        best_seed = 4
        best_iter = 8300
    elif timestep == 8:
        best_threshold = 0.5
        best_seed = 2
        best_iter = 500

    else: 
        raise ValueError(f"Unsupported timestep {timestep}. Supported: 1,2,4,8.")

    if timestep == 8:
        return (rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\d_BCQ\logs'
            rf'/BCQ_as{action_space}_dt{timestep}h_grid_latent128d/dt{timestep}_threshold{best_threshold}seed{int(best_seed)}'
            rf'/checkpoints/step={int(best_iter):04}.ckpt')

    return (rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\d_BCQ\logs'
            rf'/BCQ_as{action_space}_dt{timestep}h_grid_latent128d/dt{timestep}_threshold{best_threshold}seed{int(best_seed)}'
            rf'/checkpoints/step={int(best_iter):04}-v1.ckpt')

def get_data_path(timestep):
    action_space = 'NormThreshold'  
    return (rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\data'
            rf'\data_as{action_space}_dt{timestep}h\episodes+encoded_state+knn_pibs_final\val_data.pt')

def evaluate_cross_dt(model, dataset_t2, t1, t2, eps=0.01, clipping=1.438, pib='policy', eval_method='wis', data_path_t2=None, reward_args=None):
    """Wrapper to call the appropriate cross-Δt evaluation function.
    """
    if eval_method in ('wis', 'phwis'):
        weighting = 'phwis' if eval_method == 'phwis' else 'wis'
        if pib == 'policy':
            return evaluate_cross_dt_policy_pib(model, dataset_t2, t1, t2, eps, clipping, weighting=weighting)
        elif pib == 'dataset':
            return evaluate_cross_dt_dataset_pib(model, dataset_t2, t1, t2, eps, clipping, weighting=weighting)
        else:
            raise ValueError(f"Unsupported pib type '{pib}'. Supported: 'policy', 'dataset'.")
    elif eval_method == 'fqe':
        if data_path_t2 is None:
            raise ValueError("data_path_t2 must be provided for FQE.")
        return evaluate_cross_dt_fqe(model, dataset_t2, data_path_t2, t1, t2, eps, reward_args=reward_args)
    else:
        raise ValueError(f"Unsupported eval_method '{eval_method}'. Supported: 'wis', 'phwis', 'fqe'.")
    
def main():
    """
    Entry: evaluate a t1 policy checkpoint on a t2 dataset using cross-Δt OPE.
    """

    t1 = 8
    t2 = 8
    pib_type = 'policy'  # 'policy' or 'dataset'
    eval_method = 'wis' # 'wis' or 'fqe'

    print(f'Evaluating t1={t1}h policy on t2={t2}h dataset using {eval_method.upper()}...\n')
    model_checkpoint = get_ckpt_path(t1)
    data_t2_path = get_data_path(t2)
    
    # Load model & dataset
    model = BCQ.load_from_checkpoint(model_checkpoint)
    model.eval()

    dataset_t2 = load_dataset_t2(data_t2_path, t2, action_space='NormThreshold')
    reward_args = SimpleNamespace(**{'R_immed': 0.0, 'R_death': 0.0, 'R_disch': 100.0})
    dataset_t2.reward = remap_rewards(dataset_t2.reward, reward_args)

    start = time.time()
    value, metric = evaluate_cross_dt(
        model, 
        dataset_t2, 
        t1, 
        t2, 
        eps=0.1, 
        clipping=1.438, 
        pib=pib_type,
        eval_method=eval_method,
        data_path_t2=data_t2_path,
        reward_args=reward_args
    )
    end = time.time()

    if eval_method == 'wis':
        print(f'Cross-Δt OPE (t1={t1}h -> t2={t2}h): WIS = {value:.6f}, ESS = {metric:.2f}, Time taken: {end - start:.6f} seconds')
    elif eval_method == 'fqe':
        print(f'Cross-Δt OPE (t1={t1}h -> t2={t2}h): FQE = {value:.6f}, Time taken: {end - start:.6f} seconds')


if __name__ == '__main__':
    main()
