from typing import Any, Literal
from pathlib import Path

import pytorch_lightning as pl
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime

def ewma_vectorized(data, alpha, offset=None, dtype=None, order: Literal['C', 'F', 'A'] = 'C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out



def clean_and_rename_metrics(directory):
    """
    In the given directory (recursively), delete all 'metrics_new.csv' files
    and rename all 'metrics_deepcopy_new.csv' files to 'metrics_new.csv'.
    """
    base_path = Path(directory)
    
    # Step 1: Delete all 'metrics_new.csv'
    for file in base_path.rglob('metrics_new.csv'):
        try:
            file.unlink()
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")
    
    # Step 2: Rename all 'metrics_deepcopy_new.csv' to 'metrics_n`ew.csv'
    for file in base_path.rglob('metrics_deepcopy_new.csv'):
        new_name = file.with_name('metrics_new.csv')
        try:
            file.rename(new_name)
            print(f"Renamed: {file} -> {new_name}")
        except Exception as e:
            print(f"Failed to rename {file}: {e}")


clean_and_rename_metrics(rf"F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\d_BCQ\logs\mimic_dBCQ_8h_grid")


def save_everything():
    return 

    def offline_evaluation_dr(self, eval_buffer, weighted=True, eps=0.01,
                                analyze: bool = False, clipping: float = 95.0):
            """Offline evaluation function. Returns estimated WIS value and ESS. 
            'analyze': determines whether analyze the IR distribution and draw figure.
            'clipping': automatically distinguish the clipping method (threshold/percentile).
            """
            states, actions, rewards, not_dones, pibs, estm_pibs = eval_buffer
            rewards = rewards[:, :, 0].cpu().numpy()
            n, horizon, _ = states.shape
            gamma = self.eval_discount  
            
            ir = np.ones((n, horizon))
            v_hat_all    = np.zeros((n, horizon))         
            q_hat_logged = np.zeros((n, horizon))         
            traj_len     = np.zeros(n, dtype=int)

            for idx in range(n):
                # 'lng' is the length of the trajectory.
                # TODO: Read paper <Off By A Beat>
                # TODO: Check if the step length of per-step clipping is correct.
                lng = (not_dones[idx, :, 0].sum() + 1).item()
                traj_len[idx] = lng

                q, imt, _ = self.Q(states[idx])
                # TODO: detach()
                q_np = q.detach().cpu().numpy()
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True).values > self.threshold).float()

                # Use large negative number to mask actions from argmax.
                a_id = (imt * q 
                        + (1. - imt) 
                        * torch.finfo().min).argmax(axis=1).cpu().numpy()
                pie_soft = np.zeros((horizon, self.num_actions))
                pie_soft += eps * estm_pibs[idx].cpu().numpy() # Soften using training behavior policy
                pie_soft[range(horizon), a_id] += (1.0 - eps)

                # Compute importance sampling ratios.
                a_obs = actions[idx, :, 0]
                ir[idx, :lng] = (pie_soft[range(lng), a_obs[:lng].cpu().numpy()] 
                                / pibs[idx, range(lng), a_obs[:lng]].cpu().numpy())
                
                # Calculate V̂(s_t) and Q̂(s_t,a_t_logged) for DR.
                v_hat_all[idx] = (pie_soft * q_np).sum(axis=1)
                q_hat_logged[idx] = q_np[np.arange(horizon), a_obs.cpu().numpy()]
                
            rho_prefix  = np.cumprod(ir, axis=1) 
            rho_final  = rho_prefix[:, -1]

            # Automaticlly distinguishes the clipping method. If clipping passed is.
            # TODO: Add comments.
            rho_final_clip = []
            if clipping > 100:
                threshold = clipping
                rho_final_clip = np.clip(rho_final, 0, threshold)

            elif clipping >= 100:
                M = np.percentile(rho_final, clipping)
                rho_prefix_clip = np.minimum(rho_prefix, M)
                rho_final_clip  = np.minimum(rho_final,  M)
                ir_clip = np.minimum(ir, M)

            elif clipping < 5 and clipping > 0:
                # Calculated clipped importance ratio via a threshold list 
                # based on valid step size for each trajectory.
                # 'clipping' is by default 1.438 (19th root of 1000), 
                # where 19 is the maximum step size for 4-h time step.
                # For 4-h: 'ir': (2785, 19) -> 'valid_step_list': (2785) -> 'threshold_list' -> (2785)   
                valid_step_list = np.count_nonzero(ir, axis=1)
                threshold_list = clipping ** valid_step_list
                rho_final_clip = np.clip(rho_final, 0, threshold_list)

            elif clipping == 0:
                # Doing nothing.
                rho_final_clip = rho_final


            if analyze:
                fig, _ = self.rho_fig_pair(rho_final, rho_final_clip, tag=f'timestep_threshold_{timestep}h_')
                self.analyze_rho_clip(rho_final_clip)
            
            dr_ret = np.zeros(n)                             # V_DR at t=0 for each traj
            for i in range(n):
                v_next = 0.0
                for t in range(traj_len[i] - 1, -1, -1):     # backward pass only over real steps
                    v_next = (v_hat_all[i, t] +
                            ir[i, t] *
                            (rewards[i, t] + gamma * v_next - q_hat_logged[i, t]))
                dr_ret[i] = v_next

            if weighted:                                     
                w = rho_final_clip
                w = np.array(w)
                dr_est = (w * dr_ret).sum() / w.sum() 
                ess = (w.sum())**2 / np.square(w).sum()
            else:                                           
                dr_est = dr_ret.mean()
                ess = n
            return dr_est, ess


    def stats(self, arr, name):
            print(f"{name}: min={arr.min():.4g}, max={arr.max():.4g}, "
                f"mean={arr.mean():.4g}, std={arr.std():.4g}")
            
    def analyze_ir(self, ir, save_fig=False, tag=""):
        """
        Analyse the step-wise IS ratios tensor `ir`  (shape = n traj × H steps).

        Parameters
        ----------
        ir : np.ndarray or torch.Tensor
            Step-IS ratios  r_{it} = pi(a|s)/mu(a|s)
        save_fig : bool
            If True, save a log10-histogram png.
        tag : str
            Optional suffix for the png filename.
        """
        # # -------- tensor → numpy ----------
        # if not isinstance(ir, np.ndarray):``
        #     ir = ir.cpu().numpy()
        n, H = ir.shape
        flat = ir.ravel()

        print(f"ir.shape = {ir.shape}")
        self.stats(ir, "ir (all steps)")

        # -------- per-step stats ----------
        mean_t = ir.mean(axis=0)          # (H,)
        std_t  = ir.std(axis=0)
        print(f"per-step mean range : {mean_t.min():.3g} – {mean_t.max():.3g}")
        print(f"per-step std  range : {std_t.min():.3g} – {std_t.max():.3g}")

        # -------- quantiles ----------
        qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        qv = np.quantile(flat, qs)
        print("quantiles:", ", ".join([f"{p*100:.0f}%={v:.3g}" for p, v in zip(qs, qv)]))

        # -------- extreme counts ----------
        small = (flat < 1e-3).sum()
        big   = (flat > 1e1).sum()
        tot   = flat.size
        print(f"(<1e-3)  : {small}  ({small/tot:.2%})")
        print(f"(>1e+1)  : {big}    ({big/tot:.2%})")

        # -------- histogram (log10 scale) ----------
        if save_fig:
            plt.figure(figsize=(6,3.5))
            plt.hist(np.log10(flat + 1e-12), bins=np.linspace(-12, 4, 80).tolist(),
                    color="steelblue", alpha=.85)
            plt.yscale("log")
            plt.xlabel("log10(step IS ratio)")
            plt.ylabel("count (log)")
            plt.title("Distribution of step-wise IS ratios")
            plt.grid(ls="--", alpha=.3)
            plt.tight_layout()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"ir_hist_{tag}_{ts}.png" if tag else f"ir_hist_{ts}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"[saved → {fname}]")

    def analyze_rho_clip(self, rho_clip):

        print('Analyze rho_clip (clipped weights)')
        print(f"rho_clip.shape = {rho_clip.shape}")

        self.stats(rho_clip, "weights(clipped)")

        clip_threshold = rho_clip.max()
        print(f'Clipping Threshold: {clip_threshold:3g}')
        num_clipped = np.sum(rho_clip == clip_threshold)
        total_elem  = rho_clip.size
        print(f"Clipped Count: {num_clipped}/{total_elem} "
            f"({100*num_clipped/total_elem:.2f}%)")
        print('\n')
            
    def calculate_ess(self, w):
            return (w.sum())**2 / (w**2).sum()

    def rho_fig(self, rho, type='clip'):
        from matplotlib.ticker import ScalarFormatter, FixedLocator
        plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size"  : 10,         # base font
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "mathtext.fontset": "stix",
        })
        logw = np.log10(rho)
        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
        bins = np.linspace(-12, 18, 120)
        plt.hist(logw, bins=bins.tolist(), color='steelblue', alpha=.85)
        plt.axvline(3, color='red', ls=':', lw=1, label=r'clip = $10^{3}$')
        plt.yscale('log')
        plt.grid(ls='--', alpha=.3)
        plt.xlabel(r'$\log_{10}\,W$')
        plt.ylabel('Count (log scale)')

        xticks = [-12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(FixedLocator([]))

        ax.grid(ls='--', alpha=.3, which='major')
        title = ('Distribution of Raw Trajectory Weights'
                if type == 'raw'
                else 'Distribution of Clipped Trajectory Weights')
        ax.set_title(title, pad=6)

        ax.legend(frameon=False)
        fig.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if type == 'raw':
            plt.savefig(f"rho_raw_{timestamp}.png", dpi=300)
        elif type == 'clip':    
            plt.savefig(f"rho_clip_{timestamp}.png", dpi=300)

    def rho_fig_pair(self, rho_raw, rho_clip, *, bins=np.linspace(-12, 18, 120), ymax_pad=1.2, save=True, tag=''):
        from matplotlib.ticker import ScalarFormatter, FixedLocator

        plt.rcParams.update({
            "font.family"      : "Times New Roman",
            "font.size"        : 10,
            "axes.labelsize"   : 10,
            "axes.titlesize"   : 11,
            "legend.fontsize"  : 9,
            "mathtext.fontset" : "stix",
        })

        # ───  Pre-compute log10 weights  ────────────────────────
        log_raw  = np.log10(np.asarray(rho_raw,  dtype=float))
        log_clip = np.log10(np.asarray(rho_clip, dtype=float))

        # ───  Figure & Axes  ────────────────────────────────────
        fig, axs = plt.subplots(
            nrows=1, ncols=2, sharey=True,
            figsize=(6.9, 3.2), dpi=300,
            gridspec_kw=dict(wspace=0.08)
        )
        
        # Helper for both panels
    def _one_hist(ax, data, title):
        ax.hist(data, bins=bins, color='steelblue', alpha=.85)
        # ax.axvline(3, color='crimson', ls=':', lw=0.5,
        #         label=r'Threshold $10^{3}$')
        ax.set_yscale('log')
        ax.grid(ls='--', alpha=.3, which='major')
        ax.set_xlabel(r'$\log_{10} W$')
        ax.set_title(title, pad=4)
        # Major x-ticks
        xticks = [-12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        # Format y-axis in plain numbers (not scientific)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(FixedLocator([]))
        from matplotlib.lines import Line2D
        
        ax.legend(frameon=False, loc='upper right')

        _one_hist(axs[0], log_raw,  f'Raw Weights (Δt = {timestep} h)')
        _one_hist(axs[1], log_clip, f'Clipped Weights (Δt = {timestep} h)')

        # Give both panels the same y-axis upper bound for cleaner comparison
        ymax = max(ax.get_ylim()[1] for ax in axs) * ymax_pad
        for ax in axs:
            ax.set_ylim(bottom=1, top=ymax)

        axs[0].set_ylabel('Count (log scale)')

        fig.tight_layout()

        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            import os
            os.makedirs('rho_pair_test', exist_ok = True)
            fig.savefig(f"rho_pair_test/rho_pair_{tag}{ts}.png", bbox_inches='tight', dpi=300)

    return fig, axs

    def pd_is(self, ir, discounted_rewards, weighted = False):
        ratio_pdis = ir 

        if weighted:
            weights  = ratio_pdis / (np.sum(ratio_pdis, axis=0, keepdims=True) + 1e-12)
            step_matrix  = weights  * discounted_rewards
            pdis = step_matrix.sum() 
        else:
            traj_returns = np.sum((ratio_pdis * discounted_rewards), axis = 1)
            pdis = traj_returns.mean()

        w = ratio_pdis[:, -1]
        ess = (w.sum()**2) / np.sum(np.square(w))  
        return float(pdis), float(ess)

    def ph_is(self, ir, discounted_rewards, traj_lens):
        """
        # TODO: Use small test case to test the function.
        """
        n = ir.shape[0]

        # prefix products ρ_{1:t}
        rho_prefix = np.cumprod(ir, axis=1)

        # final cumulative weight ρ_{1:L_i}  and  return  R_i
        final_w = rho_prefix[np.arange(n), traj_lens - 1]               
        returns = np.array([discounted_rewards[i, :L].sum()
                            for i, L in enumerate(traj_lens)])         

        # group trajectories by their horizon
        phwis_est = 0.0
        for L in np.unique(traj_lens):
            idx = np.where(traj_lens == L)[0]       
            w_L  = final_w[idx]
            r_L  = returns[idx]

            # per-horizon WIS estimate
            v_L = (np.sum(w_L * r_L) / np.sum(w_L)  + 1e-12)

            # weight by proportion of trajectories with that horizon
            phwis_est += (len(idx) / n) * v_L

        ess = (final_w.sum()**2) / (np.square(final_w).sum())
        return float(phwis_est), float(ess)

    def analyze_rewards(self, rewards):
        """Analyzes the 'rewards' list extracted from the 'eval_buffer' in 'offline_evaluation'.
        Prints: 
        1. The shape before/after the adjustment.
        2. The number of trajectories that have reward = 100 and reward = 0. 
        3. The sum of '2.'. 
        """
        print(f'rewards before adjusting: {rewards.shape}')
        rewards = rewards[:, :, 0].cpu().numpy()
        print(f'rewards after adjusting: {rewards.shape}')
        rewards_analysis = rewards.sum(-1)
        print(f'Count 100 in rewards_analysis: {np.sum(rewards_analysis == 100)}')
        print(f'Count   0 in rewards_analysis: {np.sum(rewards_analysis == 0)}')
        print(f'Count 100 + Count 0: {np.sum(rewards_analysis == 100) + np.sum(rewards_analysis == 0)}')