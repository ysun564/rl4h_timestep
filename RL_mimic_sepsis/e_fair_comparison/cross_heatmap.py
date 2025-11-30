"""
Cross-Δt action heatmap generator.

For each pair (t1 -> t2):
- Read the selected checkpoint info from CSVs in:
    F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/rebuttal_phwis/selected
- Load the corresponding model checkpoint.
- Load the t2 episodic dataset (encoded_state + knn_pibs).
- Predict BCQ greedy actions with behavior mask, aggregate to 5x5 counts.
- Aggregate clinician observed actions to 5x5 counts.
- Plot and save two heatmaps (BCQ vs Clinician) with t1, t2 annotated.

Outputs saved to:
    F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/heatmap

Notes:
- Uses CSVs selected_checkpoints_dataset_{t2}h.csv to decide run_dir+iteration.
- Robustly resolves checkpoint as step=XXXX-v1.ckpt then fallback step=XXXX.ckpt.
- Handles both model structures (with/without internal behavior model πb) when computing masks.
"""

import os
import sys
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Fixed paths (Windows focus as requested)
# -----------------------------------------------------------------------------
PROJECT_ROOT_WIN = Path(r"F:\time_step\OfflineRL_FactoredActions")
SELECTED_DIR = PROJECT_ROOT_WIN / r"RL_mimic_sepsis\d_BCQ\figs\cross_pareto\bcnet_phwis\selected"
OUTPUT_DIR = PROJECT_ROOT_WIN / r"RL_mimic_sepsis\d_BCQ\figs\cross_pareto\bcnet_phwis\heatmap_new"

# Add project root to sys.path for module imports
if str(PROJECT_ROOT_WIN) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_WIN))

# Local imports from project
from RL_mimic_sepsis.e_fair_comparison.cross_evaluation import load_dataset_t2
from RL_mimic_sepsis.d_BCQ.src.data import remap_rewards
from RL_mimic_sepsis.d_BCQ.src.model import BCQ as BCQ_new


# -----------------------------------------------------------------------------
# Plot style & action labels
# -----------------------------------------------------------------------------
def set_plot_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
    })


def get_action_ranges(timestep: int, action_space: str = 'NormThreshold'):
    """Return x/y labels for 5x5 action bins. Baseline 4h; y-axis scales by t."""
    if action_space != 'NormThreshold':
        raise ValueError(f'Unsupported action space: {action_space}')
    xranges = ['0', '0-0.08', '0.08-0.20', '0.20-0.45', '≥ 0.45']
    if timestep == 4:
        yranges = ['0', '0-500', '500-1000', '1000-2000', '≥ 2000']
        return xranges, yranges
    base = [0, 500, 1000, 2000]
    scale = timestep / 4.0
    e1 = int(round(base[1] * scale))
    e2 = int(round(base[2] * scale))
    e3 = int(round(base[3] * scale))
    yranges = ['0', f'0-{e1}', f'{e1}-{e2}', f'{e2}-{e3}', f'≥ {e3}']
    return xranges, yranges


# -----------------------------------------------------------------------------
# Checkpoint resolution from selected CSVs
# -----------------------------------------------------------------------------
def _csv_for_t2(t2: int) -> Path:
    return SELECTED_DIR / f"selected_checkpoints_dataset_{t2}h.csv"


def _resolve_ckpt_path(run_dir: Path, iteration: int) -> Path:
    """Try step=XXXX-v1.ckpt then step=XXXX.ckpt under run_dir/checkpoints."""
    ckpt_dir = run_dir / 'checkpoints'
    p1 = ckpt_dir / f"step={int(iteration):04}-v1.ckpt"
    p2 = ckpt_dir / f"step={int(iteration):04}.ckpt"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    # Return preferred naming even if not present; load() will raise with a clear path
    return p1


def get_ckpt_for_pair(t1: int, t2: int) -> Path:
    """Read the selected CSV for t2 and return the checkpoint path for policy_dt == t1."""
    csv_path = _csv_for_t2(t2)
    if not csv_path.exists():
        raise FileNotFoundError(f"Selected CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Expect columns: dataset_dt, policy_dt, threshold, seed, iteration, run_dir, ...
    row = df.loc[df['policy_dt'] == t1]
    if row.empty:
        raise ValueError(f"No row for t1={t1} in {csv_path}")
    row = row.iloc[0]
    run_dir = Path(str(row['run_dir']))  # ensure Windows-style paths are handled
    iteration = int(row['iteration'])
    ckpt_path = _resolve_ckpt_path(run_dir, iteration)
    return ckpt_path


# -----------------------------------------------------------------------------
# Model loading (supports new/old structures)
# -----------------------------------------------------------------------------
def load_model_dynamic(ckpt_path: Path, device: torch.device):
    ckpt_content = torch.load(str(ckpt_path), map_location='cpu')
    state_dict_keys = ckpt_content.get('state_dict', {}).keys()
    if any('Q.πb.0.weight' in k for k in state_dict_keys):
        # Old model with internal behavior πb
        from RL_mimic_sepsis.d_BCQ.src.model_old import BCQ as BCQ_to_load
    else:
        BCQ_to_load = BCQ_new
    model = BCQ_to_load.load_from_checkpoint(checkpoint_path=str(ckpt_path), map_location=device)
    model = model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Dataset and prediction
# -----------------------------------------------------------------------------
_DATASET_CACHE = {}


def get_data_path_t2(t2: int) -> Path:
    return (PROJECT_ROOT_WIN / r"RL_mimic_sepsis\data" /
            f"data_asNormThreshold_dt{t2}h" /
            r"episodes+encoded_state+knn_pibs_k5sqrtn_uniform" / "val_data.pt")


def _get_eval_dataset(data_t2_path: Path, t2: int):
    key = (str(data_t2_path), int(t2))
    if key not in _DATASET_CACHE:
        reward_cfg = SimpleNamespace(**{'R_immed': 0.0, 'R_death': 0.0, 'R_disch': 100.0})
        buf = load_dataset_t2(str(data_t2_path), t2)
        buf.reward = remap_rewards(buf.reward, reward_cfg)
        _DATASET_CACHE[key] = buf
    return _DATASET_CACHE[key]


@torch.no_grad()
def predict_actions_heatmap(model, dataset, device: torch.device, num_actions: int = 25):
    """
    Return flattened arrays of predicted (BCQ) and observed (clinician) action ids across episodes.
    Padded steps after episode end are set to -1 and ignored downstream.
    """
    dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(dl))
    if isinstance(batch, (list, tuple)) and len(batch) >= 6:
        states, actions, _, not_dones, _, estm_pibs = batch[:6]
    else:
        raise RuntimeError('Unexpected dataset batch structure; expected at least 6-tuple.')

    states = states.to(device)
    actions = actions.to(device)
    not_dones = not_dones.to(device)
    estm_pibs = estm_pibs.to(device)

    N, T = states.shape[0], states.shape[1]
    preds, obs = [], []

    for i in tqdm(range(N), desc='Predicting (cross-Δt)'):
        lng = (not_dones[i, :, 0].sum() + 1).item()

        q_out = model.Q(states[i])
        if isinstance(q_out, (list, tuple)) and len(q_out) >= 2:
            q, log_pibs = q_out[0], q_out[1]
            estm = log_pibs.exp()
        else:
            q = q_out
            estm = estm_pibs[i]

        # Behavior mask via threshold
        thr = float(getattr(model, 'threshold', 0.0))
        mx = estm.max(dim=1, keepdim=True).values
        # avoid zero-division; if mx==0, fall back to all-ones mask
        mask = torch.where(mx > 0, (estm / mx) > thr, torch.ones_like(estm, dtype=torch.bool))
        mask = mask.to(q.dtype)

        a_id = (mask * q + (1.0 - mask) * torch.finfo(q.dtype).min).argmax(dim=1).detach().cpu().numpy()
        a_id[int(lng):] = -1
        preds.append(a_id)

        a_obs = actions[i, :, 0].detach().cpu().numpy()
        a_obs[int(lng):] = -1
        obs.append(a_obs)

    pred_flat = np.concatenate(preds)
    obs_flat = np.concatenate(obs)
    return pred_flat, obs_flat


def to_5x5_counts(arr_1d, num_actions: int = 25):
    s = pd.Series(arr_1d)
    counts = s.value_counts().reindex(range(num_actions), fill_value=0).values.reshape(5, 5)
    return counts


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_and_save_heatmaps(counts_policy, counts_clin, t1: int, t2: int, out_dir: Path, action_space='NormThreshold'):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=600)

    # BCQ heatmap
    ax = axes[0]
    sns.heatmap(counts_policy, ax=ax, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
    ax.invert_yaxis()
    xr, yr = get_action_ranges(t2, action_space)
    ax.set_title(f'BCQ ($t_\pi$={t1}h → $t_D$={t2}h)')
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5]); ax.set_xticklabels(xr, rotation=0); ax.xaxis.set_tick_params(pad=5)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5]); ax.set_yticklabels(yr, rotation=90); ax.yaxis.set_tick_params(pad=5)

    # Clinician heatmap
    ax = axes[1]
    sns.heatmap(counts_clin, ax=ax, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
    ax.invert_yaxis()
    xr, yr = get_action_ranges(t2, action_space)
    ax.set_title(f'Clinician ($t_D$={t2}h)')
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5]); ax.set_xticklabels(xr, rotation=0); ax.xaxis.set_tick_params(pad=5)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5]); ax.set_yticklabels(yr, rotation=90); ax.yaxis.set_tick_params(pad=5)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f'cross_heatmap_t1-{t1}h_to_t2-{t2}h.pdf'
    fig.savefig(str(out_pdf), bbox_inches='tight')
    plt.close(fig)
    print(f'Saved heatmap: {out_pdf}')


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_one_pair(t1: int, t2: int, device_str: str = 'auto'):
    # Resolve checkpoint from selected CSVs
    ckpt_path = get_ckpt_for_pair(t1, t2)
    print(f"Using checkpoint for t1={t1} -> t2={t2}: {ckpt_path}")

    # Device
    if device_str == 'cpu':
        device = torch.device('cpu')
    elif device_str.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model & dataset
    model = load_model_dynamic(ckpt_path, device)
    data_t2_path = get_data_path_t2(t2)
    dataset_t2 = _get_eval_dataset(data_t2_path, t2)

    # Predict actions and build counts
    pred_flat, obs_flat = predict_actions_heatmap(model, dataset_t2, device=device, num_actions=25)
    counts_policy = to_5x5_counts(pred_flat, num_actions=25)
    counts_clin = to_5x5_counts(obs_flat, num_actions=25)

    # Save raw counts for reproducibility
    counts_dir = OUTPUT_DIR / 'counts'
    counts_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts_policy).to_csv(counts_dir / f'counts_policy_t1-{t1}h_to_t2-{t2}h.csv', index=False, header=False)
    pd.DataFrame(counts_clin).to_csv(counts_dir / f'counts_clinician_t2-{t2}h.csv', index=False, header=False)

    # Plot
    plot_and_save_heatmaps(counts_policy, counts_clin, t1, t2, OUTPUT_DIR, action_space='NormThreshold')


def get_counts_for_pair(t1: int, t2: int, device_str: str = 'auto'):
    """Return (counts_policy 5x5, counts_clin 5x5) for one (t1->t2) pair."""
    ckpt_path = get_ckpt_for_pair(t1, t2)

    if device_str == 'cpu':
        device = torch.device('cpu')
    elif device_str.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_model_dynamic(ckpt_path, device)
    data_t2_path = get_data_path_t2(t2)
    dataset_t2 = _get_eval_dataset(data_t2_path, t2)

    pred_flat, obs_flat = predict_actions_heatmap(model, dataset_t2, device=device, num_actions=25)
    counts_policy = to_5x5_counts(pred_flat, num_actions=25)
    counts_clin = to_5x5_counts(obs_flat, num_actions=25)
    return counts_policy, counts_clin


def plot_heatmap_panel_for_t2(t2: int, device_str: str = 'auto'):
    """Generate a 1x5 panel: t1 in [1,2,4,8] BCQ heatmaps + clinician at fixed t2."""
    set_plot_style()
    t1_list = [1, 2, 4, 8]
    counts_list = []
    counts_clin = None
    # Collect counts and find a common vmax for consistent color scaling
    vmax = 0
    for t1 in t1_list:
        try:
            c_pol, c_cli_local = get_counts_for_pair(t1, t2, device_str=device_str)
        except Exception as e:
            print(f"Panel: failed to get counts for t1={t1} -> t2={t2}: {e}")
            # Use zeros to keep panel shape
            c_pol = np.zeros((5, 5), dtype=int)
            c_cli_local = counts_clin if counts_clin is not None else np.zeros((5, 5), dtype=int)
        counts_list.append(c_pol)
        if counts_clin is None:
            counts_clin = c_cli_local
        vmax = max(vmax, int(np.max(c_pol)), int(np.max(counts_clin)))

    # Avoid zero vmax
    if vmax <= 0:
        vmax = 1

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6), dpi=600)
    xr, yr = get_action_ranges(t2, 'NormThreshold')

    # Plot t1 policy heatmaps
    for idx, t1 in enumerate(t1_list):
        ax = axes[idx]
        sns.heatmap(counts_list[idx], ax=ax, square=True, annot=True, fmt='d', cbar=False, cmap='Blues', vmin=0, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(f'BCQ ($t_\pi$={t1}h → $t_D$={t2}h)')
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5]); ax.set_xticklabels(xr, rotation=0); ax.xaxis.set_tick_params(pad=5)
        ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5]); ax.set_yticklabels(yr, rotation=90); ax.yaxis.set_tick_params(pad=5)

    # Plot clinician
    ax = axes[4]
    sns.heatmap(counts_clin, ax=ax, square=True, annot=True, fmt='d', cbar=False, cmap='Blues', vmin=0, vmax=vmax)
    ax.invert_yaxis()
    ax.set_title(f'Clinician ($t_D$={t2}h)')
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5]); ax.set_xticklabels(xr, rotation=0); ax.xaxis.set_tick_params(pad=5)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5]); ax.set_yticklabels(yr, rotation=90); ax.yaxis.set_tick_params(pad=5)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_DIR / f'cross_heatmap_panel_t2-{t2}h.pdf'
    fig.savefig(str(out_pdf), bbox_inches='tight')
    plt.close(fig)
    print(f'Saved panel heatmap: {out_pdf}')


def main():
    # Mode: run a grid of pairs or a single pair.
    run_type = 'grid'  # 'single' or 'grid'
    device_str = 'auto'  # 'cpu' | 'cuda:0' | 'auto'

    if run_type == 'single':
        t1, t2 = 4, 8
        run_one_pair(t1, t2, device_str=device_str)
    else:
        pairs = [
            (1, 1), (2, 2), (4, 4), (8, 8),
            (1, 2), (1, 4), (1, 8),
            (2, 1), (2, 4), (2, 8),
            (4, 1), (4, 2), (4, 8),
            (8, 1), (8, 2), (8, 4),
        ]
        for (t1, t2) in pairs:
            print('=' * 80)
            print(f'Generating heatmap t1={t1}h -> t2={t2}h')
            try:
                run_one_pair(t1, t2, device_str=device_str)
            except Exception as e:
                print(f'Failed for t1={t1}, t2={t2}: {e}')

        # Additionally, for each fixed t2, generate a 1x5 panel: t1 in [1,2,4,8] + clinician@t2
        for t2_fixed in [1, 2, 4, 8]:
            print('-' * 80)
            print(f'Generating panel for fixed $t_2$={t2_fixed}h: $t_1$ in [1,2,4,8]')
            try:
                plot_heatmap_panel_for_t2(t2_fixed, device_str=device_str)
            except Exception as e:
                print(f'Failed to generate panel for t2={t2_fixed}: {e}')


if __name__ == '__main__':
    # Allow MKL to load duplicate if needed on Windows
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
