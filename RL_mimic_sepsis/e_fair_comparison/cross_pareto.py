"""Generate Pareto frontiers for cross-dataset BCQ evaluations."""

import logging
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

matplotlib.use('Agg')


ROOT_LOG = Path(
    r'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/d_BCQ/logs/BCQ_rebuttal_phwis'
)
OUTPUT_DIR = Path(
    r'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/rebuttal_phwis/selected'
)
TIMESTEPS = [1, 2, 4, 8]
ESS_CUTOFF = {1: 40, 2: 60, 4: 145, 8: 275}
COLORS = {1: 'tab:red', 2: 'tab:orange', 4: 'tab:green', 8: 'tab:blue'}
CROSS_PATTERN = 'cross_metrics_phwis_pib-dataset_t1-{policy}h_to_t2-{dataset}h.csv'


def setup_logging():
    """Configure logging for console output."""
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_threshold_seed(name):
    """Extract threshold and seed identifiers from a run folder name."""
    thr_match = re.search(r'threshold(\d+\.?\d*)', name)
    seed_match = re.search(r'seed(\d+)', name)
    if not thr_match or not seed_match:
        logging.warning('Skipping folder without threshold/seed info: %s', name)
        return None, None
    return float(thr_match.group(1)), int(seed_match.group(1))


def find_run_dirs(policy_dt):
    """Yield all run directories for a given policy timestep."""
    root = ROOT_LOG / f'BCQ_asNormThreshold_dt{policy_dt}h_grid_latent128d'
    if not root.exists():
        logging.warning('Policy directory missing: %s', root)
        return []
    return [p for p in root.glob(f'dt{policy_dt}_threshold*seed*') if p.is_dir()]


def load_metrics(run_dir, policy_dt, dataset_dt):
    """Load evaluation metrics for a policy on a given dataset."""
    if policy_dt == dataset_dt:
        metrics_path = run_dir / CROSS_PATTERN.format(policy=policy_dt, dataset=dataset_dt)
    else:
        metrics_path = run_dir / CROSS_PATTERN.format(policy=policy_dt, dataset=dataset_dt)
    if not metrics_path.exists():
        logging.debug('Metrics file not found: %s', metrics_path)
        return None
    df = pd.read_csv(metrics_path)
    logging.debug('Reading metrics: %s with columns %s and %d rows', metrics_path, list(df.columns), len(df))
    # Accept both historical column names: val_wis or val_phwis
    value_col = None
    if 'val_wis' in df.columns:
        value_col = 'val_wis'
    elif 'val_phwis' in df.columns:
        value_col = 'val_phwis'
    required_cols = {'iteration', 'val_ess'} | ({value_col} if value_col else set())
    if df.empty or (value_col is None) or required_cols.difference(df.columns):
        logging.debug('Metrics file invalid or missing value column: %s; columns=%s', metrics_path, list(df.columns))
        return None
    df = df[['iteration', 'val_ess', value_col]].copy()
    if value_col != 'val_wis':
        df = df.rename(columns={value_col: 'val_wis'})
    logging.debug('Loaded metrics OK: %s (rows=%d)', metrics_path, len(df))
    threshold, seed = parse_threshold_seed(run_dir.name)
    df['threshold'] = threshold
    df['seed'] = seed
    df['policy_dt'] = policy_dt
    df['dataset_dt'] = dataset_dt
    df['run_dir'] = str(run_dir)
    df['metrics_path'] = str(metrics_path)
    return df


def collect_dataset_data(dataset_dt):
    """Collect evaluation metrics for all policies on a target dataset."""
    frames = []
    for policy_dt in TIMESTEPS:
        run_dirs = find_run_dirs(policy_dt)
        if not run_dirs:
            continue
        for run_dir in run_dirs:
            df = load_metrics(run_dir, policy_dt, dataset_dt)
            if df is not None:
                frames.append(df)
    if not frames:
        raise RuntimeError(f'No metrics available for dataset {dataset_dt}h')
    return pd.concat(frames, ignore_index=True)


def pareto_front(points):
    """Return indices of Pareto-optimal points for maximization in 2D."""
    if not points:
        return []
    sorted_pts = sorted(enumerate(points), key=lambda x: (x[1][0], x[1][1]), reverse=True)
    best = -np.inf
    front_idx = []
    for idx, (_, (_, ess)) in enumerate(sorted_pts):
        if idx == 0 or ess > best:
            front_idx.append(sorted_pts[idx][0])
            best = ess
    return front_idx


def plot_dataset(dataset_dt, df):
    """Plot Pareto frontiers for all policies and return selected checkpoints."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4.5), dpi=300)
    handles = []
    labels = []
    selected_records = []
    selected_plotted = False
    for policy_dt in TIMESTEPS:
        policy_df = df[df['policy_dt'] == policy_dt]
        if policy_df.empty:
            continue
        pts = policy_df[['val_ess', 'val_wis']].values.tolist()
        front_idx = pareto_front(pts)
        if not front_idx:
            continue
        front = policy_df.iloc[front_idx].sort_values(by='val_ess')
        line, = plt.plot(
            front['val_ess'],
            front['val_wis'],
            ls='--',
            lw=1.5,
            marker='o',
            ms=4,
            color=COLORS.get(policy_dt, 'black')
        )
        handles.append(line)
        labels.append(f'{policy_dt} h policy')
        cutoff = ESS_CUTOFF.get(dataset_dt)
        if cutoff is not None:
            eligible = policy_df[policy_df['val_ess'] >= cutoff]
            if not eligible.empty:
                best_idx = eligible['val_wis'].idxmax()
                best = policy_df.loc[best_idx]
                plt.scatter(
                    [best['val_ess']],
                    [best['val_wis']],
                    facecolors='none',
                    edgecolors='red',
                    s=90,
                    linewidths=1.5
                )
                # Mark that we should add the legend entry for selected checkpoints later
                selected_plotted = True
                selected_records.append(
                    {
                        'dataset_dt': dataset_dt,
                        'policy_dt': policy_dt,
                        'threshold': best['threshold'],
                        'seed': best['seed'],
                        'iteration': int(best['iteration']),
                        'val_ess': float(best['val_ess']),
                        'val_wis': float(best['val_wis']),
                        'run_dir': best['run_dir'],
                        'metrics_path': best['metrics_path']
                    }
                )
    cutoff = ESS_CUTOFF.get(dataset_dt)
    if cutoff is not None:
        plt.axvline(cutoff, ls=':', lw=1.5, color='grey')
        handles.append(Line2D([], [], ls=':', color='grey'))
        labels.append(f'ESS cutoff = {cutoff}')
    # Ensure 'Selected checkpoint' appears last in the legend ordering
    if selected_plotted:
        handles.append(
            Line2D(
                [],
                [],
                marker='o',
                markersize=8,
                markerfacecolor='none',
                markeredgecolor='red',
                linestyle='None'
            )
        )
        labels.append('Selected checkpoint')
    plt.xlabel('Effective Sample Size')
    plt.ylabel('Estimated Policy Value (WIS)')
    plt.title(f'Pareto Frontier by Policy on {dataset_dt}h Dataset')
    plt.xlim(left=0)
    plt.ylim(94, 100.5)
    if handles:
        plt.legend(handles, labels, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    pdf_path = OUTPUT_DIR / f'cross_pareto_dataset_{dataset_dt}h.pdf'
    plt.savefig(pdf_path)
    plt.close()
    logging.info('Saved Pareto figure: %s', pdf_path)
    return selected_records


def main():
    """Entry point for generating cross-dataset Pareto figures."""
    setup_logging()
    for dataset_dt in TIMESTEPS:
        try:
            df = collect_dataset_data(dataset_dt)
        except RuntimeError as exc:
            logging.error(exc)
            continue
        selected = plot_dataset(dataset_dt, df)
        if selected:
            selected_df = pd.DataFrame(selected)
            out_path = OUTPUT_DIR / f'selected_checkpoints_dataset_{dataset_dt}h.csv'
            selected_df.to_csv(out_path, index=False)
            logging.info('Saved selected checkpoints: %s', out_path)
        else:
            logging.info('No checkpoints exceeded cutoff for dataset %dh', dataset_dt)


if __name__ == '__main__':
    main()
