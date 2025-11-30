"""BCQ stability diagnostics limited to mean±CI, cross-seed std, and seed Spearman rho."""

import logging
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')


ROOT_BASE = Path(
    r'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/d_BCQ/logs/BCQ_ml4h'
)
METRICS_FILENAME = 'metrics_100multiple_eps0.1.csv'
OUTPUT_DIR_BASE = Path(
    r'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/d_BCQ/notebooks/outputs'
)
DT_LIST = [1, 2, 4, 8]

OUTPUT_DIR = None
SUMMARY_PATH = None


def setup_logging():
    """Configure root logger for console output."""
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_threshold_seed(folder_name):
    """Parse threshold and seed id from folder name."""
    thr_match = re.search(r'threshold(\d+\.?\d*)', folder_name)
    seed_match = re.search(r'seed(\d+)', folder_name)
    if not thr_match or not seed_match:
        logging.warning('Could not parse threshold/seed from folder %s', folder_name)
        return None, None
    return float(thr_match.group(1)), int(seed_match.group(1))


def iter_metrics_files(root_dir):
    """Yield all metric CSV paths under the given log root."""
    if not root_dir.exists():
        raise FileNotFoundError(f'Log directory not found: {root_dir}')
    for metrics_path in root_dir.rglob(METRICS_FILENAME):
        if metrics_path.is_file():
            yield metrics_path


def load_metrics(root_dir):
    """Load per-run metrics and return a concatenated DataFrame."""
    records = []
    for metrics_path in iter_metrics_files(root_dir):
        threshold, seed = parse_threshold_seed(metrics_path.parent.name)
        if threshold is None or seed is None:
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            logging.warning('Metrics file empty: %s', metrics_path)
            continue
        if {'iteration', 'step', 'val_wis', 'val_ess'} - set(df.columns):
            logging.warning('Metrics file missing required columns: %s', metrics_path)
            continue
        df = df[['step', 'val_wis', 'val_ess']].copy()
        df['threshold'] = threshold
        df['seed'] = seed
        df.rename(columns={'val_wis': 'WIS', 'val_ess': 'ESS'}, inplace=True)
        records.append(df[['threshold', 'seed', 'step', 'WIS', 'ESS']])
    if not records:
        raise RuntimeError('No metrics loaded. Check directory structure or parsing logic.')
    combined = pd.concat(records, ignore_index=True)
    combined.sort_values(['threshold', 'seed', 'step'], inplace=True)
    return combined


def compute_step_summary(df):
    """Aggregate seed statistics per (threshold, step) pair with 95% CI."""
    grouped = df.groupby(['threshold', 'step'])
    summary = grouped.agg(
        WIS_mean=('WIS', 'mean'),
        WIS_std=('WIS', 'std'),
        ESS_mean=('ESS', 'mean'),
        ESS_std=('ESS', 'std'),
        n_seeds=('seed', 'nunique')
    ).reset_index()
    summary.sort_values(['threshold', 'step'], inplace=True)

    try:
        from scipy.stats import t as student_t

        def t_crit(dfree):
            return float(student_t.ppf(0.975, dfree)) if dfree >= 1 else 0.0

    except Exception:
        t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

        def t_crit(dfree):
            if dfree < 1:
                return 0.0
            return float(t_table.get(dfree, 2.0))

    summary['WIS_se'] = summary['WIS_std'] / (summary['n_seeds'] ** 0.5)
    summary['ESS_se'] = summary['ESS_std'] / (summary['n_seeds'] ** 0.5)

    def _ci_half(row, col):
        n = int(row['n_seeds']) if not pd.isna(row['n_seeds']) else 0
        dfree = max(0, n - 1)
        val = row[col]
        if pd.isna(val):
            return 0.0
        return t_crit(dfree) * val

    summary['WIS_CI'] = summary.apply(lambda r: _ci_half(r, 'WIS_se'), axis=1)
    summary['ESS_CI'] = summary.apply(lambda r: _ci_half(r, 'ESS_se'), axis=1)
    return summary


def ensure_output_dir():
    """Create the per-dt output directory."""
    if OUTPUT_DIR is None:
        raise RuntimeError('OUTPUT_DIR is not set')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_mean_ci(summary_df, metric, threshold):
    """Plot mean ±95% CI curves for a metric."""
    if OUTPUT_DIR is None:
        raise RuntimeError('OUTPUT_DIR is not configured')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df['step'], summary_df[f'{metric}_mean'], label='mean', color='C0')
    half = summary_df[f'{metric}_CI']
    lower = summary_df[f'{metric}_mean'] - half
    upper = summary_df[f'{metric}_mean'] + half
    ax.fill_between(summary_df['step'], lower, upper, color='C0', alpha=0.25, label='±95% CI')
    ax.set_title(
        f'{metric} mean ±95% CI | dataset=validation, seeds=5, checkpoints=100'
    )
    ax.set_xlabel('step')
    ax.set_ylabel(f'{metric} (validation)')
    if metric == 'WIS':
        ax.set_ylim(85, 100)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    filename = OUTPUT_DIR / f'{metric.lower()}_mean_ci_thr{threshold}.png'
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def summarize_cross_seed_std(df):
    """Summarize per-step cross-seed std for WIS and ESS."""
    records = []
    for threshold in sorted(df['threshold'].unique()):
        thr_df = df[df['threshold'] == threshold]
        pivot = thr_df.pivot_table(index='step', columns='seed')
        wis_std = pivot['WIS'].std(axis=1, ddof=1)
        ess_std = pivot['ESS'].std(axis=1, ddof=1)
        def stats(series):
            vals = series.dropna()
            if vals.empty:
                return {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'iqr': np.nan}
            return {
                'mean': float(vals.mean()),
                'std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                'median': float(vals.median()),
                'iqr': float(vals.quantile(0.75) - vals.quantile(0.25))
            }
        wis_stats = stats(wis_std)
        ess_stats = stats(ess_std)
        records.append(
            {
                'threshold': threshold,
                'WIS_std_mean_over_steps': wis_stats['mean'],
                'WIS_std_std_over_steps': wis_stats['std'],
                'WIS_std_median': wis_stats['median'],
                'WIS_std_iqr': wis_stats['iqr'],
                'ESS_std_mean_over_steps': ess_stats['mean'],
                'ESS_std_std_over_steps': ess_stats['std'],
                'ESS_std_median': ess_stats['median'],
                'ESS_std_iqr': ess_stats['iqr']
            }
        )
    return pd.DataFrame(records)


def summarize_seed_spearman(df):
    """Compute per-seed Spearman rho between ESS and WIS and summarize per threshold."""
    try:
        from scipy.stats import spearmanr

        def rho(x, y):
            if len(x) < 2:
                return np.nan
            result = spearmanr(x, y, nan_policy='omit')
            try:
                stat = result.statistic  # type: ignore[attr-defined]
            except AttributeError:
                stat = result[0] if isinstance(result, tuple) else result  # type: ignore[index]
            if isinstance(stat, (list, tuple, np.ndarray)):
                stat = np.asarray(stat, dtype=float)
                stat = stat.reshape(-1)[0]
            return float(np.asarray(stat, dtype=float))

    except Exception:
        def rho(x, y):
            if len(x) < 2:
                return np.nan
            return float(pd.Series(x).corr(pd.Series(y), method='spearman'))

    summary_rows = []
    detail_rows = []
    for threshold in sorted(df['threshold'].unique()):
        thr_df = df[df['threshold'] == threshold]
        seed_rhos = []
        for seed, seed_df in thr_df.groupby('seed'):
            aligned = seed_df.sort_values('step')
            if aligned['WIS'].isna().all() or aligned['ESS'].isna().all():
                continue
            val = rho(aligned['WIS'].values, aligned['ESS'].values)
            detail_rows.append({'threshold': threshold, 'seed': seed, 'spearman_rho': val})
            if not np.isnan(val):
                seed_rhos.append(val)
        if seed_rhos:
            vals = np.array(seed_rhos)
            summary_rows.append(
                {
                    'threshold': threshold,
                    'spearman_mean': float(vals.mean()),
                    'spearman_std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    'spearman_median': float(np.median(vals)),
                    'spearman_iqr': float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    'n_seeds': len(seed_rhos)
                }
            )
        else:
            summary_rows.append(
                {'threshold': threshold, 'spearman_mean': np.nan, 'spearman_std': np.nan,
                 'spearman_median': np.nan, 'spearman_iqr': np.nan, 'n_seeds': 0}
            )
    summary_df = pd.DataFrame(summary_rows).sort_values('threshold')
    detail_df = pd.DataFrame(detail_rows).sort_values(['threshold', 'seed'])
    return summary_df, detail_df


def main():
    """Run stability diagnostics for each timestep."""
    setup_logging()
    for dt in DT_LIST:
        global OUTPUT_DIR, SUMMARY_PATH
        root_log_dir = ROOT_BASE / f'BCQ_asNormThreshold_dt{dt}h_grid'
        OUTPUT_DIR = OUTPUT_DIR_BASE / f'dt{dt}'
        SUMMARY_PATH = OUTPUT_DIR / 'step_summary.csv'

        ensure_output_dir()
        logging.info('Processing dt=%d: loading metrics from %s', dt, root_log_dir)
        try:
            df = load_metrics(root_log_dir)
        except Exception as exc:
            logging.error('Failed to load metrics for dt=%d: %s', dt, exc)
            continue
        logging.info('Loaded %d rows of metrics for dt=%d', len(df), dt)

        step_summary = compute_step_summary(df)
        step_summary.to_csv(SUMMARY_PATH, index=False)
        logging.info('Wrote step summary (with CI) to %s', SUMMARY_PATH)

        for threshold in sorted(df['threshold'].unique()):
            thr_summary = step_summary[step_summary['threshold'] == threshold]
            if thr_summary.empty:
                continue
            plot_mean_ci(thr_summary, 'WIS', threshold)
            plot_mean_ci(thr_summary, 'ESS', threshold)

        std_summary = summarize_cross_seed_std(df)
        std_summary.to_csv(OUTPUT_DIR / 'cross_seed_std_summary.csv', index=False)
        logging.info('Wrote cross-seed std summary to %s', OUTPUT_DIR / 'cross_seed_std_summary.csv')

        spearman_summary, spearman_detail = summarize_seed_spearman(df)
        spearman_summary.to_csv(OUTPUT_DIR / 'seed_spearman_summary.csv', index=False)
        spearman_detail.to_csv(OUTPUT_DIR / 'seed_spearman_detail.csv', index=False)
        logging.info('Wrote seed Spearman summaries to %s', OUTPUT_DIR)


if __name__ == '__main__':
    main()
