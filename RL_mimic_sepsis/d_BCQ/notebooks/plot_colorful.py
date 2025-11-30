#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCQ validation scatter & Pareto plotting script.

Reads metrics.csv from:
  .../d_BCQ/logs/BCQ_as{action_space}_dt{timestep}h_grid/
      dt{timestep}_threshold{threshold}seed{seed}/metrics.csv

Plots:
  1) All points: ESS vs WIS colored by tau.
  2) Pareto frontier per tau (maximize both WIS and ESS).

No hparams reading; tau values come from threshold_list.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================
# Config
# ================================
base_dir = Path('F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/d_BCQ/logs')
action_space = 'NormThreshold'
timestep = 8
metrics_name = 'metrics_100multiple_eps0.1.csv'
ESS_cutoff = 200  

seed_list = [0, 1, 2, 3, 4]
threshold_list = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 0.9999]

# Limit rows per run for plotting speed; set to None to use all rows.
max_rows = None

# Figure saving
save_fig = True
fig_dir = base_dir.parent / 'figs' / 'colorful'
xlim = (-10, 400)
ylim = (50, 101)

# Color palette for taus
palette = {
    0.0: 'tab:red',
    0.01: 'tab:orange',
    0.05: 'tab:olive',
    0.1: 'tab:green',
    0.3: 'tab:cyan',
    0.5: 'tab:blue',
    0.75: 'tab:purple',
    0.9999: 'tab:pink',
}


# ================================
# Utils
# ================================
def run_dir(root, action_space, timestep, tau, seed):
    """
    Build the directory path for a single (tau, seed) combination.
    """
    grid_name = f'BCQ_as{action_space}_dt{timestep}h_grid'
    folder = f'dt{timestep}_threshold{tau}seed{seed}'
    return root / grid_name / folder


def color_for_tau(tau):
    """
    Map a tau to a color with tolerance for float noise.
    """
    keys = np.array(list(palette.keys()), dtype=float)
    idx = int(np.argmin(np.abs(keys - float(tau))))
    return palette[keys[idx]]


def collect_metrics():
    """
    Collect metrics from all (tau, seed) combinations.
    Returns a DataFrame with columns: ['tau', 'seed', ...metrics columns...].
    """
    rows = []
    total = 0
    found = 0

    for tau in threshold_list:
        for seed in seed_list:
            d = run_dir(base_dir, action_space, timestep, tau, seed)
            csv_path = d / metrics_name
            total += 1
            if not csv_path.exists():
                print(f'[WARN] Missing: {csv_path}')
                continue

            df = pd.read_csv(csv_path)
            if max_rows is not None:
                df = df.iloc[:max_rows].copy()
            df.insert(0, 'seed', seed)
            df.insert(0, 'tau', float(tau))
            rows.append(df)
            found += 1

    if not rows:
        raise RuntimeError('No metrics found. Check paths and config.')

    print(f'[INFO] Found {found}/{total} (tau, seed) combos.')
    data = pd.concat(rows, ignore_index=True)
    return data


def pareto_indices(df, x_col='val_wis', y_col='val_ess'):
    """
    Compute Pareto frontier indices (maximize both x_col and y_col).
    Returns original DataFrame index list lying on the frontier.

    Algorithm: sort by x desc; sweep keeping strictly increasing best y.
    """
    pts = df[[x_col, y_col]].values
    order = np.argsort(-pts[:, 0], kind='mergesort')
    pts_sorted = pts[order]
    idx_sorted = df.index.values[order]

    frontier = []
    best_y = -np.inf
    for i, p in zip(idx_sorted, pts_sorted):
        if p[1] > best_y:
            frontier.append(i)
            best_y = p[1]
    return frontier


def ensure_fig_dir():
    """
    Create figure directory if saving is enabled.
    """
    if save_fig:
        fig_dir.mkdir(parents=True, exist_ok=True)


# ================================
# Plotting
# ================================
def plot_all_points(data):
    """
    Scatter plot of all points: ESS vs WIS colored by tau.
    """
    plt.figure()
    for tau, sub in data.groupby('tau'):
        plt.scatter(
            sub['val_ess'], sub['val_wis'],
            s=50, marker='.', c=color_for_tau(tau), alpha=0.35, linewidths=0
        )
    for k in palette:
        plt.plot([], [], '.', c=palette[k], label=f'tau={k}')
    plt.xlabel('ESS')
    plt.ylabel('WIS')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.axvline(ESS_cutoff, ls=':', lw=1, c='gray')
    plt.legend(loc='lower right', title='Clipping τ', ncol=2)
    plt.title(f'Validation Performance (All Hyperparams) — Δt={timestep}h — {action_space}')
    if save_fig:
        out = fig_dir / f'colorful_eval_{action_space}_dt{timestep}h_all.pdf'
        plt.savefig(out, bbox_inches='tight')
        print(f'[INFO] Saved: {out}')
    plt.show()
    plt.close()


def plot_pareto_per_tau(data):
    """
    Scatter plot of Pareto frontier points per tau.
    """
    plt.figure()
    for tau, sub in data.groupby('tau'):
        idx = pareto_indices(sub, x_col='val_wis', y_col='val_ess')
        front = sub.loc[idx]
        plt.scatter(
            front['val_ess'], front['val_wis'],
            s=50, marker='.', c=color_for_tau(tau), alpha=0.6, linewidths=0
        )
    for k in palette:
        plt.plot([], [], '.', c=palette[k], label=f'tau={k}')
    plt.xlabel('ESS')
    plt.ylabel('WIS')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.axvline(ESS_cutoff, ls=':', lw=1, c='gray')
    plt.legend(loc='lower right', title='Clipping τ', ncol=2)
    plt.title(f'Validation Performance (Pareto Frontier) — Δt={timestep}h — {action_space}')
    if save_fig:
        out = fig_dir / f'pareto_eval_{action_space}_dt{timestep}h_per_tau.pdf'
        plt.savefig(out, bbox_inches='tight')
        print(f'[INFO] Saved: {out}')
    plt.show()
    plt.close()


# ================================
# Main
# ================================
def main():
    """
    Entry point.
    """
    ensure_fig_dir()
    data = collect_metrics()
    print(f'[INFO] Loaded rows: {len(data)}')

    # Basic sanity check
    needed = {'val_ess', 'val_wis'}
    if not needed.issubset(set(data.columns)):
        raise KeyError(f'Missing columns in metrics: {needed - set(data.columns)}')

    plot_all_points(data)
    plot_pareto_per_tau(data)


if __name__ == '__main__':
    main()
