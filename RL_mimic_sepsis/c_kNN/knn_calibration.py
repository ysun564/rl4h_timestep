#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute ECE and reliability diagrams for kNN behavior cloning probabilities.

This script loads episodic datasets that contain per-step action probabilities
computed by a kNN behavior policy. It flattens the episodes into step-level
labels and probabilities, computes Expected Calibration Error (ECE), and saves
reliability diagrams and metrics for the train/val/test splits.

All paths are handled via pathlib.Path. No argument parsers are used.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('MPLBACKEND', 'Agg')

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def default_data_root():
    """Return the expected default dataset root on the local machine."""
    return Path('F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/data')


def dataset_paths(timestep):
    """Return train/val/test PT file paths for NormThreshold with chosen timestep."""
    base_dir = (
        default_data_root()
        / f'data_asNormThreshold_dt{timestep}h'
        / 'episodes+encoded_state+knn_pibs_k5sqrtn_uniform'
    )
    return {
        'train': base_dir / 'train_data.pt',
        'val': base_dir / 'val_data.pt',
        'test': base_dir / 'test_data.pt',
    }


def output_dir(timestep):
    """Return the directory to save calibration outputs for kNN."""
    return Path(f'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/c_kNN/calibration/knn/dt{timestep}h')


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def load_episode_file(path):
    """Load a torch serialized episodic dictionary."""
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {path}')
    return torch.load(path)


def flatten_labels_and_probs(data_dict, probs_key):
    """Flatten episodic labels and probabilities into step-level arrays.

    Align actions a_{t+1} as labels with probabilities at step t.
    Use the provided probability tensor key ('pibs' or 'estm_pibs').
    """
    labels_collect = []
    probs_collect = []

    lengths = data_dict['lengths']
    actions = data_dict['actions']

    if probs_key not in data_dict:
        raise KeyError(f"'{probs_key}' not found in dataset.")
    probs_tensor = data_dict[probs_key]

    probs_tensor = probs_tensor.cpu().numpy()
    actions = actions.cpu().numpy()

    for index in range(len(lengths)):
        length = int(lengths[index])
        if length <= 1:
            continue
        # Labels correspond to actions 1..L-1.
        labels_collect.append(actions[index][1:length])
        # Probabilities correspond to steps 0..L-2.
        probs_collect.append(probs_tensor[index][: length - 1, :])

    if not labels_collect:
        raise ValueError('No valid trajectories found when flattening data.')

    labels = np.concatenate(labels_collect).astype(np.int64, copy=False).ravel()
    probs = np.vstack(probs_collect).astype(np.float32, copy=False)
    return labels, probs


# ---------------------------------------------------------------------------
# Calibration metrics and plotting
# ---------------------------------------------------------------------------


def compute_ece(labels_true, probabilities, n_bins=15):
    """Compute Expected Calibration Error (ECE) using max-probability confidence.

    The ECE is computed by binning samples according to their predicted
    confidence (maximum class probability) and comparing per-bin average
    confidence with per-bin accuracy.
    """
    predicted_confidence = probabilities.max(axis=1)
    predicted_label = probabilities.argmax(axis=1)
    correctness = (predicted_label == labels_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(predicted_confidence, bin_edges[1:-1], right=True)

    bin_accuracy = np.zeros(n_bins, dtype=np.float32)
    bin_confidence = np.zeros(n_bins, dtype=np.float32)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        in_bin = bin_indices == b
        count = in_bin.sum()
        if count > 0:
            bin_accuracy[b] = correctness[in_bin].mean()
            bin_confidence[b] = predicted_confidence[in_bin].mean()
            bin_count[b] = count

    total = max(len(labels_true), 1)
    ece = np.sum((bin_count / total) * np.abs(bin_accuracy - bin_confidence))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return float(ece), {
        'bin_edges': bin_edges.tolist(),
        'bin_centers': bin_centers.tolist(),
        'bin_accuracy': bin_accuracy.tolist(),
        'bin_confidence': bin_confidence.tolist(),
        'bin_count': bin_count.tolist(),
    }


def plot_reliability_diagram(bin_stats, title, save_path):
    """Create and save a reliability diagram with both bars and line trend."""
    bin_centers = np.array(bin_stats['bin_centers'])
    bin_accuracy = np.array(bin_stats['bin_accuracy'])
    bin_count = np.array(bin_stats['bin_count'])

    fig, ax = plt.subplots(figsize=(6, 6))
    width = 1.0 / (len(bin_centers) + 2)

    # Bar plot for per-bin accuracy.
    ax.bar(
        bin_centers,
        bin_accuracy,
        width=width,
        color='#4e79a7',
        edgecolor='black',
        alpha=0.5,
        label='Per-bin accuracy'
    )

    # Line plot for smooth trend.
    ax.plot(
        bin_centers,
        bin_accuracy,
        'o-',
        color='#2c5282',
        linewidth=2,
        label='Trend'
    )

    # Perfect calibration line.
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

    # Axes setup.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.legend(loc='lower right')

    # Add sample counts.
    for x, h, c in zip(bin_centers, bin_accuracy, bin_count):
        if c > 0:
            ax.text(x, min(h + 0.03, 0.97), str(int(c)),
                    ha='center', va='bottom', fontsize=8, color='black')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def default_config():
    """Return default configuration for timestep and sources."""
    cfg = {
        'timestep': '8',
        'sources': ['pibs', 'estm_pibs'],
    }
    return cfg


def main():
    """Compute ECE and save reliability diagrams for both pibs and estm_pibs."""
    cfg = default_config()
    paths = dataset_paths(cfg['timestep'])
    out_dir_base = output_dir(cfg['timestep'])
    out_dir_base.mkdir(parents=True, exist_ok=True)

    for source in cfg['sources']:
        split_metrics = {}
        out_dir = out_dir_base / source
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing source: {source} | Output: {out_dir}")

        for split_name in ['train', 'val', 'test']:
            episode = load_episode_file(paths[split_name])
            try:
                labels_true, probabilities = flatten_labels_and_probs(episode, source)
            except KeyError as e:
                print(f"[WARN] {e} Skipping {source} for split '{split_name}'.")
                continue

            ece_value, bin_stats = compute_ece(labels_true, probabilities, n_bins=15)

            # Save plot.
            fig_path = out_dir / f'reliability_{split_name}_{source}.png'
            title = f"Reliability Diagram (kNN {source}, {split_name.title()}, dt={cfg['timestep']}h)"
            plot_reliability_diagram(bin_stats, title, fig_path)

            # Record metrics.
            split_metrics[split_name] = {
                'ece': float(ece_value),
                'num_samples': int(len(labels_true)),
            }
            print(f"[INFO] {split_name.title()} [{source}] ECE: {ece_value:.4f} | Samples: {len(labels_true)}")

        # Save metrics JSON per source.
        with (out_dir / f'metrics_{source}.json').open('w', encoding='utf-8') as f:
            json.dump(split_metrics, f, indent=2)
        print(f"[INFO] Saved metrics and plots under: {out_dir}")


if __name__ == '__main__':
    main()
