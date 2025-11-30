#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a neural-network behavior cloning policy, report metrics, and save artifacts.

This script mirrors the data handling pipeline from ``knn_behavior_cloning.py``
while replacing the kNN classifier with a small feed-forward neural network.
It loads the prepared episodic tensors, flattens them into step-level
(state, action) pairs, trains the network using cross-entropy loss, and then
reports multi-class AUROC scores on the train/validation/test splits.

Additionally, it computes Expected Calibration Error (ECE) and generates
reliability diagrams for each split. The learned model and metrics are saved
under the folder:
    F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/c_kNN/network

All paths are handled via pathlib.Path.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Data utilities (mirroring knn_behavior_cloning helpers)
# ---------------------------------------------------------------------------


def default_data_root():
    """Return the expected default dataset root on the local machine."""
    return Path('F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/data')


def build_paths(action_space, timestep, data_root):
    """Derive dataset file paths for the chosen action space and timestep."""
    base_dir = (
        data_root
        / f'data_as{action_space}_dt{timestep}h'
        / 'episodes+encoded_state+knn_pibs_k5sqrtn_uniform'
    )
    return {
        'train': base_dir / 'train_data.pt',
        'val': base_dir / 'val_data.pt',
        'test': base_dir / 'test_data.pt',
    }


def load_episode_file(path):
    """Load a torch serialized episodic dictionary."""
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {path}')
    return torch.load(path)


def flatten_state_action(data_dict):
    """Flatten episodic state/action sequences into step-level arrays.

    Align states s_t with next-step actions a_{t+1} using the stored episode
    lengths, matching the original behavior cloning pipeline.
    """
    states_collect = []
    labels_collect = []
    lengths = data_dict['lengths']
    state_vectors = data_dict['statevecs'].cpu().numpy()
    actions = data_dict['actions'].cpu().numpy()

    for index in range(len(lengths)):
        length = int(lengths[index])
        if length <= 1:
            continue
        states_collect.append(state_vectors[index][: length - 1])
        labels_collect.append(actions[index][1:length])

    if not states_collect:
        raise ValueError('No valid trajectories found when flattening data.')

    states = np.vstack(states_collect).astype(np.float32, copy=False)
    labels = np.concatenate(labels_collect).astype(np.int64, copy=False)
    return states, labels


# ---------------------------------------------------------------------------
# Model definition and training helpers
# ---------------------------------------------------------------------------


class FC_BC(nn.Module):
    """Batch-normalized fully connected behavior cloning network."""

    def __init__(self, state_dim, num_actions, num_nodes):
        super().__init__()
        self.l1 = nn.Linear(state_dim, num_nodes)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.l2 = nn.Linear(num_nodes, num_nodes)
        self.bn2 = nn.BatchNorm1d(num_nodes)
        self.l3 = nn.Linear(num_nodes, num_actions)

    def forward(self, state):
        """Forward pass returning class logits."""
        out = F.relu(self.l1(state))
        out = self.bn1(out)
        out = F.relu(self.l2(out))
        out = self.bn2(out)
        return self.l3(out)


def build_model(state_dim, action_dim, hidden_dim, architecture):
    """Instantiate the requested behavior cloning policy architecture."""
    arch = architecture.lower()
    if arch == 'fc_bc':
        return FC_BC(state_dim=state_dim, num_actions=action_dim, num_nodes=hidden_dim)
    if arch == 'mlp':
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    raise ValueError(f"Unknown architecture '{architecture}'. Choose from ['mlp', 'fc_bc'].")


class TrainConfig:
    """Simple container for training configuration without type annotations."""

    def __init__(self, epochs=20, batch_size=256, lr=1e-3, hidden_dim=256, device='cpu', architecture='mlp', action_space='NormThreshold', timestep='1'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.device = device
        self.architecture = architecture
        self.action_space = action_space
        self.timestep = timestep


def make_dataloader(states, labels, cfg, shuffle):
    """Wrap numpy arrays into a PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(labels),
    )
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for states_batch, labels_batch in dataloader:
        states_batch = states_batch.to(device=device, dtype=torch.float32)
        labels_batch = labels_batch.to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(states_batch)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()

        batch_count = labels_batch.size(0)
        total_loss += loss.item() * batch_count
        total_samples += batch_count

    return total_loss / max(total_samples, 1)


def evaluate_accuracy(model, dataloader, device):
    """Compute simple accuracy over a dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for states_batch, labels_batch in dataloader:
            states_batch = states_batch.to(device=device, dtype=torch.float32)
            labels_batch = labels_batch.to(device=device, dtype=torch.long)
            logits = model(states_batch)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)
    return correct / max(total, 1)


def collect_probabilities(model, dataloader, device):
    """Gather true labels and predicted probabilities for metrics computation."""
    model.eval()
    labels_true = []
    probabilities = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for states_batch, labels_batch in dataloader:
            states_batch = states_batch.to(device=device, dtype=torch.float32)
            logits = model(states_batch)
            probs = softmax(logits).cpu().numpy()
            probabilities.append(probs)
            labels_true.append(labels_batch.numpy())

    return np.concatenate(labels_true), np.vstack(probabilities)


def compute_auroc_scores(labels_true, probabilities):
    """Compute macro, weighted, and micro AUROC scores."""
    scores = {}
    for average in ('macro', 'weighted', 'micro'):
        try:
            scores[average] = roc_auc_score(
                labels_true,
                probabilities,
                multi_class='ovr',
                average=average,
            )
        except ValueError:
            scores[average] = float('nan')
    return scores


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

    # --- Bar plot for per-bin accuracy ---
    ax.bar(
        bin_centers,
        bin_accuracy,
        width=width,
        color='#4e79a7',
        edgecolor='black',
        alpha=0.5,
        label='Per-bin accuracy'
    )

    # --- Add line plot for smooth trend ---
    ax.plot(
        bin_centers,
        bin_accuracy,
        'o-',
        color='#2c5282',
        linewidth=2,
        label='Trend'
    )

    # --- Perfect calibration line ---
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

    # --- Axes setup ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.legend(loc='lower right')

    # --- Add sample counts ---
    for x, h, c in zip(bin_centers, bin_accuracy, bin_count):
        if c > 0:
            ax.text(x, min(h + 0.03, 0.97), str(int(c)),
                    ha='center', va='bottom', fontsize=8, color='black')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------


def main(cfg):
    """Run training, evaluation, calibration, plotting, and saving artifacts."""
    data_root = default_data_root()
    paths = build_paths(cfg.action_space, cfg.timestep, data_root)

    print('[INFO] Loading episodic datasets...')
    train_episode = load_episode_file(paths['train'])
    val_episode = load_episode_file(paths['val'])
    test_episode = load_episode_file(paths['test'])

    num_actions = int(train_episode['actionvecs'].shape[-1])
    state_dim = int(train_episode['statevecs'].shape[-1])
    print(f'[INFO] State dimension = {state_dim}, Action dimension = {num_actions}')

    print('[INFO] Flattening trajectories into step-level datasets...')
    states_train, labels_train = flatten_state_action(train_episode)
    states_val, labels_val = flatten_state_action(val_episode)
    states_test, labels_test = flatten_state_action(test_episode)
    print(
        f"[INFO] Train samples: {states_train.shape[0]}, Val samples: {states_val.shape[0]}, "
        f"Test samples: {states_test.shape[0]}"
    )

    device = torch.device(cfg.device)

    train_loader = make_dataloader(states_train, labels_train, cfg, shuffle=True)
    val_loader = make_dataloader(states_val, labels_val, cfg, shuffle=False)
    test_loader = make_dataloader(states_test, labels_test, cfg, shuffle=False)

    model = build_model(
        state_dim=state_dim,
        action_dim=num_actions,
        hidden_dim=cfg.hidden_dim,
        architecture=cfg.architecture,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_macro_auroc = -1.0
    best_state = None
    train_loss_history = []
    val_acc_history = []
    val_macro_auroc_history = []

    print('[INFO] Starting training...')
    start_time = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)

        labels_true, probabilities = collect_probabilities(model, val_loader, device)
        val_auroc = compute_auroc_scores(labels_true, probabilities)
        val_macro_auroc = float(val_auroc.get('macro', float('nan')))

        if val_macro_auroc > best_val_macro_auroc:
            best_val_macro_auroc = val_macro_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        train_loss_history.append(float(train_loss))
        val_acc_history.append(float(val_acc))
        val_macro_auroc_history.append(val_macro_auroc)

        print(
            f'Epoch {epoch:02d}/{cfg.epochs} | '
            f'Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro AUROC: {val_macro_auroc:.4f}' 
        )

    elapsed = time.perf_counter() - start_time
    print(f'[INFO] Training complete in {elapsed/60:.2f} minutes')

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'[INFO] Restored model weights with best validation macro AUROC = {best_val_macro_auroc:.4f}')

    print('[INFO] Evaluating on train/val/test splits...')
    split_loaders = [('train', train_loader), ('val', val_loader), ('test', test_loader)]
    split_metrics = {}

    for split_name, loader in split_loaders:
        labels_true, probabilities = collect_probabilities(model, loader, device)
        accuracy = evaluate_accuracy(model, loader, device)
        auroc_scores = compute_auroc_scores(labels_true, probabilities)
        ece_value, bin_stats = compute_ece(labels_true, probabilities, n_bins=15)

        print(f"\n[{split_name.title()}] Accuracy: {accuracy:.4f}")
        for average, score in auroc_scores.items():
            print(f'[{split_name.title()}] {average.title()} AUROC: {score:.4f}')
        print(f'[{split_name.title()}] ECE: {ece_value:.4f}')

        split_metrics[split_name] = {
            'accuracy': float(accuracy),
            'auroc': {k: float(v) for k, v in auroc_scores.items()},
            'ece': float(ece_value),
        }

        output_dir = Path(f'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/c_kNN/calibration/bc_network_dt{cfg.timestep}h')
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f'reliability_{split_name}.png'
        title = f'Reliability Diagram ({split_name.title()})'
        plot_reliability_diagram(bin_stats, title, fig_path)

    # Save model and artifacts.
    output_dir = Path(f'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/c_kNN/calibration/bc_network_dt{cfg.timestep}h')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'model_best.pt'
    torch.save({'state_dict': model.state_dict()}, model_path)

    metadata = {
        'state_dim': int(state_dim),
        'action_dim': int(num_actions),
        'architecture': cfg.architecture,
        'hidden_dim': cfg.hidden_dim,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'learning_rate': cfg.lr,
        'device': cfg.device,
        'best_val_macro_auroc': float(best_val_macro_auroc),
        'training_loss': train_loss_history,
        'validation_accuracy': val_acc_history,
        'validation_macro_auroc': val_macro_auroc_history,
        'metrics': split_metrics,
    }
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved metrics and plots under: {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def default_config():
    """Return a default configuration object without using argument parsers."""
    return TrainConfig(
        epochs=20,
        batch_size=512,
        lr=1e-3,
        hidden_dim=256,
        device='cuda',
        architecture='mlp',
        action_space='NormThreshold',
        timestep='8',
    )


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    main(default_config())
