#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/local/scratch/ysun564/anaconda3/envs/rl4h_rep_new/bin/python /local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/c_kNN/knn_behavior_cloning.py --submit_slurm --action_space NormThreshold --timestep 1 --K_train 3873 --K_val 1795 --cpus_per_task 18 --memory_mb 80000
python knn_behavior_cloning.py --action_space NormThreshold --timestep 1 --K_train 3873 --K_val 1795

"""

import os
import time
import math
import numpy as np
import torch

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from test_tube import HyperOptArgumentParser, SlurmCluster





def default_data_root():
    """
    Return a sensible default data root on HPC.
    """
    return '/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/data'


def build_paths(hparams):
    """
    Build all input/output paths from hparams.
    """
    data_root = hparams.data_root or default_data_root()
    base_dir = os.path.join(
        data_root,
        f"data_as{hparams.action_space}_dt{hparams.timestep}h",
        'episodes+encoded_state',
    )
    save_dir = os.path.join(
        data_root,
        f"data_as{hparams.action_space}_dt{hparams.timestep}h",
        'episodes+encoded_state+knn_pibs_',
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return {
        'base_dir': base_dir,
        'save_dir': save_dir,
        'train_pt': os.path.join(base_dir, 'train_data.pt'),
        'val_pt': os.path.join(base_dir, 'val_data.pt'),
        'test_pt': os.path.join(base_dir, 'test_data.pt'),
        'npz_out': os.path.join(
            save_dir,
            f"knn_output_ktrain{hparams.K_train}_kval{hparams.K_val}.npz",
        ),
        'train_out': os.path.join(save_dir, f"train_data.pt"),
        'val_out': os.path.join(save_dir, f"val_data.pt"),
        'test_out': os.path.join(save_dir, f"test_data.pt"),
    }


def load_episode_dicts(paths):
    """
    Load episodic tensors for train/val/test from torch .pt files.
    """
    train = torch.load(paths['train_pt'])
    val = torch.load(paths['val_pt'])
    test = torch.load(paths['test_pt'])
    return train, val, test


def flatten_sa(data_dict):
    """
    Flatten episodic (S,A) into step-level arrays (X, y).
    Uses lengths to align S(t)->A(t+1).
    """
    X_list, y_list = [], []
    n_episodes = len(data_dict['icustayids'])
    for i in range(n_episodes):
        lng = int(data_dict['lengths'][i])
        if lng <= 1:
            continue
        X_list.append(data_dict['statevecs'][i][:lng - 1].cpu().numpy())
        y_list.append(data_dict['actions'][i][1:lng].cpu().numpy())
    X = np.vstack(X_list).astype(np.float32, copy=False)
    y = np.concatenate(y_list).astype(np.int64, copy=False)
    return X, y


def fit_knn(X, y, k, n_jobs):
    """
    Fit a KNN classifier with distance weighting and L2 (p=2).
    """
    clf = KNeighborsClassifier(
        n_neighbors=k,
        weights='uniform',
        metric='minkowski',
        p=2,
        n_jobs=n_jobs,
    )
    clf.fit(X, y)
    return clf


def align_proba(proba, classes, num_actions):
    """
    Align scikit-learn proba (only for seen classes) to full action space.
    """
    out = np.zeros((proba.shape[0], num_actions), dtype=np.float32)
    idx = np.asarray(classes, dtype=np.int64)
    out[:, idx] = proba.astype(np.float32, copy=False)
    return out


def predict_proba_in_chunks(clf, X, num_actions, batch_size):
    """
    Predict probabilities in chunks with class alignment to avoid OOM.
    """
    n = X.shape[0]
    bs = max(1, int(batch_size))
    out = np.zeros((n, num_actions), dtype=np.float32)
    classes = clf.classes_
    for s in range(0, n, bs):
        e = min(n, s + bs)
        proba = clf.predict_proba(X[s:e])
        out[s:e] = align_proba(proba, classes, num_actions)
    return out


def attach_probs_back(episodic, pibs_flat, estm_flat):
    """
    Attach step-level pibs and estm_pibs back to episodic tensors.
    """
    num_actions = episodic['actionvecs'].shape[-1]
    episodic['pibs'] = torch.zeros_like(episodic['actionvecs'])
    episodic['estm_pibs'] = torch.zeros_like(episodic['actionvecs'])

    ptr = 0
    n_episodes = len(episodic['icustayids'])
    for i in range(n_episodes):
        lng = int(episodic['lengths'][i])
        steps = max(0, lng - 1)
        if steps == 0:
            continue
        episodic['pibs'][i, :steps, :num_actions] = torch.from_numpy(
            pibs_flat[ptr:ptr + steps]
        )
        episodic['estm_pibs'][i, :steps, :num_actions] = torch.from_numpy(
            estm_flat[ptr:ptr + steps]
        )
        ptr += steps


def sanity_check(episodic):
    """
    Basic consistency checks on shapes and padding.
    """
    n_actions = episodic['actionvecs'].shape[-1]
    n_eps = len(episodic['icustayids'])
    assert len(episodic['lengths']) == n_eps
    assert episodic['pibs'].shape[-1] == n_actions
    assert episodic['estm_pibs'].shape[-1] == n_actions
    # Spot-check first few episodes
    for i in range(min(5, n_eps)):
        lng = int(episodic['lengths'][i])
        if lng <= 1:
            continue
        assert episodic['pibs'][i, lng - 1:].abs().sum().item() == 0
        assert episodic['estm_pibs'][i, lng - 1:].abs().sum().item() == 0


def train_knn(hparams):
    """
    End-to-end pipeline: load data, build flat datasets, train KNNs,
    compute pibs/estm_pibs with chunked prediction, and save outputs.
    """
    
    paths = build_paths(hparams)

    print('[INFO] Loading episodic datasets...')
    train_ep, val_ep, test_ep = load_episode_dicts(paths)
    num_actions = int(train_ep['actionvecs'].shape[-1])
    print(f"[INFO] Action space size: {num_actions}")

    print('[INFO] Flattening (S,A) datasets...')
    X_tr, y_tr = flatten_sa(train_ep)
    X_va, y_va = flatten_sa(val_ep)
    X_te, y_te = flatten_sa(test_ep)
    print(f"[INFO] Train: X={X_tr.shape}, y={y_tr.shape}")
    print(f"[INFO] Val:   X={X_va.shape}, y={y_va.shape}")
    print(f"[INFO] Test:  X={X_te.shape}, y={y_te.shape}")

    n_jobs = hparams.n_jobs if hparams.n_jobs is not None else -1
    bs = int(hparams.batch_size)

    t0 = time.perf_counter()
    print(f"[INFO] Fitting KNN (train, k={hparams.K_train})...")
    clf_train = fit_knn(X_tr, y_tr, int(hparams.K_train), n_jobs)

    print('[INFO] Predicting train pibs/estm_pibs...')
    train_pibs = predict_proba_in_chunks(clf_train, X_tr, num_actions, bs)
    train_estm = train_pibs  # same classifier by definition

    print(f"[INFO] Fitting KNN (val, k={hparams.K_val})...")
    clf_val = fit_knn(X_va, y_va, int(hparams.K_val), n_jobs)

    print('[INFO] Predicting val pibs...')
    val_pibs = predict_proba_in_chunks(clf_val, X_va, num_actions, bs)
    print('[INFO] Predicting val estm_pibs (by train clf)...')
    val_estm = predict_proba_in_chunks(clf_train, X_va, num_actions, bs)

    print(f"[INFO] Fitting KNN (test, k={hparams.K_val})...")
    clf_test = fit_knn(X_te, y_te, int(hparams.K_val), n_jobs)

    print('[INFO] Predicting test pibs...')
    test_pibs = predict_proba_in_chunks(clf_test, X_te, num_actions, bs)
    print('[INFO] Predicting test estm_pibs (by train clf)...')
    test_estm = predict_proba_in_chunks(clf_train, X_te, num_actions, bs)

    dt = time.perf_counter() - t0
    print(f"[INFO] KNN pipeline done in {dt/60:.2f} min")

    print(f"[INFO] Saving raw npz to: {paths['npz_out']}")
    np.savez(
        paths['npz_out'],
        train_pibs=train_pibs,
        train_estm_pibs=train_estm,
        val_pibs=val_pibs,
        val_estm_pibs=val_estm,
        test_pibs=test_pibs,
        test_estm_pibs=test_estm,
    )

    print('[INFO] Attaching probabilities back to episodes...')
    attach_probs_back(train_ep, train_pibs, train_estm)
    attach_probs_back(val_ep, val_pibs, val_estm)
    attach_probs_back(test_ep, test_pibs, test_estm)

    print('[INFO] Sanity checking...')
    sanity_check(train_ep)
    sanity_check(val_ep)
    sanity_check(test_ep)

    print('[INFO] Saving updated episodic .pt files...')
    torch.save(train_ep, paths['train_out'])
    torch.save(val_ep, paths['val_out'])
    torch.save(test_ep, paths['test_out'])

    print('[INFO] All done.')
    return {
        'minutes': dt / 60.0,
        'train_steps': X_tr.shape[0],
        'val_steps': X_va.shape[0],
        'test_steps': X_te.shape[0],
        'num_actions': num_actions,
    }


def main(hparams, cluster=None):
    """
    Entry point for both local run and Slurm jobs.
    """
    result = train_knn(hparams)
    print('[RESULT]', result)


def optimize_on_cluster(hparams):
    """
    Submit this script to Slurm using cluster settings.
    """
    cluster = SlurmCluster(
        hyperparam_optimizer=hparams,
        log_path='/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/slurm',
    )

    cluster.notify_job_status(
        email='ysun564@emory.edu',
        on_done=False,
        on_fail=False,
    )
    cluster.per_experiment_nb_gpus = 0
    cluster.per_experiment_nb_nodes = 1
    cluster.job_time = hparams.job_time
    cluster.memory_mb_per_node = hparams.memory_mb

    cluster.add_command('eval "$(/local/scratch/ysun564/anaconda3/bin/conda shell.bash hook)"')
    cluster.add_command('conda activate rl4h_rep_new')
    cluster.add_slurm_cmd('partition', hparams.partition, comment='')
    cluster.add_slurm_cmd('ntasks', '1', comment='')
    cluster.add_slurm_cmd('cpus-per-task', str(hparams.cpus_per_task), comment='')
    cluster.add_slurm_cmd('gpus', '0', comment='')

    cluster.add_command('export OMP_NUM_THREADS=1')
    cluster.add_command('export MKL_NUM_THREADS=1')
    cluster.add_command('export OPENBLAS_NUM_THREADS=1')
    cluster.add_command('export NUMEXPR_NUM_THREADS=1')

    job_name = (
        f"mimic_sepsis_KNN_as{hparams.action_space}_"
        f"dt{hparams.timestep}h_ktr{hparams.K_train}_kva{hparams.K_val}_"
        f"{int(time.time())}"
    )

    cluster.optimize_parallel_cluster_cpu(
        main,
        nb_trials=hparams.nb_trials,
        job_name=job_name,
    )


def build_hparams():
    """
    Define CLI arguments (including Slurm switches) and parse them.
    """
    parser = HyperOptArgumentParser(strategy='random_search')

    # Data & task
    parser.add_argument('--action_space', default='NormThreshold')
    parser.add_argument('--timestep', default='1')
    parser.add_argument('--data_root', default=None)

    # KNN params
    parser.add_argument('--K_train', default=3873)
    parser.add_argument('--K_val', default=1795)
    parser.add_argument('--n_jobs', default=None)
    parser.add_argument('--batch_size', default=10000)

    # Slurm/cluster params
    parser.add_argument('--submit_slurm', action='store_true', default=False)
    parser.add_argument('--partition', default='hopper')
    parser.add_argument('--cpus_per_task', default=36)
    parser.add_argument('--memory_mb', default=80000)
    parser.add_argument('--job_time', default='12:00:00')
    parser.add_argument('--nb_trials', default=1)

    return parser.parse_args()


if __name__ == '__main__':
    hparams = build_hparams()
    if hparams.n_jobs is None:
        hparams.n_jobs = int(hparams.cpus_per_task)
    if hparams.submit_slurm:
        optimize_on_cluster(hparams)
    else:
        main(hparams)
