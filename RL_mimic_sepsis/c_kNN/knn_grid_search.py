"""
KNN grid search for behavior cloning with Slurm submission and
result tracking. Now with progress bars (tqdm) for dataset build and GridSearchCV.

Usage (local):
  python knn_grid_search.py --dataset_type val --timestep 1

Usage (submit to Slurm with your settings):
/local/scratch/ysun564/anaconda3/envs/rl4h_rep_new/bin/python /local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/c_kNN/knn_grid_search.py --dataset_type train --timestep 1 --submit_slurm
"""

import os
import json
import time
import math
import numpy as np
import torch
import joblib

from pathlib import Path
from contextlib import contextmanager

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, ParameterGrid
from sklearn.metrics import make_scorer, top_k_accuracy_score, roc_auc_score

from test_tube import HyperOptArgumentParser, SlurmCluster
from tqdm.auto import tqdm



def top3_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)
    labels = estimator.classes_
    return float(top_k_accuracy_score(y, proba, k=3, labels=labels))

def _safe_multiclass_auc(y_true, y_score, labels, average):
    """
    Robust multiclass ROC AUC:
    - force 'labels' to align with columns in y_score
    - if a fold has <2 present classes, return 0.5 (chance level)
    """
    try:
        return float(
            roc_auc_score(
                y_true, y_score,
                labels=labels,
                multi_class='ovr',
                average=average
            )
        )
    except ValueError:
        present = np.intersect1d(labels, np.unique(y_true))
        if len(present) < 2:
            return 0.5
        idx = [i for i, c in enumerate(labels) if c in present]
        return float(
            roc_auc_score(
                y_true, y_score[:, idx],
                labels=present,
                multi_class='ovr',
                average=average
            )
        )

def macro_auc_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)
    labels = estimator.classes_
    return _safe_multiclass_auc(y, proba, labels, average='macro')

def weighted_auc_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)
    labels = estimator.classes_
    return _safe_multiclass_auc(y, proba, labels, average='weighted')

def make_odd(k):
    """Return the nearest odd integer ≥ 1 for a given value k."""
    k = int(round(k))
    return k if k % 2 == 1 else k + 1


def knn_k_grid_dense(n, cv=5):
    n_train = (cv - 1) * n // cv
    kmax = min(n_train - 1, int(5 * math.sqrt(n)))

    tiny = {1, 3, 5, 9, 15, 21}
    around_sqrt = {make_odd(t * math.sqrt(n)) for t in [0.6, 0.8, 1.0, 1.2, 1.4]}
    fracs = {
        make_odd(f * n)
        for f in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
    }
    grid = sorted(k for k in (tiny | around_sqrt | fracs) if 1 <= k <= kmax)
    return grid


def _kmax_from_n_5sqrt(n, cv=5):
    """Compute an upper bound for k given sample size n and cv splits."""
    n_train = (cv - 1) * n // cv
    return max(1, min(n_train - 1, int(5 * math.sqrt(n))))

def _kmax_from_n_15sqrt(n, cv=5):
    """Compute an upper bound for k given sample size n and cv splits."""
    n_train = (cv - 1) * n // cv
    return max(1, min(n_train - 1, int(15 * math.sqrt(n))))

def knn_k_grid_log(n, cv=5, k_min=21, num=8, base=2.0, ensure_odd=True):
    """
    Build a logarithmically spaced K grid within [k_min, kmax].

    Parameters
    ----------
    n : int
        Total number of training samples used to derive kmax.
    cv : int
        Number of CV splits for estimating the available train size.
    k_min : int
        Minimum k to start the log grid (inclusive).
    num : int
        Number of points to generate.
    base : float
        Logarithmic base, e.g., 2.0 for powers of two, 10.0 for decades.
    ensure_odd : bool
        If True, force all k to be odd.

    Returns
    -------
    list
        Sorted, unique integer ks within [1, kmax], log-spaced.
    """
    kmax = _kmax_from_n_15sqrt(n, cv=cv)
    k_min = max(1, int(k_min))
    if k_min > kmax:
        return [make_odd(kmax) if ensure_odd else int(kmax)]

    # numpy.logspace with custom base
    start = math.log(k_min, base)
    stop = math.log(kmax, base)
    ks = np.logspace(start, stop, num=num, base=base)

    if ensure_odd:
        ks = [make_odd(x) for x in ks]
    else:
        ks = [max(1, int(round(x))) for x in ks]

    # Deduplicate and clamp
    ks = sorted(set(min(kmax, max(1, k)) for k in ks))
    return ks

def refine_k_log_around(k_center, n, cv=5, lower=0.75, upper=1.5, num=5, ensure_odd=True):
    """
    Build a local logarithmic refinement grid around a given k_center.

    Parameters
    ----------
    k_center : int
        The anchor k around which to refine.
    n : int
        Total number of training samples used to derive kmax.
    cv : int
        Number of CV splits for estimating the available train size.
    lower : float
        Lower multiplicative bound relative to k_center (e.g., 0.75).
    upper : float
        Upper multiplicative bound relative to k_center (e.g., 1.5).
    num : int
        Number of points to generate in the refinement.
    ensure_odd : bool
        If True, force all k to be odd.

    Returns
    -------
    list
        Sorted, unique integer ks within [1, kmax], log-spaced around k_center.
    """
    kmax = _kmax_from_n_5sqrt(n, cv=cv)
    a = max(1, k_center * lower)
    b = max(1, k_center * upper)
    if a > b:
        a, b = b, a

    start = math.log(a, 10.0)
    stop = math.log(b, 10.0)
    ks = np.logspace(start, stop, num=num, base=10.0)

    if ensure_odd:
        ks = [make_odd(x) for x in ks]
    else:
        ks = [max(1, int(round(x))) for x in ks]

    ks = sorted(set(min(kmax, max(1, k)) for k in ks))
    return ks

def build_paths(hparams):
    base_dir = (
        f'{hparams.data_root}/data_as{hparams.action_space}_dt{hparams.timestep}h/'
        f'episodes+encoded_state'
    )
    data_pt = f'{base_dir}/{hparams.dataset_type}_data.pt'

    run_stamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        hparams.out_dir,
        f'as{hparams.action_space}_dt{hparams.timestep}h',
        hparams.dataset_type,
        run_stamp,
    )
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return data_pt, run_dir

def load_dataset(data_pt):
    data = torch.load(data_pt)

    train_statevecs = []
    train_actions = []
    groups_list = []

    n_episodes = len(data['icustayids'])
    it = range(n_episodes)
    if tqdm is not None:
        it = tqdm(it, desc='Build dataset', leave=True)

    for i in it:
        lng = int(data['lengths'][i])
        states_i = data['statevecs'][i][: lng - 1].cpu().numpy()
        actions_i = data['actions'][i][1:lng].cpu().numpy().astype(np.int64)
        train_statevecs.append(states_i)
        train_actions.append(actions_i)
        stay_id = int(data['icustayids'][i])
        groups_list.append(np.full(lng - 1, stay_id, dtype=np.int64))

    X = np.vstack(train_statevecs).astype(np.float32, copy=False)
    y = np.concatenate(train_actions)
    groups = np.concatenate(groups_list)

    return X, y, groups


@contextmanager
def tqdm_joblib(tqdm_bar):
    """
    Patch joblib to report into tqdm progress bar given as argument.
    If tqdm is None, yields without patching.
    """
    if tqdm_bar is None or tqdm is None:
        yield
        return

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_bar.close()


def run_grid_search(hparams):
    """
    Run KNN grid search with grouped CV, save artifacts, and log metrics.
    """
    data_pt, run_dir = build_paths(hparams)
    X, y, groups = load_dataset(data_pt)

    print('X shape:', X.shape, 'y shape:', y.shape)

    k = len(y)
    k_min = _kmax_from_n_5sqrt(k)
    k_grid = knn_k_grid_log(len(y), cv=hparams.n_splits, k_min=k_min)
    print('k grid (len={}):'.format(len(k_grid)), k_grid[:20], '...')

    pipe = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_jobs=1, algorithm='brute')),
        ]
    )

    param_grid = {
        'knn__n_neighbors': k_grid,
        'knn__weights': ['distance'],
        'knn__metric': ['minkowski'],
        'knn__p': [2],
    }

    cv = StratifiedGroupKFold(
        n_splits=hparams.n_splits, shuffle=True, random_state=hparams.seed
    )

    scoring = {
        'macro_auc': macro_auc_scorer,
        'weighted_auc': weighted_auc_scorer,
        'top3': top3_scorer,
    }

    # Cap cv_jobs to available CPUs (local or Slurm).
    cpus_env = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    cv_jobs_eff = max(1, min(hparams.cv_jobs, cpus_env))
    if cv_jobs_eff != hparams.cv_jobs:
        print(f'[info] adjust cv_jobs: {hparams.cv_jobs} -> {cv_jobs_eff}')

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit='weighted_auc',
        cv=cv,
        n_jobs=cv_jobs_eff,
        verbose=3,  
        pre_dispatch=2*cv_jobs_eff,
        error_score=np.nan,
        return_train_score=False,
    )

    # total tasks ~= (#param combos) * (n_splits)
    total_tasks = len(ParameterGrid(param_grid)) * hparams.n_splits
    pbar = tqdm(total=total_tasks, desc='GridSearchCV', leave=True) if tqdm else None

    with tqdm_joblib(pbar):
        gs.fit(X, y, groups=groups)

    best_idx = gs.best_index_
    best_macro = float(gs.best_score_)
    best_weighted = float(gs.cv_results_['mean_test_weighted_auc'][best_idx])
    best_top3 = float(gs.cv_results_['mean_test_top3'][best_idx])

    print('Best params:', gs.best_params_)
    print('Best CV Macro AUROC: {:.4f}'.format(best_macro))
    print('Weighted AUROC @best: {:.4f}'.format(best_weighted))
    print('Top-3 accuracy @best: {:.4f}'.format(best_top3))

    cv_csv = os.path.join(run_dir, 'cv_results.csv')
    best_json = os.path.join(run_dir, 'best_summary.json')
    model_path = os.path.join(run_dir, 'best_pipeline.joblib')
    kgrid_txt = os.path.join(run_dir, 'k_grid.txt')

    import pandas as pd
    pd.DataFrame(gs.cv_results_).to_csv(cv_csv, index=False)

    with open(best_json, 'w') as f:
        json.dump(
            {
                'best_params': gs.best_params_,
                'best_macro_auc': best_macro,
                'best_weighted_auc': best_weighted,
                'best_top3_acc': best_top3,
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'n_classes': int(len(np.unique(y))),
                'seed': int(hparams.seed),
                'n_splits': int(hparams.n_splits),
                'dataset_pt': data_pt,
            },
            f,
            indent=2,
        )

    with open(kgrid_txt, 'w') as f:
        f.write('\n'.join(str(k) for k in k_grid))

    joblib.dump(gs.best_estimator_, model_path)

    print('Artifacts saved to:', run_dir)
    return {
        'run_dir': run_dir,
        'best_params': gs.best_params_,
        'best_macro_auc': best_macro,
        'best_weighted_auc': best_weighted,
        'best_top3_acc': best_top3,
    }


def optimize_on_cluster(hyperparams):
    """
    Submit this script to Slurm using cluster settings.
    """
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='/local/scratch/ysun564/project/OfflineRL_TimeStep/'
                 'RL_mimic_sepsis/slurm',
    )

    cluster.notify_job_status(
        email='ysun564@emory.edu', on_done=False, on_fail=False
    )
    cluster.per_experiment_nb_gpus = 0
    cluster.per_experiment_nb_nodes = 1
    cluster.job_time = '12:00:00'
    cluster.memory_mb_per_node = 8000
    cluster.add_command(
        'eval "$(/local/scratch/ysun564/anaconda3/bin/conda shell.bash hook)"'
    )
    cluster.add_command('conda activate rl4h_rep_new')
    cluster.add_slurm_cmd('partition', 'hopper', comment='')
    cluster.add_slurm_cmd('ntasks', '1', comment='')
    cluster.add_slurm_cmd('cpus-per-task', '36', comment='')
    cluster.add_slurm_cmd('gpus', '0', comment='')
    
    # Add command for accelerate.
    cluster.add_command('export OMP_NUM_THREADS=1')
    cluster.add_command('export MKL_NUM_THREADS=1')
    cluster.add_command('export OPENBLAS_NUM_THREADS=1')
    cluster.add_command('export NUMEXPR_NUM_THREADS=1')

    cluster.optimize_parallel_cluster_cpu(
        main,
        nb_trials=hyperparams.nb_trials,
        job_name=(
            f'mimic_sepsis_KNN_grid_as{hyperparams.action_space}_'
            f'dt{hyperparams.timestep}h_data_{hyperparams.dataset_type}_5to15sqrtk_{int(time.time())}'
        ),
    )

def running_on_slurm():
    return any(k in os.environ for k in ['SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_CLUSTER_NAME'])

def main(hparams, cluster=None):
    """
    Entry point for both local run and Slurm jobs.
    The second arg is provided by SlurmCluster; it's optional for local runs.
    """
    result = run_grid_search(hparams)
    print('Done. Best summary:', result)


def build_hparams():
    """
    Build args for both local run and Slurm submission.
    """
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument('--action_space', default='NormThreshold')
    parser.add_argument('--timestep', default=1, type=int)
    parser.add_argument('--dataset_type', default='val')
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cv_jobs', default=36, type=int)
    parser.add_argument('--cv_verbose', default=3, type=int)

    parser.add_argument(
        '--data_root',
        default='/local/scratch/ysun564/project/OfflineRL_TimeStep/'
                'RL_mimic_sepsis/data',
    )
    parser.add_argument(
        '--out_dir',
        default='/local/scratch/ysun564/project/OfflineRL_TimeStep/'
                'RL_mimic_sepsis/c_kNN/logs',
    )

    parser.add_argument('--submit_slurm', action='store_true')
    parser.add_argument('--nb_trials', default=1, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    hparams = build_hparams()
    if hparams.submit_slurm and not running_on_slurm():
        optimize_on_cluster(hparams)
    else:
        main(hparams)
