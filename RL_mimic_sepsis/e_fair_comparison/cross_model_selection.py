'''Cross-timestep model selection runner (cross-Δt).

This script mirrors the model selection runner but evaluates models trained
at timestep t1 on a dataset at timestep t2 using the cross-Δt evaluation
code. It chooses filesystem paths depending on the host OS (Windows vs
non-Windows) so we can run the same script on Windows laptop an Ubuntu server.

TO RUN:
1. Change TASK_NAME
2. Change base_dir and data_t2_path
3. Change the hyperparameters
4. Run
    
'''

import os
import sys
import platform
import time
import random
import pandas as pd
import numpy as np
import torch

from pathlib import Path
from types import SimpleNamespace
from joblib import Parallel, delayed
import subprocess
import tempfile
import platform

# ensure project root is on sys.path when running as a module
# choose project root based on host OS so the same script works on Windows and Linux
project_root_windows = r'F:\time_step\OfflineRL_FactoredActions'
project_root_unix = '/local/scratch/ysun564/project/OfflineRL_TimeStep'
_system = platform.system()
if _system == 'Windows':
    _proj_root = project_root_windows
else:
    _proj_root = project_root_unix

# insert at front if not already present
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from RL_mimic_sepsis.d_BCQ.src.model import BCQ as BCQ_new
from RL_mimic_sepsis.d_BCQ.src.data import remap_rewards
from RL_mimic_sepsis.e_fair_comparison.cross_evaluation import (
    evaluate_cross_dt, load_dataset_t2
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

TASK_NAME = 'bcnet_phwis_task'

def _paths_for_system(t1: int, t2: int):
    """Return (base_dir, data_t2_path) depending on platform.

    base_dir: directory that contains dt* subfolders for models trained at t1.
    data_t2_path: path to the t2 validation dataset to evaluate on.
    """
    system = platform.system()
    if system == 'Windows':
        base_dir = Path(rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\d_BCQ\logs\BCQ_asNormThreshold_dt{t1}h_grid_latent128d')
        data_t2_path = Path(rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\data\data_asNormThreshold_dt{t2}h\episodes+encoded_state+knn_pibs_final\val_data.pt')
    else:
        # Ubuntu Server.
        base_dir = Path(rf'/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/d_BCQ/logs/BCQ_bcnet/BCQ_asNormThreshold_dt{t1}h_grid_bcnet')
        data_t2_path = Path(rf'/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/data/data_asNormThreshold_dt{t2}h/episodes+encoded_state+knn_pibs_k5sqrtn_uniform/val_data.pt')
    return base_dir, data_t2_path


_DATASET_CACHE = {}


def _get_eval_dataset(data_t2_path: Path, t2: int):
    """Load (and cache) the episodic buffer for timestep ``t2``."""
    key = (str(data_t2_path), int(t2))
    if key not in _DATASET_CACHE:
        reward_cfg = SimpleNamespace(**{'R_immed': 0.0, 'R_death': 0.0, 'R_disch': 100.0})
        dataset_t2 = load_dataset_t2(str(data_t2_path), t2)
        dataset_t2.reward = remap_rewards(dataset_t2.reward, reward_cfg)
        _DATASET_CACHE[key] = dataset_t2
    return _DATASET_CACHE[key]


def evaluate_checkpoint_cross(
    ckpt_path,
    data_t2_path,
    t1,
    t2,
    clipping_value: float,
    pib_type: str = 'policy',
    running_device: str = 'cuda:0',
    eps: float = 0.1,
    eval_method: str = 'wis',
):
    """Evaluate one checkpoint (trained at t1) on a t2 dataset using cross-Δt OPE.

    Returns a dict with iteration, step, val_wis, and val_ess.
    """
    print(f'Path of loaded checkpoint: {ckpt_path}')

    t_start = time.time()

    fname = os.path.basename(str(ckpt_path))
    name, _ = os.path.splitext(fname)
    try:
        iteration = int(name.split("=", 1)[1].split("-", 1)[0])
    except Exception:
        nums = ''.join(ch for ch in name if ch.isdigit())
        iteration = int(nums) if nums else -1

    if running_device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- DYNAMIC MODEL LOADING ---
    # 1. Inspect the checkpoint to determine its structure.
    ckpt_content = torch.load(ckpt_path, map_location='cpu')
    state_dict_keys = ckpt_content.get('state_dict', {}).keys()
    
    # 2. Decide which model class to use.
    # If keys for the internal behavior policy (πb) exist, it's the old model.
    if any('Q.πb.0.weight' in k for k in state_dict_keys):
        print("Detected 'model_old.py' structure (with internal πb).")
        from RL_mimic_sepsis.d_BCQ.src.model_old import BCQ as BCQ_to_load
    else:
        print("Detected 'model.py' structure (without internal πb).")
        BCQ_to_load = BCQ_new
    
    model = BCQ_to_load.load_from_checkpoint(checkpoint_path=str(ckpt_path), map_location=device)
    model = model.to(device)
    model.eval()

    dataset_t2 = _get_eval_dataset(Path(data_t2_path), t2)

    wis_est, wis_ess = evaluate_cross_dt(
        model,
        dataset_t2,
        t1,
        t2,
        eps=eps,
        clipping=clipping_value,
        pib=pib_type,
        eval_method=eval_method,
        data_path_t2=str(data_t2_path)
    )

    t_end = time.time()
    elapsed = t_end - t_start

    val_key = 'val_phwis' if eval_method == 'phwis' else 'val_wis'
    metrics = {
        'iteration': iteration,
        'step': iteration - 1,
        val_key: wis_est,
        'val_ess': wis_ess,
        'eval_time_sec': elapsed,
    }

    print(metrics)
    return metrics



def _parse_iteration_from_path(p: Path):
    '''Extract iteration integer from a checkpoint Path. Returns -1 if none found.'''
    name = p.stem
    # try step=####-v1 style
    if '=' in name:
        try:
            return int(name.split('=', 1)[1].split('-', 1)[0])
        except Exception:
            pass
    nums = ''.join(ch for ch in name if ch.isdigit())
    return int(nums) if nums else -1


def rebuild_metrics_cross(
    data_t2_path,
    base_dir: Path,
    t1: int,
    t2: int,
    clipping_value: float,
    parallel: bool = True,
    pib_type: str = 'policy',
    running_device='cuda:0',
    eps: float = 0.1,
    n_jobs: int = 8,
    eval_method: str = 'wis',
    ckpt_shard_idx: int = 0,
    ckpt_shard_total: int = 1,
):
    """Scan dt* folders under base_dir, evaluate each checkpoint on t2 dataset,
    and write metrics CSV under each version folder. The CSV filename includes t1 and t2.

    Sharding:
    - ckpt_shard_idx / ckpt_shard_total split the list of pending checkpoints within each
      version_dir into disjoint subsets by index (i % shard_total == shard_idx).
      This enables safe parallelization across Slurm array tasks or multiple jobs
      without overlapping work or duplicate rows.
    """
    use_cols = ['iteration', 'step', ('val_phwis' if eval_method == 'phwis' else 'val_wis'), 'val_ess', 'eval_time_sec']

    # f'cross_metrics_pibnn-{pib_type}_t1-{t1}h_to_t2-{t2}h_clipping{clipping_value}.csv'
    metrics_name = f'cross_metrics_{eval_method}_pib-dataset_t1-{t1}h_to_t2-{t2}h'

    for version_dir in sorted(base_dir.glob('dt*')):
        if not version_dir.is_dir():
            continue
        print(f'→ Processing {version_dir.name}')

        ckpt_dir = version_dir / 'checkpoints'
        if not ckpt_dir.exists():
            print(f' Skipping {version_dir} (no checkpoints)')
            continue

        # prepare CSV header (create if missing)
        new_csv = version_dir / metrics_name
        if not new_csv.exists():
            pd.DataFrame(columns=use_cols).to_csv(new_csv, index=False)
        else:
            existing_header = pd.read_csv(new_csv, nrows=0).columns.tolist()
            if existing_header != use_cols:
                existing_df = pd.read_csv(new_csv)
                for col in use_cols:
                    if col not in existing_df.columns:
                        existing_df[col] = np.nan
                existing_df = existing_df[use_cols]
                existing_df.to_csv(new_csv, index=False)

        # collect checkpoint files (deduplicated)
        files_v1 = list(ckpt_dir.glob('step=*-v1.ckpt'))
        files_all = list(ckpt_dir.glob('step=*.ckpt'))
        from collections import OrderedDict
        combined = files_v1 + files_all
        unique_files = list(OrderedDict((p.resolve(), p) for p in combined).values())
        ckpt_files = sorted(unique_files, key=lambda p: _parse_iteration_from_path(p))

        # read existing iterations so we don't duplicate rows
        try:
            existing_df = pd.read_csv(new_csv, usecols=['iteration'])
            existing_iters = set(existing_df['iteration'].astype(int).tolist())
        except Exception:
            existing_iters = set()

        # build pending list (skip already-present iterations quickly)
        pending = [p for p in ckpt_files if _parse_iteration_from_path(p) not in existing_iters]
        if not pending:
            print(f' No new checkpoints to evaluate in {version_dir.name}')
            continue

        # Apply disjoint checkpoint sharding, if requested
        if ckpt_shard_total and ckpt_shard_total > 1:
            if ckpt_shard_idx < 0 or ckpt_shard_idx >= ckpt_shard_total:
                raise ValueError(f'ckpt_shard_idx {ckpt_shard_idx} out of range for total {ckpt_shard_total}')
            pending = [p for i, p in enumerate(pending) if (i % ckpt_shard_total) == ckpt_shard_idx]
            if not pending:
                print(f' Shard {ckpt_shard_idx}/{ckpt_shard_total} has nothing to do in {version_dir.name}')
                continue

        # Decide parallel strategy
        rows = []
        if running_device == 'cpu' or not torch.cuda.is_available():
            # CPU path: safe to parallelize
            if parallel and len(pending) > 1:
                jobs = min(n_jobs, len(pending))
                rows = Parallel(n_jobs=jobs)(
                    delayed(evaluate_checkpoint_cross)(
                        ckpt,
                        data_t2_path,
                        t1,
                        t2,
                        clipping_value,
                        pib_type,
                        running_device='cpu',
                        eps=eps,
                        eval_method=eval_method,
                    )
                    for ckpt in pending
                )
            else:
                for ckpt in pending:
                    rows.append(
                        evaluate_checkpoint_cross(
                            ckpt,
                            data_t2_path,
                            t1,
                            t2,
                            clipping_value,
                            pib_type,
                            running_device='cpu',
                            eps=eps,
                            eval_method=eval_method,
                        )
                    )
        else:
            # GPU path: if multiple GPUs available, distribute jobs across them
            num_gpus = torch.cuda.device_count()
            if num_gpus <= 1:
                # single GPU: run sequentially to avoid contention
                for ckpt in pending:
                    rows.append(
                        evaluate_checkpoint_cross(
                            ckpt,
                            data_t2_path,
                            t1,
                            t2,
                            clipping_value,
                            pib_type,
                            running_device='cuda:0',
                            eps=eps,
                            eval_method=eval_method,
                        )
                    )
            else:
                jobs = min(n_jobs, len(pending))
                # assign GPUs round-robin by job index
                rows = Parallel(n_jobs=jobs)(
                    delayed(evaluate_checkpoint_cross)(
                        ckpt,
                        data_t2_path,
                        t1,
                        t2,
                        clipping_value,
                        pib_type,
                        running_device=f'cuda:{(i % num_gpus)}',
                        eps=eps,
                        eval_method=eval_method,
                    )
                    for i, ckpt in enumerate(pending)
                )

        # append rows to CSV one-by-one to preserve incremental behavior
        written = 0
        for metrics in rows:
            if metrics is None:
                continue
            it = int(metrics.get('iteration', -1))
            # Acquire a simple cross-process file lock to avoid CSV write races
            lock_path = str(new_csv) + '.lock'
            while True:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    os.close(fd)
                    break
                except FileExistsError:
                    time.sleep(0.1 + random.random() * 0.2)

            try:
                # Re-check existing iterations under the lock to prevent duplicates
                try:
                    _existing_df = pd.read_csv(new_csv, usecols=['iteration'])
                    _existing_iters = set(_existing_df['iteration'].astype(int).tolist())
                except Exception:
                    _existing_iters = set()

                if it in _existing_iters:
                    print(f' Skipping append for iteration={it} (already present)')
                    continue
                pd.DataFrame([metrics])[use_cols].to_csv(new_csv, mode='a', header=False, index=False)
                existing_iters.add(it)
                written += 1
            finally:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass

        print(f' Wrote {new_csv} ({written} new entries, total candidates={len(ckpt_files)})')

def submit_to_slurm(t1: int,
                    t2: int,
                    pib_type: str = 'policy',
                    gpu: bool = True,
                    partition: str = 'hopper',
                    cpus_per_task: int = 2,
                    n_jobs: int = 1,
                    walltime: str = '04:00:00',
                    conda_env: str = 'rl4h_rep_new',
                    project_root_unix: str = '/local/scratch/ysun564/project/OfflineRL_TimeStep',
                    eval_method: str = 'wis',
                    array_size: int = 0):
    """Create and submit an sbatch script to run cross-model selection on Slurm.

    This writes a small bash script which activates the conda env, sets PYTHONPATH
    to the project root, and runs a python here-doc that calls
    rebuild_metrics_cross(...) with the provided parameters.

    Only works on non-Windows hosts (Slurm clusters).
    Returns the sbatch submission output (stdout) on success.
    """
    system = platform.system()
    if system == 'Windows':
        raise RuntimeError('submit_to_slurm can only run on a Unix/Slurm host')

    # Where to place sbatch scripts and logs
    slurm_dir = Path(project_root_unix) / 'RL_mimic_sepsis' / 'slurm' / 'cross_model_selection' / TASK_NAME
    slurm_dir.mkdir(parents=True, exist_ok=True)

    # Build sbatch header
    job_name = f'cross_ms_t1{t1}_t2{t2}_pib-{pib_type}'
    sbatch_lines = [
        '#!/bin/bash',
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --output={slurm_dir}/{job_name}-%j.out',
        f'#SBATCH --error={slurm_dir}/{job_name}-%j.err',
        f'#SBATCH --time={walltime}',
        f'#SBATCH --cpus-per-task={cpus_per_task}',
        f'#SBATCH --partition={partition}',
        f'#SBATCH --mem=4G'
    ]

    if gpu:
        # request one GPU
        sbatch_lines.append('#SBATCH --gres=gpu:1')
    # Slurm array for safe sharding across tasks
    if array_size and array_size > 1:
        sbatch_lines.append(f'#SBATCH --array=0-{array_size-1}')

    # Activate conda and run the python here-doc
    sbatch_lines += [
        'echo "Starting job on $(hostname)"',
        f'eval "$(/local/scratch/ysun564/anaconda3/bin/conda shell.bash hook)"',
        f'conda activate {conda_env}',
        f'export PYTHONPATH={project_root_unix}:$PYTHONPATH',
        'python - <<\'PY\'',
        'import os',
        'import traceback',
        'from RL_mimic_sepsis.e_fair_comparison.cross_model_selection import rebuild_metrics_cross, _paths_for_system',
        f't1 = {t1}',
        f't2 = {t2}',
        f'pib_type = "{pib_type}"',
        f'eps = 0.1',
        f'clipping_value = 1.438',
        f'running_device = "cuda:0" if {str(gpu)} else "cpu"',
        f'n_jobs = {n_jobs}',
        f'eval_method = "{eval_method}"',
        'base_dir, data_t2_path = _paths_for_system(t1, t2)',
        'shard_idx = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))',
        f'shard_total = {max(1, array_size)}',
        'print(f"Running rebuild_metrics_cross t1={t1}, t2={t2}, pib={pib_type}, method={eval_method}, device={running_device}, shard={shard_idx}/{shard_total}")',
        'try:',
    '    rebuild_metrics_cross(data_t2_path, base_dir, t1, t2, clipping_value, parallel=False, pib_type=pib_type, running_device=running_device, eps=eps, n_jobs=n_jobs, eval_method=eval_method, ckpt_shard_idx=shard_idx, ckpt_shard_total=shard_total)',
        '    print("rebuild_metrics_cross finished successfully")',
        'except Exception as e:',
        '    print("Exception during rebuild_metrics_cross:")',
        '    traceback.print_exc()',
        '    raise',
        'PY',
    ]

    script_text = '\n'.join(sbatch_lines)

    # write to file
    fd, script_path = tempfile.mkstemp(suffix='.sh', dir=str(slurm_dir))
    os.close(fd)
    with open(script_path, 'w') as f:
        f.write(script_text)

    # submit
    try:
        res = subprocess.run(['sbatch', script_path], capture_output=True, text=True, check=True)
        print('Submitted sbatch:', res.stdout.strip())
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print('sbatch failed:', e.stderr)
        raise


if __name__ == '__main__':
    # Choose between a single run ("folder") or multiple manually specified combinations ("grid").
    run_type = 'grid' 

    default_clipping_value = 1.438
    default_eps = 0.1
    eval_method = 'phwis'

    if run_type == 'folder':
        # Example pair: evaluate models trained at t1 on dataset at t2
        t1 = 1
        t2 = 8
        pib_type = 'policy'  # 'policy' or 'dataset'
        running_device = 'cpu'
        parallel = False
        n_jobs = 8

        base_dir, data_t2_path = _paths_for_system(t1, t2)
        rebuild_metrics_cross(
            data_t2_path,
            base_dir,
            t1,
            t2,
            default_clipping_value,
            parallel=parallel,
            pib_type=pib_type,
            eps=default_eps,
            running_device=running_device,
            n_jobs=n_jobs,
            eval_method = eval_method
        )

    elif run_type == 'grid':
        grid_jobs = [



            {
                't1': 4, 't2': 8, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 2, 't2': 8, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 2, 't2': 4, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 1, 't2': 2, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 1, 't2': 4, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            
            {
                't1': 1, 't2': 8, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 8, 't2': 1, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 8, 't2': 2, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 8, 't2': 4, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 4, 't2': 1, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 4, 't2': 2, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },
            
            {
                't1': 2, 't2': 1, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },

            {
                't1': 1, 't2': 1, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },
# , 'array_size': 5
            {
                't1': 2, 't2': 2, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },
            {
                't1': 4, 't2': 4, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },
            {
                't1': 8, 't2': 8, 'pib_type': 'dataset', 'running_device': 'cpu', 'parallel': False, 'n_jobs': 8,
                'clipping_value': default_clipping_value, 'eps': default_eps,
            },                   

        ]
        # Toggle this flag to switch between local evaluation and Slurm submission.
        grid_submit = True

        # Default Slurm parameters (can be overridden per job dict).
        default_partition = 'hopper'
        default_walltime = '36:00:00'
        default_conda_env = 'rl4h_rep_new'
        default_cpus_per_task = 2

        # If submitting on a non-Unix system, automatically fall back to local runs.
        if platform.system() == 'Windows' and grid_submit:
            print('grid_submit=True but running on Windows; falling back to local execution.')
            grid_submit = False

        if grid_submit:
            print(f'Submitting {len(grid_jobs)} grid jobs to Slurm (one per combination).')
        else:
            print(f'Executing {len(grid_jobs)} grid jobs locally.')

        for idx, job in enumerate(grid_jobs, start=1):
            t1 = job['t1']
            t2 = job['t2']
            pib_type = job.get('pib_type', 'policy')
            running_device = job.get('running_device', 'cpu')
            parallel = job.get('parallel', False)
            n_jobs = job.get('n_jobs', 8)
            clipping_value = job.get('clipping_value', default_clipping_value)
            eps = job.get('eps', default_eps)

            # Derive GPU flag from running_device unless explicitly provided.
            gpu_flag = job.get('gpu', running_device.startswith('cuda'))
            partition = job.get('partition', default_partition)
            walltime = job.get('walltime', default_walltime)
            conda_env = job.get('conda_env', default_conda_env)
            cpus_per_task = job.get('cpus_per_task', default_cpus_per_task)

            print('=' * 80)
            print(
                f'Grid combo {idx}/{len(grid_jobs)} -> t1={t1}, t2={t2}, pib={pib_type}, device={running_device}, '
                f'parallel={parallel}, n_jobs={n_jobs}, submit={grid_submit}'
            )

            if grid_submit:
                try:
                    submit_out = submit_to_slurm(
                        t1=t1,
                        t2=t2,
                        pib_type=pib_type,
                        gpu=gpu_flag,
                        partition=partition,
                        cpus_per_task=cpus_per_task,
                        n_jobs=n_jobs,
                        walltime=walltime,
                        conda_env=conda_env,
                        eval_method = eval_method,
                        array_size=job.get('array_size', 0)
                    )
                    print(f'Submitted Slurm job for (t1={t1}, t2={t2}, pib={pib_type}) -> {submit_out}')
                except Exception as e:
                    print(f'Failed to submit Slurm job for (t1={t1}, t2={t2}, pib={pib_type}): {e}')
            else:
                base_dir, data_t2_path = _paths_for_system(t1, t2)
                rebuild_metrics_cross(
                    data_t2_path,
                    base_dir,
                    t1,
                    t2,
                    clipping_value,
                    parallel=parallel,
                    pib_type=pib_type,
                    eps=eps,
                    running_device=running_device,
                    n_jobs=n_jobs,
                    eval_method = eval_method,
                )

    else:
        raise ValueError(f'Unsupported run_type: {run_type}')


# /local/scratch/ysun564/anaconda3/envs/rl4h_rep_new/bin/python /local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/e_fair_comparison/cross_model_selection.py

