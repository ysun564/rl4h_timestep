#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Cross-Δt OPE evaluation.

- Iterate over all (t1, t2) in {1, 2, 4, 8} * {1, 2, 4, 8}
- Append one CSV row per finished pair (including errors)
- Ensure (t1=1, t2=8) is evaluated last
"""

import argparse
import os
import sys
import csv
import platform
import random
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve()
E_FAIR_DIR = SCRIPT_PATH.parent
RL_ROOT = E_FAIR_DIR.parent
REPO_ROOT = RL_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


from RL_mimic_sepsis.d_BCQ.src.model import BCQ
from RL_mimic_sepsis.d_BCQ.src.model_old import BCQ as BCQ_old
from RL_mimic_sepsis.e_fair_comparison.cross_evaluation import (
    evaluate_cross_dt,
    load_dataset_t2,
)
from RL_mimic_sepsis.d_BCQ.src.data import remap_rewards


T_LIST = [1, 2, 4, 8]
ACTION_SPACE = 'NormThreshold'
EPSILON = 0.1
CLIPPING = 1.438
BC_NET = True
OUT_DIR = E_FAIR_DIR
DATA_DIR = RL_ROOT / 'data'
LOGS_DIR = RL_ROOT / 'd_BCQ' / 'logs'
LOGS_RUN_ROOT = LOGS_DIR / 'BCQ_bcnet'
FIGS_DIR = RL_ROOT / 'd_BCQ' / 'figs'
SELECTED_DIR = FIGS_DIR / 'cross_pareto'
SELECTED_OUTFILE = OUT_DIR /'evaluation_results' / f'selected_checkpoint_results_clipping{CLIPPING}.csv'
BOOTSTRAP_DIR = OUT_DIR / 'bootstrap'

REWARD_ARGS = SimpleNamespace(R_immed=0.0, R_death=0.0, R_disch=100.0)


def set_random_seed(seed):
    """Set global random seeds for reproducible FQE evaluations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CKPT_METADATA = {
    1: dict(threshold=0.5, seed=1, iteration=600, suffix='-v114515'),
    2: dict(threshold=0.5, seed=1, iteration=200, suffix='-v114515'),
    4: dict(threshold=0.75, seed=4, iteration=8300, suffix='-v114515'),
    8: dict(threshold=0.5, seed=2, iteration=500, suffix=None),
}


def resolve_ckpt_path(timestep):
    """Resolve the default checkpoint path for a given timestep."""
    if timestep not in CKPT_METADATA:
        raise ValueError(f'Unsupported timestep {timestep}. Supported: {sorted(CKPT_METADATA)}')

    meta = CKPT_METADATA[timestep]
    subdir = f'dt{timestep}_threshold{meta["threshold"]}seed{int(meta["seed"])}'
    run_dir = (
        LOGS_RUN_ROOT
        / f'BCQ_as{ACTION_SPACE}_dt{timestep}h_grid_bcnet'
        / subdir
        / 'checkpoints'
    )

    suffix = meta.get('suffix') or ''
    filename = f'step={int(meta["iteration"]):04d}{suffix}.ckpt'
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f'Default checkpoint not found: {path}')
    return path


def get_test_data_path(timestep):
    """Return the test dataset path for the given timestep (t2).

    We prefer the ``...uniform`` variant but fall back to other encodings
    when necessary so the evaluation can proceed even if only weighted or
    distance-based rollouts are available.
    """

    root = DATA_DIR / f'data_as{ACTION_SPACE}_dt{timestep}h'
    candidates = [
        'episodes+encoded_state+knn_pibs_k5sqrtn_uniform',
    ]
    for candidate in candidates:
        data_dir = root / candidate
        test_path = data_dir / 'test_data.pt'
        if test_path.exists():
            return test_path



def default_outfile(eval_method, pib, clipping):
    """Select a default CSV filename based on the evaluation configuration."""
    if eval_method == 'wis':
        suffix = f'wis_{pib}_clipping{clipping}'
    elif eval_method == 'phwis':
        suffix = f'phwis_{pib}_clipping{clipping}'
    else:
        suffix = 'fqe'
    filename = f'cross_dt_grid_results_{suffix}.csv'
    return OUT_DIR / filename


def ensure_outfile(out_file, eval_method):
    """Ensure the output directory exists and the CSV has a header row."""
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        header = ['timestamp', 't1', 't2']
        if eval_method == 'wis':
            header.extend(['WIS', 'ESS'])
        elif eval_method == 'phwis':
            header.extend(['PHWIS', 'ESS'])
        else:
            header.extend(['FQE', 'ESS'])
        header.extend(['status', 'note'])
        with out_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def write_result_row(out_file, t1, t2, value, aux_metric, status, note=''):
    """Append one result row into the CSV."""
    out_path = Path(out_file)
    with out_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(timespec='seconds'),
                t1,
                t2,
                f'{value:.6f}' if value is not None else '',
                f'{aux_metric:.6f}' if aux_metric is not None else '',
                status,
                note,
            ]
        )


def ensure_selected_outfile(path):
    """Ensure the selected checkpoint result CSV exists with a header."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        header = [
            'timestamp',
            'dataset_dt',
            'policy_dt',
            'threshold',
            'seed',
            'iteration',
            'ckpt_path',
            'WIS',
            'ESS',
            'PHWIS',
            'PHWIS_ESS',
            'FQE',
        ]
        with out_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def load_selected_records(selected_dir):
    """Load selected checkpoint metadata from CSV files."""
    selected_dir = Path(selected_dir)
    if not selected_dir.exists():
        print(f'Selected directory not found: {selected_dir}', flush=True)
        return []
    records = []
    for dt in T_LIST:
        csv_path = selected_dir / f'selected_checkpoints_dataset_{dt}h.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if 'dataset_dt' not in df.columns:
            df['dataset_dt'] = dt
        for row in df.to_dict('records'):
            records.append(row)
    return records


def filter_selected_records(records, args):
    """Filter selected records by pair arguments or t1/t2 lists.

    Priority:
      1) If --pair-t1/--pair-t2 is set, keep only that single pair.
      2) If --pairs is set, keep only those explicit pairs.
      3) Else, if --t1-list and/or --t2-list are set, keep cross filter.
      4) Else, keep all records.
    """
    # Single pair override.
    if getattr(args, 'pair_t1', None) is not None and getattr(args, 'pair_t2', None) is not None:
        t1 = int(args.pair_t1)
        t2 = int(args.pair_t2)
        return [r for r in records if int(r.get('policy_dt')) == t1 and int(r.get('dataset_dt')) == t2]

    # Explicit pairs list.
    if getattr(args, 'pairs', None):
        allowed = set(parse_pairs_list(args.pairs))
        return [
            r for r in records
            if (int(r.get('policy_dt')), int(r.get('dataset_dt'))) in allowed
        ]

    # Cross filter from lists.
    t1_list = parse_int_list(getattr(args, 't1_list', None))
    t2_list = parse_int_list(getattr(args, 't2_list', None))
    if t1_list is not None or t2_list is not None:
        if t1_list is None:
            t1_list = T_LIST
        if t2_list is None:
            t2_list = T_LIST
        t1_set = set(int(x) for x in t1_list)
        t2_set = set(int(x) for x in t2_list)
        return [
            r for r in records
            if int(r.get('policy_dt')) in t1_set and int(r.get('dataset_dt')) in t2_set
        ]

    return records


def find_checkpoint_path(run_dir, iteration):
    """Locate the checkpoint file for a given run directory and iteration."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / 'checkpoints'
    if not ckpt_dir.exists():
        raise FileNotFoundError(f'Checkpoint folder missing: {ckpt_dir}')
    iteration = int(iteration)
    patterns = [f'step={iteration}']
    if iteration >= 0:
        patterns.append(f'step={iteration:04d}')
        patterns.append(f'step={iteration:05d}')
    matches = []
    last_pattern = None
    for pattern in patterns:
        last_pattern = pattern
        matches = sorted(ckpt_dir.glob(f'{pattern}*.ckpt'))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(f'No checkpoint matching {last_pattern}*.ckpt under {ckpt_dir}')
    return matches[0]


def evaluate_selected_checkpoint(record, args, dataset_cache, data_path_cache):
    """Evaluate WIS and FQE for a selected checkpoint record."""
    dataset_dt = int(record['dataset_dt'])
    policy_dt = int(record['policy_dt'])
    iteration = int(record['iteration'])
    threshold = record.get('threshold')
    seed = record.get('seed')
    run_dir = record.get('run_dir')
    if not run_dir:
        raise ValueError('run_dir missing from selected record')
    ckpt_path = find_checkpoint_path(run_dir, iteration)



    if BC_NET == True:
        model = BCQ_old.load_from_checkpoint(str(ckpt_path))
    else:
        model = BCQ.load_from_checkpoint(str(ckpt_path))
    model.eval()
    device = _resolve_device(getattr(args, 'device', 'auto'))
    if device == 'cuda':
        model.to(device)

    if dataset_dt not in data_path_cache:
        data_path_cache[dataset_dt] = get_test_data_path(dataset_dt)
    data_path = data_path_cache[dataset_dt]

    if dataset_dt not in dataset_cache:
        dataset_t2 = load_dataset_t2(str(data_path), dataset_dt, action_space=ACTION_SPACE)
        dataset_t2.reward = remap_rewards(dataset_t2.reward, REWARD_ARGS)
        # Keep episodic data on CPU for WIS/PHWIS (these paths use .numpy()).
        # For FQE, eval buffer can remain on CPU as well; device will be respected in inner code.
        dataset_cache[dataset_dt] = dataset_t2
    dataset_t2 = dataset_cache[dataset_dt]

    wis_value = None
    wis_aux = None
    phwis_value = None
    phwis_aux = None

    if args.eval_method == 'phwis':
        phwis_value, phwis_aux = evaluate_cross_dt(
            model,
            dataset_t2,
            policy_dt,
            dataset_dt,
            eps=args.eps,
            clipping=args.clipping,
            pib=args.pib,
            eval_method='phwis',
            data_path_t2=str(data_path),
            reward_args=REWARD_ARGS,
        )
    else:
        wis_value, wis_aux = evaluate_cross_dt(
            model,
            dataset_t2,
            policy_dt,
            dataset_dt,
            eps=args.eps,
            clipping=args.clipping,
            pib=args.pib,
            eval_method='wis',
            data_path_t2=str(data_path),
            reward_args=REWARD_ARGS,
        )

    set_random_seed(42)
    fqe_value, _ = evaluate_cross_dt(
        model,
        dataset_t2,
        policy_dt,
        dataset_dt,
        eps=args.eps,
        clipping=args.clipping,
        pib=args.pib,
        eval_method='fqe',
        data_path_t2=str(data_path),
        reward_args=REWARD_ARGS,
    )

    del model
    # Safely convert potential torch tensors (including CUDA) to Python floats
    def _to_float(x):
        if x is None:
            return None
        if isinstance(x, (float, int)):
            return float(x)
        if hasattr(x, 'item') and getattr(x, 'numel', lambda:1)() == 1:
            try:
                return float(x.detach().cpu().item())
            except Exception:
                return float(x.item())  # last resort (will still fail if CUDA tensor)
        return float(x)  # fallback; may raise if unsupported type

    return {
        'dataset_dt': dataset_dt,
        'policy_dt': policy_dt,
        'threshold': threshold,
        'seed': seed,
        'iteration': iteration,
        'ckpt_path': str(ckpt_path),
        'WIS': _to_float(wis_value),
        'ESS': _to_float(wis_aux),
        'PHWIS': _to_float(phwis_value),
        'PHWIS_ESS': _to_float(phwis_aux),
        'FQE': _to_float(fqe_value),
    }


def evaluate_selected_checkpoints(selected_dir, outfile, args):
    """Evaluate all selected checkpoints and append results to CSV."""
    records = load_selected_records(selected_dir)
    if not records:
        print('No selected checkpoints found.', flush=True)
        return
    ensure_selected_outfile(outfile)
    dataset_cache = {}
    data_path_cache = {}
    for record in records:
        try:
            result = evaluate_selected_checkpoint(record, args, dataset_cache, data_path_cache)
            with Path(outfile).open('a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                t1 = result['policy_dt']
                t2 = result['dataset_dt']
                w = result.get('WIS')
                ess = result.get('ESS')
                phw = result.get('PHWIS')
                phess = result.get('PHWIS_ESS')
                fqe = result['FQE']
                w_val = f'{w:.6f}' if w is not None else ''
                ess_val = f'{ess:.6f}' if ess is not None else ''
                phw_val = f'{phw:.6f}' if phw is not None else ''
                phess_val = f'{phess:.6f}' if phess is not None else ''
                fqe_val = f'{fqe:.6f}' if fqe is not None else ''
                writer.writerow(
                    [
                        datetime.now().isoformat(timespec='seconds'),
                        t2,
                        t1,
                        result['threshold'],
                        result['seed'],
                        result['iteration'],
                        result['ckpt_path'],
                        w_val,
                        ess_val,
                        phw_val,
                        phess_val,
                        fqe_val,
                    ]
                )
            t1 = result['policy_dt']
            t2 = result['dataset_dt']
            w = result.get('WIS')
            ess = result.get('ESS')
            phw = result.get('PHWIS')
            phess = result.get('PHWIS_ESS')
            fqe = result['FQE']
            label = 'PHWIS' if phw is not None else 'WIS'
            val = phw if phw is not None else w
            ess_val = phess if phw is not None else ess
            print(
                'Selected checkpoint evaluated: '
                f't1={t1}h, t2={t2}h, '
                f'{label}={val:.4f}, ESS={ess_val:.2f}, '
                f'FQE={fqe:.4f}',
                flush=True,
            )
        except Exception as exc:
            policy_dt = record.get('policy_dt')
            dataset_dt = record.get('dataset_dt')
            iteration = record.get('iteration')
            print(
                'Failed to evaluate selected checkpoint '
                f'(policy_dt={policy_dt}, dataset_dt={dataset_dt}, iteration={iteration}): {exc}',
                flush=True,
            )


def default_bootstrap_outfile(eval_method, pib):
    """Default location for bootstrap summaries."""
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    return BOOTSTRAP_DIR / f'bootstrap_{eval_method}_{pib}.csv'


def default_pair_bootstrap_outfile(eval_method, pib, t1, t2):
    """Default per-pair bootstrap CSV to avoid write contention in multi-job runs."""
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    return BOOTSTRAP_DIR / f'bootstrap_{eval_method}_{pib}_t1{int(t1)}_t2{int(t2)}.csv'


def ensure_bootstrap_outfile(path):
    """Ensure the bootstrap CSV has a header."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        header = [
            'timestamp',
            'iteration',
            't1',
            't2',
            'value',
            'aux_metric',
            'eval_method',
            'pib',
        ]
        with out_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def ensure_selected_bootstrap_outfile(path):
    """Ensure the selected-bootstrap CSV has a header."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        header = [
            'timestamp',
            'record_index',
            'policy_dt',
            'dataset_dt',
            'iteration',
            'threshold',
            'seed',
            'ckpt_path',
            'bootstrap_iter',
            'value',
            'aux_metric',
            'eval_method',
            'pib',
        ]
        with out_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def bootstrap_sample(dataset, indices):
    """Create a bootstrap resample of the provided episodic dataset."""
    # Ensure index tensor lives on the same device as data tensors to avoid device mismatch errors.
    if isinstance(indices, np.ndarray):
        idx = torch.from_numpy(indices).long()
    else:
        idx = torch.as_tensor(indices, dtype=torch.long)
    data_device = getattr(dataset.state, 'device', None)
    if data_device is not None:
        idx = idx.to(data_device)
    return SimpleNamespace(
        state=dataset.state.index_select(0, idx),
        action=dataset.action.index_select(0, idx),
        reward=dataset.reward.index_select(0, idx),
        not_done=dataset.not_done.index_select(0, idx),
        pibs=dataset.pibs.index_select(0, idx),
        estm_pibs=dataset.estm_pibs.index_select(0, idx),
    )


def run_bootstrap(args):
    """Execute bootstrap evaluations for every (t1, t2) pair."""
    device = _resolve_device(getattr(args, 'device', 'auto'))
    outfile = (
        Path(args.bootstrap_outfile)
        if args.bootstrap_outfile
        else default_bootstrap_outfile(args.eval_method, args.pib)
    ).resolve()
    ensure_bootstrap_outfile(outfile)

    dataset_cache = {}
    data_path_cache = {}
    model_cache = {}

    for t1 in T_LIST:
        ckpt_path = resolve_ckpt_path(t1)
        model = BCQ.load_from_checkpoint(str(ckpt_path))
        model.eval()
        if device == 'cuda':
            model.to(device)
        model_cache[t1] = model

    for dataset_dt in T_LIST:
        data_path = get_test_data_path(dataset_dt)
        data_path_cache[dataset_dt] = data_path
        dataset_t2 = load_dataset_t2(str(data_path), dataset_dt, action_space=ACTION_SPACE)
        dataset_t2.reward = remap_rewards(dataset_t2.reward, REWARD_ARGS)
        # For WIS/PHWIS, keep episodic data on CPU (code converts to numpy internally).
        # For FQE, eval buffer can stay on CPU; training uses transition buffer and model device.
        dataset_cache[dataset_dt] = dataset_t2

    rng = np.random.default_rng(args.bootstrap_seed)
    iterations = args.bootstrap_iterations

    with Path(outfile).open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        combos = build_custom_combos(args) or [(a, b) for a in T_LIST for b in T_LIST]
        for t1, t2 in combos:
            model = model_cache[t1]
            dataset_t2 = dataset_cache[t2]
            episode_count = dataset_t2.state.shape[0]
            data_path = str(data_path_cache[t2])

            for iteration in range(iterations):
                indices = rng.integers(0, episode_count, size=episode_count)
                resampled_dataset = bootstrap_sample(dataset_t2, indices)

                if args.eval_method == 'fqe':
                    set_random_seed(42)

                # Keep resampled dataset on CPU to avoid .numpy() issues in WIS/PHWIS logic.

                value, aux_metric = evaluate_cross_dt(
                    model,
                    resampled_dataset,
                    t1,
                    t2,
                    eps=args.eps,
                    clipping=args.clipping,
                    pib=args.pib,
                    eval_method=args.eval_method,
                    data_path_t2=data_path,
                    reward_args=REWARD_ARGS,
                )

                writer.writerow(
                    [
                        datetime.now().isoformat(timespec='seconds'),
                        iteration,
                        t1,
                        t2,
                        f'{value:.6f}' if value is not None else '',
                        f'{aux_metric:.6f}' if aux_metric is not None else '',
                        args.eval_method,
                        args.pib,
                    ]
                )
                # For FQE runs, flush each row immediately so progress is visible on disk
                if args.eval_method == 'fqe':
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass

            print(
                f'Bootstrap completed for t1={t1}h, t2={t2}h ({iterations} samples).',
                flush=True,
            )

    print(f'Bootstrap results saved to: {outfile}', flush=True)

    for model in model_cache.values():
        del model


def run_bootstrap_selected(args):
    """Execute bootstrap evaluations for each selected checkpoint record only."""
    records = load_selected_records(args.selected_dir)
    records = filter_selected_records(records, args)
    if not records:
        print('No selected checkpoints for bootstrap.', flush=True)
        return

    outfile = (
        Path(args.bootstrap_outfile)
        if args.bootstrap_outfile
        else (BOOTSTRAP_DIR / f'selected_bootstrap_{args.eval_method}_{args.pib}.csv')
    ).resolve()
    ensure_selected_bootstrap_outfile(outfile)

    device = _resolve_device(getattr(args, 'device', 'auto'))
    dataset_cache = {}
    data_path_cache = {}
    rng = np.random.default_rng(args.bootstrap_seed)
    iterations = args.bootstrap_iterations

    with Path(outfile).open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for idx, record in enumerate(records):
            try:
                policy_dt = int(record['policy_dt'])
                dataset_dt = int(record['dataset_dt'])
                iteration_ckpt = int(record['iteration'])
                threshold = record.get('threshold')
                seed = record.get('seed')
                run_dir = record.get('run_dir')
                if not run_dir:
                    raise ValueError('run_dir missing from selected record')

                ckpt_path = find_checkpoint_path(run_dir, iteration_ckpt)
                model = (BCQ_old if BC_NET else BCQ).load_from_checkpoint(str(ckpt_path))
                model.eval()
                if device == 'cuda':
                    model.to(device)

                if dataset_dt not in data_path_cache:
                    data_path_cache[dataset_dt] = get_test_data_path(dataset_dt)
                data_path = data_path_cache[dataset_dt]

                if dataset_dt not in dataset_cache:
                    ds = load_dataset_t2(str(data_path), dataset_dt, action_space=ACTION_SPACE)
                    ds.reward = remap_rewards(ds.reward, REWARD_ARGS)
                    dataset_cache[dataset_dt] = ds
                base_ds = dataset_cache[dataset_dt]
                episode_count = base_ds.state.shape[0]

                for b in range(iterations):
                    indices = rng.integers(0, episode_count, size=episode_count)
                    ds_boot = bootstrap_sample(base_ds, indices)
                    if args.eval_method == 'fqe':
                        set_random_seed(42)

                    value, aux_metric = evaluate_cross_dt(
                        model,
                        ds_boot,
                        policy_dt,
                        dataset_dt,
                        eps=args.eps,
                        clipping=args.clipping,
                        pib=args.pib,
                        eval_method=args.eval_method,
                        data_path_t2=str(data_path),
                        reward_args=REWARD_ARGS,
                    )

                    writer.writerow(
                        [
                            datetime.now().isoformat(timespec='seconds'),
                            idx,
                            policy_dt,
                            dataset_dt,
                            iteration_ckpt,
                            threshold,
                            seed,
                            str(ckpt_path),
                            b,
                            f'{value:.6f}' if value is not None else '',
                            f'{aux_metric:.6f}' if aux_metric is not None else '',
                            args.eval_method,
                            args.pib,
                        ]
                    )
                    if args.eval_method == 'fqe':
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass

                print(
                    f'Selected bootstrap completed for record={idx} (t1={policy_dt}h, t2={dataset_dt}h).',
                    flush=True,
                )
                del model
            except Exception as e:
                print(f'Error bootstrapping selected record {idx}: {e}', flush=True)

    print(f'Selected bootstrap results saved to: {outfile}', flush=True)


def submit_slurm_bootstrap(args):
    """Submit an sbatch job that runs the bootstrap procedure on a cluster."""
    if platform.system().lower().startswith('win'):
        print('SLURM submission is only supported on POSIX platforms.', flush=True)
        return

    script_dir = RL_ROOT / 'slurm'
    script_dir.mkdir(parents=True, exist_ok=True)

    # Build combination list; if --pairs is provided, we'll submit one job per pair
    combos = build_custom_combos(args) or [(a, b) for a in T_LIST for b in T_LIST]

    if getattr(args, 'pairs', None):
        # Per-pair submission when explicit pairs are provided
        submitted = 0
        for (t1, t2) in combos:
            per_outfile = (
                Path(args.bootstrap_outfile)
                if args.bootstrap_outfile
                else default_pair_bootstrap_outfile(args.eval_method, args.pib, t1, t2)
            )

            command_args = [
                str(Path(sys.executable).resolve()),
                str(SCRIPT_PATH),
                '--bootstrap-run',
                '--bootstrap-iterations', str(args.bootstrap_iterations),
                '--bootstrap-seed', str(args.bootstrap_seed),
                '--eval-method', args.eval_method,
                '--pib', args.pib,
                '--eps', str(args.eps),
                '--clipping', str(args.clipping),
                '--skip-grid',
                '--bootstrap-outfile', str(per_outfile),
                '--device', args.device,
                '--bootstrap-batch-size', str(args.bootstrap_batch_size),
                '--pair-t1', str(int(t1)),
                '--pair-t2', str(int(t2)),
            ]
            if getattr(args, 'amp', False):
                command_args.append('--amp')
            command = ' '.join(shlex.quote(part) for part in command_args)

            script_path = script_dir / f'cross_bcq_bootstrap_t1{t1}_t2{t2}.sbatch'
            log_path = (script_dir / f'cross_bcq_bootstrap_t1{t1}_t2{t2}_%j.log').resolve()

            script_contents = dedent(
                f'''#!/bin/bash
#SBATCH --job-name=bcq_bs_t{t1}-{t2}
#SBATCH --output={log_path}
#SBATCH --time={args.slurm_time}
#SBATCH --cpus-per-task={args.slurm_cpus}
#SBATCH --mem={args.slurm_mem}
'''
            )
            if args.slurm_partition:
                script_contents += f'#SBATCH --partition={args.slurm_partition}\n'
            if args.slurm_account:
                script_contents += f'#SBATCH --account={args.slurm_account}\n'
            if getattr(args, 'slurm_gpus', 0) > 0:
                script_contents += f'#SBATCH --gres=gpu:{int(args.slurm_gpus)}\n'

            script_contents += dedent(
                f'''
set -euo pipefail
cd {shlex.quote(str(REPO_ROOT))}
{command}
'''
            )

            script_path.write_text(script_contents)
            try:
                script_path.chmod(0o755)
            except PermissionError:
                pass
            print(f'Bootstrap SLURM script written to: {script_path}', flush=True)

            result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f'sbatch submission failed for (t1={t1}, t2={t2}): {result.stderr.strip()}'
                )
            print(result.stdout.strip(), flush=True)
            submitted += 1

        print(f'Submitted {submitted} per-pair bootstrap jobs.', flush=True)
        return

    # Fallback: single job processing all combos (forward filters if provided)
    outfile = (
        Path(args.bootstrap_outfile)
        if args.bootstrap_outfile
        else default_bootstrap_outfile(args.eval_method, args.pib)
    )

    script_path = script_dir / 'cross_bcq_bootstrap.sbatch'
    log_path = (script_dir / 'cross_bcq_bootstrap_%j.log').resolve()

    command_args = [
        str(Path(sys.executable).resolve()),
        str(SCRIPT_PATH),
        '--bootstrap-run',
        '--bootstrap-iterations', str(args.bootstrap_iterations),
        '--bootstrap-seed', str(args.bootstrap_seed),
        '--eval-method', args.eval_method,
        '--pib', args.pib,
        '--eps', str(args.eps),
        '--clipping', str(args.clipping),
        '--skip-grid',
        '--bootstrap-outfile', str(outfile),
        '--device', args.device,
        '--bootstrap-batch-size', str(args.bootstrap_batch_size),
    ]
    if getattr(args, 'amp', False):
        command_args.append('--amp')
    # Forward custom selection filters
    if getattr(args, 'pairs', None):
        command_args += ['--pairs', args.pairs]
    if getattr(args, 't1_list', None):
        command_args += ['--t1-list', args.t1_list]
    if getattr(args, 't2_list', None):
        command_args += ['--t2-list', args.t2_list]

    command = ' '.join(shlex.quote(part) for part in command_args)

    script_contents = dedent(
        f'''#!/bin/bash
#SBATCH --job-name=bcq_bootstrap
#SBATCH --output={log_path}
#SBATCH --time={args.slurm_time}
#SBATCH --cpus-per-task={args.slurm_cpus}
#SBATCH --mem={args.slurm_mem}
'''
    )

    if args.slurm_partition:
        script_contents += f'#SBATCH --partition={args.slurm_partition}\n'
    if args.slurm_account:
        script_contents += f'#SBATCH --account={args.slurm_account}\n'
    if getattr(args, 'slurm_gpus', 0) > 0:
        script_contents += f'#SBATCH --gres=gpu:{int(args.slurm_gpus)}\n'

    script_contents += dedent(
        f'''
set -euo pipefail
cd {shlex.quote(str(REPO_ROOT))}
{command}
'''
    )

    script_path.write_text(script_contents)
    try:
        script_path.chmod(0o755)
    except PermissionError:
        pass
    print(f'Bootstrap SLURM script written to: {script_path}', flush=True)

    result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'sbatch submission failed: {result.stderr.strip()}')

    print(result.stdout.strip(), flush=True)


def submit_slurm_bootstrap_selected(args):
    """Submit an sbatch job that runs bootstrap over selected checkpoints only."""
    if platform.system().lower().startswith('win'):
        print('SLURM submission is only supported on POSIX platforms.', flush=True)
        return

    script_dir = RL_ROOT / 'slurm'
    script_dir.mkdir(parents=True, exist_ok=True)

    outfile = (
        Path(args.bootstrap_outfile)
        if args.bootstrap_outfile
        else (BOOTSTRAP_DIR / f'selected_bootstrap_{args.eval_method}_{args.pib}.csv')
    )

    script_path = script_dir / 'cross_bcq_bootstrap_selected.sbatch'
    log_path = (script_dir / 'cross_bcq_bootstrap_selected_%j.log').resolve()

    command_args = [
        str(Path(sys.executable).resolve()),
        str(SCRIPT_PATH),
        '--bootstrap-selected-run',
        '--eval-method', args.eval_method,
        '--selected-dir', str(args.selected_dir),
        '--bootstrap-iterations', str(args.bootstrap_iterations),
        '--bootstrap-seed', str(args.bootstrap_seed),
        '--pib', args.pib,
        '--eps', str(args.eps),
        '--clipping', str(args.clipping),
        '--device', args.device,
        '--bootstrap-outfile', str(outfile),
        '--skip-grid',
    ]
    if getattr(args, 'amp', False):
        command_args.append('--amp')
    # Forward selection filters to the inner run.
    if getattr(args, 'pairs', None):
        command_args += ['--pairs', args.pairs]
    if getattr(args, 't1_list', None):
        command_args += ['--t1-list', args.t1_list]
    if getattr(args, 't2_list', None):
        command_args += ['--t2-list', args.t2_list]
    if getattr(args, 'pair_t1', None) is not None and getattr(args, 'pair_t2', None) is not None:
        command_args += ['--pair-t1', str(int(args.pair_t1)), '--pair-t2', str(int(args.pair_t2))]
    command = ' '.join(shlex.quote(part) for part in command_args)

    script_contents = dedent(
        f'''#!/bin/bash
#SBATCH --job-name=bcq_sel_bootstrap
#SBATCH --output={log_path}
#SBATCH --time={args.slurm_time}
#SBATCH --cpus-per-task={args.slurm_cpus}
#SBATCH --mem={args.slurm_mem}
'''
    )
    if args.slurm_partition:
        script_contents += f'#SBATCH --partition={args.slurm_partition}\n'
    if args.slurm_account:
        script_contents += f'#SBATCH --account={args.slurm_account}\n'
    if getattr(args, 'slurm_gpus', 0) > 0:
        script_contents += f'#SBATCH --gres=gpu:{int(args.slurm_gpus)}\n'

    script_contents += dedent(
        f'''
set -euo pipefail
cd {shlex.quote(str(REPO_ROOT))}
{command}
'''
    )

    script_path.write_text(script_contents)
    try:
        script_path.chmod(0o755)
    except PermissionError:
        pass
    print(f'Selected bootstrap SLURM script written to: {script_path}', flush=True)

    result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'sbatch submission failed: {result.stderr.strip()}')
    print(result.stdout.strip(), flush=True)


def submit_slurm_selected(args):
    """Submit an sbatch job that ONLY evaluates selected checkpoints.

    The generated SLURM script invokes this same Python module with the hidden
    flag --selected-run so that the launched process performs ONLY the selected
    checkpoint evaluation (and skips grid + bootstrap logic entirely).
    """
    if platform.system().lower().startswith('win'):
        print('SLURM submission is only supported on POSIX platforms.', flush=True)
        return

    script_dir = RL_ROOT / 'slurm'
    script_dir.mkdir(parents=True, exist_ok=True)

    # Resolve output file for selected checkpoints
    selected_outfile = (
        Path(args.selected_outfile)
        if args.selected_outfile
        else SELECTED_OUTFILE
    )

    script_path = script_dir / 'cross_bcq_selected.sbatch'
    log_path = (script_dir / 'cross_bcq_selected_%j.log').resolve()

    command_args = [
        str(Path(sys.executable).resolve()),
        str(SCRIPT_PATH),
        '--selected-run',  # internal flag to limit execution path
        '--skip-grid',  # ensure grid is not run even if code path changes later
        '--eval-method', args.eval_method,
        '--selected-dir', str(args.selected_dir),
        '--selected-outfile', str(selected_outfile),
        '--pib', args.pib,
        '--eps', str(args.eps),
        '--clipping', str(args.clipping),
        '--device', args.device,
    ]
    # Preserve AMP/device related options if meaningful for loading models
    if getattr(args, 'amp', False):
        command_args.append('--amp')

    command = ' '.join(shlex.quote(part) for part in command_args)

    script_contents = dedent(
        f'''#!/bin/bash
#SBATCH --job-name=bcq_selected
#SBATCH --output={log_path}
#SBATCH --time={args.slurm_time}
#SBATCH --cpus-per-task={args.slurm_cpus}
#SBATCH --mem={args.slurm_mem}
'''
    )
    if args.slurm_partition:
        script_contents += f'#SBATCH --partition={args.slurm_partition}\n'
    if args.slurm_account:
        script_contents += f'#SBATCH --account={args.slurm_account}\n'
    if getattr(args, 'slurm_gpus', 0) > 0:
        script_contents += f'#SBATCH --gres=gpu:{int(args.slurm_gpus)}\n'

    script_contents += dedent(
        f'''
set -euo pipefail
cd {shlex.quote(str(REPO_ROOT))}
{command}
'''
    )

    script_path.write_text(script_contents)
    try:
        script_path.chmod(0o755)
    except PermissionError:
        pass
    print(f'Selected SLURM script written to: {script_path}', flush=True)

    result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'sbatch submission failed: {result.stderr.strip()}')
    print(result.stdout.strip(), flush=True)


def evaluate_one_pair(t1, t2, args, out_file):
    """Evaluate one (t1, t2) pair and write a CSV row with the result."""
    try:
        ckpt_path = resolve_ckpt_path(t1)
        model = BCQ.load_from_checkpoint(str(ckpt_path))
        model.eval()
        device = _resolve_device(getattr(args, 'device', 'auto'))
        if device == 'cuda':
            model.to(device)

        data_path = get_test_data_path(t2)
        dataset_t2 = load_dataset_t2(str(data_path), t2, action_space=ACTION_SPACE)
        dataset_t2.reward = remap_rewards(
            dataset_t2.reward,
            REWARD_ARGS,
        )
        # Keep episodic tensors on CPU; WIS/PHWIS code calls .numpy() on them.

        if args.eval_method == 'fqe':
            set_random_seed(42)

        value, aux_metric = evaluate_cross_dt(
            model,
            dataset_t2,
            t1,
            t2,
            eps=args.eps,
            clipping=args.clipping,
            pib=args.pib,
            eval_method=args.eval_method,
            data_path_t2=str(data_path),
            reward_args=REWARD_ARGS,
        )

        metric_label = 'WIS' if args.eval_method == 'wis' else ('PHWIS' if args.eval_method == 'phwis' else 'FQE')
        print(
            f'Cross-Δt OPE (t1={t1}h → t2={t2}h): {metric_label}={value:.4f}, ESS={aux_metric:.2f}',
            flush=True,
        )
        write_result_row(out_file, t1, t2, value, aux_metric, 'OK', '')
    except Exception as e:
        err = f'ERROR for (t1={t1}, t2={t2}): {type(e).__name__}: {e}'
        print(err, flush=True)
        write_result_row(out_file, t1, t2, None, None, 'ERROR', err)


def parse_int_list(text):
    """Parse a comma-separated list of ints like '1,2,4,8'."""
    if text is None:
        return None
    vals = [s.strip() for s in str(text).split(',') if s.strip()]
    return [int(v) for v in vals]


def parse_pairs_list(text):
    """Parse pairs like '1-2,2-4,4-8' into list of (t1,t2)."""
    if text is None:
        return []
    pairs = []
    for token in str(text).split(','):
        tok = token.strip()
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-', 1)
        elif 'x' in tok.lower():
            a, b = tok.lower().split('x', 1)
        else:
            raise ValueError(f"Invalid pair token '{tok}'. Use 't1-t2' or 't1xt2'.")
        pairs.append((int(a), int(b)))
    return pairs


def build_custom_combos(args):
    """Build (t1,t2) combinations from CLI: --pairs or cross of --t1-list, --t2-list.

    Priority:
      1) if --pair-t1/--pair-t2 set, run only that single pair
      2) if --pairs set, use those exact pairs (order preserved)
      3) else use cross product of --t1-list and --t2-list if provided
      4) else default cross product of T_LIST

    Also place (1,8) at the end if present and no explicit ordering was provided.
    """
    # Single-pair override
    if getattr(args, 'pair_t1', None) is not None and getattr(args, 'pair_t2', None) is not None:
        return [(int(args.pair_t1), int(args.pair_t2))]

    # Exact list of pairs
    if getattr(args, 'pairs', None):
        return parse_pairs_list(args.pairs)

    # Cross product from t1/t2 lists
    t1_list = parse_int_list(getattr(args, 't1_list', None)) or T_LIST
    t2_list = parse_int_list(getattr(args, 't2_list', None)) or T_LIST
    combos = [(a, b) for a in t1_list for b in t2_list]
    # Heuristic: move (1,8) to the end if present
    if (1, 8) in combos:
        combos.remove((1, 8))
        combos.append((1, 8))
    return combos


def parse_args():
    parser = argparse.ArgumentParser(description='Batch cross-Δt OPE evaluation.')
    parser.add_argument(
        '--eval-method',
        choices=['wis', 'phwis', 'fqe'],
        default='wis',
        help='Choose the off-policy evaluation method to run.',
    )
    parser.add_argument(
        '--pib',
        choices=['policy', 'dataset'],
        default='dataset',
        help='Behavior distribution to build the numerator with (WIS only).',
    )
    parser.add_argument(
        '--outfile',
        default=None,
        help='Optional custom CSV file to append results to.',
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=EPSILON,
        help='Epsilon value for epsilon-soft policy construction.',
    )
    parser.add_argument(
        '--clipping',
        type=float,
        default=CLIPPING,
        help='Trajectory weight clipping threshold (WIS).',
    )
    parser.add_argument(
        '--selected-dir',
        default=str(SELECTED_DIR),
        help='Directory containing selected checkpoint CSV files.',
    )
    parser.add_argument(
        '--selected-outfile',
        default=None,
        help='Optional CSV file for selected checkpoint evaluation results.',
    )
    parser.add_argument(
        '--skip-grid',
        action='store_true',
        help='Skip the full grid evaluation and only process selected checkpoints.',
    )
    parser.add_argument(
        '--bootstrap-iterations',
        type=int,
        default=100,
        help='Number of bootstrap resamples per (t1, t2) pair.',
    )
    parser.add_argument(
        '--bootstrap-seed',
        type=int,
        default=2025,
        help='Random seed used for bootstrap resampling.',
    )
    parser.add_argument(
        '--bootstrap-outfile',
        default=None,
        help='Optional CSV file for bootstrap evaluation results.',
    )
    parser.add_argument(
        '--bootstrap-run',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--bootstrap-selected',
        action='store_true',
        help='Bootstrap evaluation over selected checkpoints only (local run).',
    )
    parser.add_argument(
        '--bootstrap-selected-run',
        action='store_true',
        help=argparse.SUPPRESS,  # internal flag used by SLURM wrapper script
    )
    parser.add_argument(
        '--slurm_bootstrap',
        action='store_true',
        help='Submit a SLURM job that performs bootstrap evaluation across all (t1, t2) pairs.',
    )
    parser.add_argument(
        '--slurm-bootstrap-and-selected',
        action='store_true',
        help='Submit SLURM bootstrap job and THEN locally evaluate selected checkpoints (optionally with --skip-grid).',
    )
    parser.add_argument(
        '--slurm-selected',
        action='store_true',
        help='Submit a SLURM job that evaluates ONLY selected checkpoints and exits.',
    )
    parser.add_argument(
        '--slurm-bootstrap-selected',
        action='store_true',
        help='Submit a SLURM job that performs bootstrap over selected checkpoints only.',
    )
    parser.add_argument(
        '--selected-run',
        action='store_true',
        help=argparse.SUPPRESS,  # internal flag used by SLURM wrapper script
    )
    parser.add_argument(
        '--slurm-partition',
        default=None,
        help='SLURM partition to submit bootstrap jobs to.',
    )
    parser.add_argument(
        '--slurm-time',
        default='24:00:00',
        help='Wall-clock time limit for the SLURM bootstrap job.',
    )
    parser.add_argument(
        '--slurm-mem',
        default='32G',
        help='Memory request for the SLURM bootstrap job.',
    )
    parser.add_argument(
        '--slurm-cpus',
        type=int,
        default=4,
        help='CPU cores requested for the SLURM bootstrap job.',
    )
    parser.add_argument(
        '--slurm-account',
        default=None,
        help='Optional SLURM account to charge the bootstrap job to.',
    )
    parser.add_argument(
        '--slurm-gpus',
        type=int,
        default=0,
        help='Number of GPUs to request for SLURM bootstrap job (0 = CPU only).',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Computation device for evaluation ("cuda", "cpu", or "auto").',
    )
    parser.add_argument(
        '--bootstrap-batch-size',
        type=int,
        default=1024,
        help='(Reserved) Batch size for any per-bootstrap training (FQE).',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable mixed precision (AMP) during any FQE training/eval (if supported).',
    )
    parser.add_argument(
        '--t1-list',
        default=None,
        help='Comma-separated list of t1 timesteps (e.g. "1,2,4").',
    )
    parser.add_argument(
        '--t2-list',
        default=None,
        help='Comma-separated list of t2 timesteps (e.g. "2,4,8").',
    )
    parser.add_argument(
        '--pairs',
        default=None,
        help='Explicit list of pairs like "1-2,2-4,4-8" (overrides t1/t2 lists).',
    )
    parser.add_argument(
        '--pair-t1',
        type=int,
        default=None,
        help='Single pair: t1 timestep.',
    )
    parser.add_argument(
        '--pair-t2',
        type=int,
        default=None,
        help='Single pair: t2 timestep.',
    )
    parser.add_argument(
        '--slurm-per-pair',
        action='store_true',
        help='Submit one SLURM job per (t1,t2) combo (bootstrap over iterations per job).',
    )
    return parser.parse_args()


def _resolve_device(dev):
    if dev == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return dev


def main():
    """Run the full grid evaluation and persist results incrementally."""
    args = parse_args()

    args.selected_dir = Path(args.selected_dir)
    args.selected_outfile = (
        Path(args.selected_outfile)
        if args.selected_outfile
        else SELECTED_OUTFILE
    )
    args.outfile = Path(args.outfile) if args.outfile else None
    args.bootstrap_outfile = (
        Path(args.bootstrap_outfile)
        if args.bootstrap_outfile
        else None
    )

    # Enforce new policy: default pair/grid evaluation disabled. Only selected evaluation or selected bootstrap.
    if not args.skip_grid and not args.selected_dir:
        print('Default cross-Δt grid evaluation has been disabled. Provide --selected-dir for selected evaluation.', flush=True)
        return

    # SLURM submission paths for selected-only operations
    if getattr(args, 'slurm_selected', False):
        submit_slurm_selected(args)
        return
    if getattr(args, 'slurm_bootstrap_selected', False):
        submit_slurm_bootstrap_selected(args)
        return
    if args.slurm_bootstrap:
        print('Default (all-pairs) bootstrap disabled. Use --slurm-bootstrap-selected instead.', flush=True)
        return
    if getattr(args, 'slurm_bootstrap_and_selected', False):
        print('Combined default bootstrap + local selected disabled. Use separate --slurm-bootstrap-selected and local selected run.', flush=True)
        return

    # Internal path triggered only inside SLURM selected job.
    if getattr(args, 'selected_run', False):
        evaluate_selected_checkpoints(args.selected_dir, args.selected_outfile, args)
        if args.selected_outfile.exists():
            print(f'Selected checkpoint results saved to: {args.selected_outfile}', flush=True)
        return

    if args.bootstrap_run:
        print('Default (all-pairs) bootstrap local run disabled. Use --bootstrap-selected instead.', flush=True)
        return
    if getattr(args, 'bootstrap_selected_run', False):
        run_bootstrap_selected(args)
        return
    if getattr(args, 'bootstrap_selected', False):
        run_bootstrap_selected(args)
        # Continue to selected evaluation CSV accumulation if desired.

    # Grid evaluation fully disabled; always skip.
    print('Grid (default pair) evaluation skipped (disabled).', flush=True)

    if args.selected_dir:
        evaluate_selected_checkpoints(args.selected_dir, args.selected_outfile, args)
        if args.selected_outfile.exists():
            print(f'Selected checkpoint results saved to: {args.selected_outfile}', flush=True)


if __name__ == '__main__':
    main()



r'''

cd F:\time_step\OfflineRL_FactoredActions
    & D:/Software/anaconda3/envs/rl4h_rep_new/python.exe -m RL_mimic_sepsis.e_fair_comparison.cross_BCQ_quant --skip-grid --eval-method phwis --selected-dir "F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\d_BCQ\figs\cross_pareto\bcnet_phwis\selected" --eps 0.1 --clipping 1.438 --pib dataset --selected-outfile "F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\e_fair_comparison\evaluation_results\phwis\selected_bcnet_phwis_results.csv"

/local/scratch/ysun564/anaconda3/envs/rl4h_rep_new/bin/python -m RL_mimic_sepsis.e_fair_comparison.cross_BCQ_quant \
--slurm-bootstrap-selected \
--eval-method phwis \
--selected-dir "/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/bcnet_phwis/selected" \
--bootstrap-outfile "/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/bcnet_phwis/bootstrap/selected_bcnet_phwis_boot_results.csv" \
--pib dataset --eps 0.1 --clipping 1.438 --bootstrap-iterations 100 \
--slurm-mem 64G --slurm-cpus 10 --slurm-gpus 0 --device cpu \   
  


  /local/scratch/ysun564/anaconda3/envs/rl4h_rep_new/bin/python -m RL_mimic_sepsis.e_fair_comparison.cross_BCQ_quant \
  --slurm-bootstrap-selected \
  --eval-method fqe \
  --selected-dir "/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/bcnet_phwis/selected" \
  --pib dataset \
  --eps 0.1 \
  --clipping 1.438 \
  --bootstrap-iterations 100 \
  --slurm-mem 64G \
  --slurm-cpus 8 \
  --slurm-gpus 1 \
  --device cuda \
  --pairs 1-1,1-2,1-4,1-8,4-1,4-2,4-4,4-8 \
  --bootstrap-outfile "/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/bcnet_phwis/bootstrap/selected_bcnet_fqe_boot_results_part1.csv"

/local/scratch/ysun564/anaconda3/envs/rl4h_rep_new/bin/python -m RL_mimic_sepsis.e_fair_comparison.cross_BCQ_quant \
  --slurm-bootstrap-selected \
  --eval-method fqe \
  --selected-dir "/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/bcnet_phwis/selected" \
  --pib dataset \
  --eps 0.1 \
  --clipping 1.438 \
  --bootstrap-iterations 100 \
  --slurm-mem 64G \
  --slurm-cpus 8 \
  --slurm-gpus 1 \
  --device cuda \
  --pairs 2-1,2-2,2-4,2-8,8-1,8-2,8-4,8-8 \
  --bootstrap-outfile "/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/d_BCQ/figs/cross_pareto/bcnet_phwis/bootstrap/selected_bcnet_fqe_boot_results_part2.csv"
'''