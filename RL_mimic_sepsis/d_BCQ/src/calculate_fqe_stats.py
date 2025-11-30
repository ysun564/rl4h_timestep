import pandas as pd
import numpy as np
import glob
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Summarize FQE bootstrap results into mean/median/std and 95% CI, with optional point estimates.')
    parser.add_argument('--result-dir', type=str, required=True,
                        help='Directory containing FQE bootstrap CSV parts')
    parser.add_argument('--pattern', type=str, default='selected_*fqe*_part*.csv',
                        help='Glob pattern (within result-dir) to match FQE bootstrap CSV files')
    parser.add_argument('--point-file', type=str, default=None,
                        help='CSV containing point estimates with columns: dataset_dt, policy_dt, FQE')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to write summary CSV')
    args = parser.parse_args()

    result_dir = args.result_dir
    pattern = os.path.join(result_dir, args.pattern)
    output_file = args.output_file
    point_file = args.point_file

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No bootstrap FQE files found with pattern: {pattern}')

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        # Ensure expected columns exist
        for col in ['value', 'policy_dt', 'dataset_dt']:
            if col not in df.columns:
                raise ValueError(f'Missing expected column {col} in {fp}')
        # Keep only FQE rows, just in case
        if 'eval_method' in df.columns:
            df = df[df['eval_method'].str.lower() == 'fqe']
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Group and compute statistics for FQE values
    # 95% CI via empirical quantiles (2.5%, 97.5%)
    agg = all_df.groupby(['dataset_dt', 'policy_dt'])['value'].agg(
        n_bootstrap='count',
        mean='mean',
        median='median',
        std=lambda x: x.std(ddof=1),
        ci_lower=lambda x: x.quantile(0.025),
        ci_upper=lambda x: x.quantile(0.975),
    ).reset_index()

    # Load point estimates and merge
    if point_file is not None and os.path.exists(point_file):
        pe = pd.read_csv(point_file)
        if {'dataset_dt', 'policy_dt', 'FQE'}.issubset(pe.columns):
            pe = pe[['dataset_dt', 'policy_dt', 'FQE']].copy()
            # In case of duplicates, average; typically one row per pair
            pe = pe.groupby(['dataset_dt', 'policy_dt'], as_index=False)['FQE'].mean()
            # Create point estimate column
            pe['point_estimate_fqe'] = pe['FQE']
            pe = pe.drop(columns=['FQE'])
            agg = agg.merge(pe, on=['dataset_dt', 'policy_dt'], how='left')
        else:
            print('Warning: point file missing required columns (dataset_dt, policy_dt, FQE); skipping point estimates.')
    elif point_file is not None:
        print(f'Warning: point file not found: {point_file}; skipping point estimates.')

    # Order columns (include point estimate if present)
    cols = ['dataset_dt', 'policy_dt']
    if 'point_estimate_fqe' in agg.columns:
        cols += ['point_estimate_fqe']
    cols += ['n_bootstrap', 'mean', 'median', 'std', 'ci_lower', 'ci_upper']
    agg = agg[cols].sort_values(by=['dataset_dt', 'policy_dt']).reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save
    agg.to_csv(output_file, index=False, float_format='%.6f')

    print(f'Saved FQE bootstrap summary to: {output_file}')
    print('\n--- Summary (first 20 rows) ---')
    print(agg.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
