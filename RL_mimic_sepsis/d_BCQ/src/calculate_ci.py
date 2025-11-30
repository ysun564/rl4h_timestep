
import pandas as pd
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Compute 95% CI for PHWIS and ESS and merge with point estimates.')
    parser.add_argument('--bootstrap-file', type=str, required=True,
                        help='CSV file containing PHWIS bootstrap samples with columns: dataset_dt, policy_dt, value, aux_metric')
    parser.add_argument('--point-file', type=str, required=True,
                        help='CSV file containing point estimates with columns: dataset_dt, policy_dt, PHWIS, PHWIS_ESS')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to write merged results CSV')
    args = parser.parse_args()

    bootstrap_file = args.bootstrap_file
    point_estimate_file = args.point_file
    output_file = args.output_file

    if not os.path.exists(bootstrap_file):
        raise FileNotFoundError(f'Bootstrap file not found: {bootstrap_file}')
    if not os.path.exists(point_estimate_file):
        raise FileNotFoundError(f'Point estimate file not found: {point_estimate_file}')

    # Load the bootstrap data
    bootstrap_df = pd.read_csv(bootstrap_file)

    required_boot_cols = {'dataset_dt', 'policy_dt', 'value', 'aux_metric'}
    if not required_boot_cols.issubset(bootstrap_df.columns):
        missing = required_boot_cols.difference(bootstrap_df.columns)
        raise ValueError(f'Missing required columns in bootstrap file: {missing}')

    # Group by policy and dataset time steps and calculate the 95% CI for PHWIS and ESS
    agg_df = bootstrap_df.groupby(['policy_dt', 'dataset_dt']).agg(
        phwis_ci_lower=('value', lambda x: x.quantile(0.025)),
        phwis_ci_upper=('value', lambda x: x.quantile(0.975)),
        ess_ci_lower=('aux_metric', lambda x: x.quantile(0.025)),
        ess_ci_upper=('aux_metric', lambda x: x.quantile(0.975))
    ).reset_index()

    # Load the point estimate data
    point_estimates_df = pd.read_csv(point_estimate_file)
    required_point_cols = {'dataset_dt', 'policy_dt', 'PHWIS', 'PHWIS_ESS'}
    if not required_point_cols.issubset(point_estimates_df.columns):
        missing = required_point_cols.difference(point_estimates_df.columns)
        raise ValueError(f'Missing required columns in point estimate file: {missing}')

    # Select relevant columns from point estimates
    point_estimates_df = point_estimates_df[['dataset_dt', 'policy_dt', 'PHWIS', 'PHWIS_ESS']].copy()
    point_estimates_df.rename(columns={'PHWIS': 'point_estimate_phwis', 'PHWIS_ESS': 'point_estimate_ess'}, inplace=True)

    # Merge the point estimates with the confidence intervals
    final_df = pd.merge(point_estimates_df, agg_df, on=['dataset_dt', 'policy_dt'], how='inner')

    # Reorder columns for clarity
    final_df = final_df[['dataset_dt', 'policy_dt', 'point_estimate_phwis', 'phwis_ci_lower', 'phwis_ci_upper', 'point_estimate_ess', 'ess_ci_lower', 'ess_ci_upper']]

    # Sort for readability
    final_df = final_df.sort_values(by=['dataset_dt', 'policy_dt']).reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the final results to a new CSV file
    final_df.to_csv(output_file, index=False, float_format='%.6f')

    print(f"Successfully created results file at: {output_file}")
    print("\n--- Results (first 20 rows) ---")
    print(final_df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
