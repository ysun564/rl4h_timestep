import csv
import itertools
import os

from RL_mimic_sepsis.e_fair_comparison.mapping_raw import PAIRS, CAPS, OUT_DIR
from RL_mimic_sepsis.e_fair_comparison.mapping_raw import thresholds, sum_interval_for_counts, overlap, decide_bucket

"""
Precompute lookup tables mapping ordered small-step tuples -> big-step buckets
using the Expected-Overlap rule with fixed caps.

Edit PAIRS to choose (t1 -> t2) mappings (require t2 % t1 == 0).
Edit CAPS to set Level-4 caps per horizon (hours).
Each lookup CSV lists every ordered tuple of length n = t2//t1 and the mapped
big-step bucket label in {'A0','A1','A2','A3','A4'}.

Example output filename:
  lookup_ordered_dt1h_to_dt4h.csv
Columns:
  dt{t1}h_l1, dt{t1}h_l2, ..., dt{t1}h_l{n}, dt{t2}h_bucket
"""


def ordered_tuples(length, n_levels=5):
    """
    Yield all ordered tuples of given length over levels {0..n_levels-1}.
    """
    for tup in itertools.product(range(n_levels), repeat=length):
        yield tup


def tuple_to_counts(tup, n_levels=5):
    """
    Convert an ordered tuple (levels per hour) to a counts 5-tuple (x0..x4).
    """
    counts = [0] * n_levels
    for lv in tup:
        counts[lv] += 1
    return tuple(counts)


def write_ordered_lookup_csv_for_pair(t1, t2, caps, out_dir):
    """
    For a given (t1->t2) pair, enumerate all ordered tuples of length n=t2/t1,
    map each tuple to a big-step bucket, and write a CSV lookup:
      dt{t1}h_l1, ..., dt{t1}h_l{n}, dt{t2}h_bucket
    """
    if t2 <= t1 or t2 % t1 != 0:
        raise ValueError(f'invalid pair: {t1}->{t2}')
    if t1 not in caps or t2 not in caps:
        raise ValueError(f'caps missing for horizons in: {t1}->{t2}')

    n = t2 // t1
    cap_small = caps[t1]
    cap_target = caps[t2]

    fname = f'lookup_ordered_dt{t1}h_to_dt{t2}h.csv'
    path = os.path.join(out_dir, fname)
    cols = [f'dt{t1}h_l{i+1}' for i in range(n)] + [f'dt{t2}h_bucket']

    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)

        for tup in ordered_tuples(n, n_levels=5):
            x = tuple_to_counts(tup, n_levels=5)
            L, U = sum_interval_for_counts(x, cap_small, t1)
            bucket = decide_bucket(L, U, t2, cap_target)
            writer.writerow(list(tup) + [bucket])


def main():
    """
    Generate one ordered-lookup CSV per (t1->t2) mapping specified in PAIRS.
    """
    for t1, t2 in PAIRS:
        write_ordered_lookup_csv_for_pair(t1, t2, CAPS, OUT_DIR)


if __name__ == '__main__':
    main()
