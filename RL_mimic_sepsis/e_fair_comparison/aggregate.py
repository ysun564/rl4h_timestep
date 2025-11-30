"""
aggregate.py

Aggregate n = (t2 // t1) small-step 25-dim joint action distributions into one
t2-step 25-dim distribution using:
  - IV: Expected-Overlap mapping with fixed caps (train-frozen)
  - Vaso: max over the n small steps
  - Joint action encoding: a = iv_bucket*5 + vaso_level

Inputs are n vectors p25_t (len=25) that represent per-hour joint probabilities
over (iv_level in {0..4}, vaso_level in {0..4}). For each hour:
  P_t[iv, vaso] = p25_t[iv*5 + vaso]

The result is a 25-dim distribution over (iv_bucket in {0..4}, vaso_max in {0..4})
at the larger horizon t2.
"""
import os
import csv
from functools import lru_cache
from math import factorial

import numpy as np

from RL_mimic_sepsis.e_fair_comparison.mapping_ordered import ordered_tuples, tuple_to_counts
from RL_mimic_sepsis.e_fair_comparison.mapping_raw import CAPS, thresholds, sum_interval_for_counts, decide_bucket, enumerate_counts


def _bucket_label_to_index(label):
    """
    Convert 'A0'..'A4' to integer 0..4.
    """
    return int(label[1])

@lru_cache(maxsize=None)
def _load_ordered_lookup_csv(t1, t2, lookup_dir):
    """
    Load the precomputed ordered-tuple -> bucket CSV produced by mapping_ordered.py.

    Returns:
      iv_tuples: (K, n) int array where K = 5**n, n = t2//t1
      bucket_idx: (K,) int array in {0..4}
    """
    fname = (rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis'
             rf'\e_fair_comparison\data\mapping_up\ordered'
             rf'\lookup_ordered_dt{t1}h_to_dt{t2}h.csv')
    path = os.path.join(lookup_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    tuples = []
    buckets = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        n = len(header) - 1
        for row in reader:
            iv_seq = tuple(int(x) for x in row[:n])
            b = _bucket_label_to_index(row[-1])
            tuples.append(iv_seq)
            buckets.append(b)

    iv_tuples = np.array(tuples, dtype=np.uint8)     
    bucket_idx = np.array(buckets, dtype=np.uint8) 
    return iv_tuples, bucket_idx


def _precompute_cond_cdf(cond_list):
    """
    Precompute CDF over vaso levels for each (t, iv).
    cond_list[t] is 5x5 with rows p(v|iv); returns cdf[t, iv, m] = sum_{v<=m} p(v|iv).
    """
    n = len(cond_list)
    cdf = np.zeros((n, 5, 5), dtype=float)
    for t in range(n):
        cdf[t] = np.cumsum(cond_list[t], axis=1)
    return cdf


def _vectorized_weights(piv_list, iv_tuples):
    """
    Compute sequence weights w(seq) = ∏_t p_iv_t[iv_t] for all sequences in a vectorized way.
    """
    K, n = iv_tuples.shape
    w = np.ones(K, dtype=float)
    for t in range(n):
        w *= piv_list[t][iv_tuples[:, t]]
    return w


def _vectorized_vaso_max_pmf(cond_cdf, iv_tuples):
    """
    For all sequences at once, compute PMF over V=max(v_1..v_n).
    Returns pmf_all with shape (K, 5).
    """
    K, n = iv_tuples.shape
    cdf_all = np.zeros((K, 5), dtype=float)

    for m in range(5):
        prod = np.ones(K, dtype=float)
        for t in range(n):
            prod *= cond_cdf[t][iv_tuples[:, t], m]
        cdf_all[:, m] = prod

    pmf_all = np.empty_like(cdf_all)
    pmf_all[:, 0] = cdf_all[:, 0]
    for m in range(1, 5):
        pmf_all[:, m] = np.maximum(0.0, cdf_all[:, m] - cdf_all[:, m - 1])

    row_sum = pmf_all.sum(axis=1, keepdims=True)
    nonzero = row_sum.squeeze() > 0
    pmf_all[nonzero] /= row_sum[nonzero]
    return pmf_all


def _split_joint_25_to_matrix(p25):
    """Reshape a 25-dim vector into a 5x5 matrix P[iv, vaso], normalized to sum 1.
    """
    P = np.reshape(np.asarray(p25, dtype=float), (5, 5))
    s = P.sum()
    if s > 0:
        P = P / s
    return P


def _per_hour_iv_marginal_and_conditionals(P):
    """
    From a 5x5 P[iv, vaso], return:
      p_iv: shape (5,)
      p_v_given_iv: shape (5,5), each row sums to 1 for rows with p_iv>0.
    """
    p_iv = P.sum(axis=1)
    p_v_given_iv = np.zeros_like(P)
    nz = p_iv > 0
    p_v_given_iv[nz] = P[nz] / p_iv[nz, None]
    return p_iv, p_v_given_iv


def _iv_bucket_for_tuple(iv_tuple, t1, t2, cap_small, cap_target):
    """
    Map an ordered IV tuple to its IV bucket index at horizon t2 (0..4).
    """
    counts = tuple_to_counts(iv_tuple, n_levels=5)
    L, U = sum_interval_for_counts(counts, cap_small, t1)
    label = decide_bucket(L, U, t2, cap_target)  # 'A0'..'A4'
    return _bucket_label_to_index(label)


def _vaso_max_pmf_given_iv_sequence(cond_list, iv_seq):
    """
    Compute PMF of V = max(v_1..v_n) given IV sequence iv_seq and per-hour
    conditionals cond_list (each a 5x5 matrix of p(v|iv)).

    Returns a length-5 PMF over m in {0..4}.
    """
    n = len(iv_seq)
    cdf = np.zeros(5, dtype=float)
    for m in range(5):
        prod = 1.0
        for t in range(n):
            i_t = iv_seq[t]
            prod *= cond_list[t][i_t, :m + 1].sum()
        cdf[m] = prod
    pmf = np.empty(5, dtype=float)
    pmf[0] = cdf[0]
    for m in range(1, 5):
        pmf[m] = max(0.0, cdf[m] - cdf[m - 1])
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


OUT_DIR = rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\e_fair_comparison\data\mapping_up\ordered'
def aggregate_t1_to_t2_25d(p25_list, t1, t2, caps=None, lookup_dir=OUT_DIR):
    """
    Aggregate n=t2//t1 small-step 25-d joint distributions into one t2-step 25-d distribution.

    Uses precomputed ordered-lookup CSV for IV bucket mapping when available,
    and falls back to on-the-fly computation otherwise.
    """
    if caps is None:
        caps = CAPS
    if t2 <= t1 or (t2 % t1) != 0:
        raise ValueError('t2 must be a positive multiple of t1 and greater than t1')

    n = t2 // t1
    if len(p25_list) != n:
        raise ValueError('p25_list length must equal t2//t1')
    if t1 not in caps or t2 not in caps:
        raise ValueError('caps must provide entries for both t1 and t2')

    # 1) per-hour joint -> P, then p_iv and p(v|iv)
    P_list = [_split_joint_25_to_matrix(p) for p in p25_list]
    piv_list, cond_list = zip(*[_per_hour_iv_marginal_and_conditionals(P) for P in P_list])

    # 2) load or fallback the IV ordered lookup
    try:
        iv_tuples, iv_bucket_idx = _load_ordered_lookup_csv(t1, t2, lookup_dir)
        if iv_tuples.shape[1] != n:
            raise ValueError('lookup CSV n mismatch with t2//t1')
    except Exception:
        from RL_mimic_sepsis.e_fair_comparison.mapping_ordered import ordered_tuples, tuple_to_counts
        from RL_mimic_sepsis.e_fair_comparison.mapping_raw import sum_interval_for_counts, decide_bucket

        cap_small = float(caps[t1])
        cap_target = float(caps[t2])

        tuples = []
        buckets = []
        for tup in ordered_tuples(n, n_levels=5):
            counts = tuple_to_counts(tup, n_levels=5)
            L, U = sum_interval_for_counts(counts, cap_small, t1)
            label = decide_bucket(L, U, t2, cap_target)
            tuples.append(tup)
            buckets.append(_bucket_label_to_index(label))
        iv_tuples = np.array(list(tuples), dtype=np.uint8)
        iv_bucket_idx = np.array(list(buckets), dtype=np.uint8)

    # 3) vectorized weights over IV sequences
    w = _vectorized_weights(piv_list, iv_tuples)  # (K,)
    nonzero_mask = w > 0
    if not np.any(nonzero_mask):
        pmf25 = np.zeros(25, dtype=float)
        return pmf25

    iv_tuples_nz = iv_tuples[nonzero_mask]
    buckets_nz = iv_bucket_idx[nonzero_mask]
    w_nz = w[nonzero_mask]

    # 4) vectorized vaso-max PMF for all sequences
    cond_cdf = _precompute_cond_cdf(cond_list)          # (n, 5, 5)
    pmf_vmax_all = _vectorized_vaso_max_pmf(cond_cdf, iv_tuples_nz)  # (K_nz, 5)

    # 5) accumulate into 25-d joint over (iv_bucket, vaso_max)
    out = np.zeros((5, 5), dtype=float)  # [iv_bucket, vaso_level]

    for b in range(5):
        mask_b = (buckets_nz == b)
        if not np.any(mask_b):
            continue
        contrib = (w_nz[mask_b][:, None] * pmf_vmax_all[mask_b, :]).sum(axis=0)  # (5,)
        out[b, :] = contrib

    pmf25 = out.reshape(-1)  # (25,)
    s = pmf25.sum()
    if s > 0:
        pmf25 /= s
    return pmf25


def aggregate_identical_t1_to_t2(p25_small, t1, t2, caps=CAPS, eps=1e-12):
    """Efficiently aggregate n identical fine-step distributions into one coarse distribution."""
    n = t2 // t1
    # 5x5 joint matrix for p25_small
    P = p25_small.reshape(5,5) / max(p25_small.sum(), eps)
    p_iv = P.sum(axis=1)                           # marginal IV probabilities (length 5)
    # Precompute vaso CDF for each iv level
    F = np.cumsum((P.T / np.maximum(p_iv, eps)).T, axis=1)  # shape (5,5), F[k,m] = P(v<=m | iv=k)
    out = np.zeros((5,5))  # joint coarse distribution matrix [iv_bucket, vaso_max]
    # Iterate over IV count combinations (c0..c4 summing to n)
    for c in enumerate_counts(n):                  # yields all 5-tuples summing to n
        c = np.array(c)
        # Multinomial probability for this IV count tuple
        prob_counts = factorial(n)
        for k in range(5):
            prob_counts *= (p_iv[k] ** c[k]) / factorial(c[k])
        if prob_counts < eps: 
            continue  # skip negligible combinations
        # Determine coarse IV bucket from counts
        L, U = sum_interval_for_counts(tuple(c), caps[t1], t1)
        bucket_label = decide_bucket(L, U, t2, caps[t2])    # e.g. 'A3'
        b = int(bucket_label[1])  # bucket index 0-4
        # Probability distribution of max vaso given this count tuple
        # P(max <= m | c) = ∏_k [F[k, m]]^{c_k}
        # Compute for m = 0..4
        pmf_max = np.zeros(5)
        prev_cdf = 0.0
        for m in range(5):
            cdf_m = 1.0
            for k in range(5):
                cdf_m *= (F[k, m] ** c[k])
            pmf_max[m] = max(0.0, cdf_m - prev_cdf)  # P(max = m)
            prev_cdf = cdf_m
        # Accumulate joint probability into the [b, vaso_max] cell
        out[b, :] += prob_counts * pmf_max
    # Normalize and return as 25-d vector
    pmf25 = out.ravel()
    return pmf25 / pmf25.sum()