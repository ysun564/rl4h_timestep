import csv
import itertools
import os


"""
Expected-Overlap mapping with fixed upper bounds.

Edit PAIRS to choose (t1 -> t2) mappings (t2 % t1 == 0).
Edit CAPS to set Level-4 caps per horizon (hours).
CSV columns:
  dt{t1}h_a0_count, ..., dt{t1}h_a4_count, dt{t2}h
"""

# ==== EDIT THESE ====
PAIRS = [(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (4, 8)]
CAPS = {1: 1942.0, 2: 2750.0, 4: 4591.0, 8: 7669.0}
OUT_DIR = rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\e_fair_comparison\data\mapping_up\raw'
# ====================


def thresholds(timestep):
    """
    Return bucket edges (in mL) for a target horizon (hours):
      t1 = 125*Δt, t2 = 250*Δt, t3 = 500*Δt.
    """
    return 125 * timestep, 250 * timestep, 500 * timestep


def enumerate_counts(n):
    """
    Yield all 5-tuples (x0..x4) of nonnegative integers summing to n.
    """
    for x0 in range(n + 1):
        for x1 in range(n - x0 + 1):
            for x2 in range(n - x0 - x1 + 1):
                for x3 in range(n - x0 - x1 - x2 + 1):
                    x4 = n - x0 - x1 - x2 - x3
                    yield (x0, x1, x2, x3, x4)


def sum_interval_for_counts(x, cap_small, timestep_small):
    """
    Compute [L, U] (mL) for counts x under the small-step horizon Δt_small,
    using fixed-cap Expected-Overlap formulation:

      L = 125*Δt1 * x2 + 250*Δt1 * x3 + 500*Δt1 * x4
      U = 125*Δt1 * x1 + 250*Δt1 * x2 + 500*Δt1 * x3 + cap_small * x4
    NOTE: x is the number of each action was executed. Each time it passes one combination.
    """
    x0, x1, x2, x3, x4 = x
    u1 = 125 * timestep_small
    u2 = 250 * timestep_small
    u3 = 500 * timestep_small

    L = u1 * x2 + u2 * x3 + u3 * x4
    U = u1 * x1 + u2 * x2 + u3 * x3 + cap_small * x4
    return L, U


def overlap(a, b, c, d):
    """
    Length of overlap between intervals [a,b] and [c,d].
    """
    return max(0.0, min(b, d) - max(a, c))


def decide_bucket(L, U, timestep_target, cap_target):
    """
    Expected-Overlap with midpoint tie-break for the target horizon.
    Returns one of 'A0','A1','A2','A3','A4'.
    NOTE: cap_target represents the upper bound.
    """
    if L == 0 and U == 0:
        return 'A0'

    # t1, t2, t3, cap_target represents action space boundary.
    t1, t2, t3 = thresholds(timestep_target)
    b1 = (0.0, float(t1))
    b2 = (float(t1), float(t2))
    b3 = (float(t2), float(t3))
    b4 = (float(t3), float(cap_target))

    scores = {
        'A1': overlap(L, U, *b1),
        'A2': overlap(L, U, *b2),
        'A3': overlap(L, U, *b3),
        'A4': overlap(L, U, *b4),
    }
    # Best value.
    best_val = max(scores.values())
    best = [k for k, v in scores.items() if v == best_val]

    # If only have one action, choose the action.
    if len(best) == 1:
        return best[0]

    # If there're same, use the position of midpoint as fianl position.
    
    m = 0.5 * (L + U)
    if m <= 0:
        return 'A0'
    if m < t1:
        return 'A1'
    if m < t2:
        return 'A2'
    if m < t3:
        return 'A3'
    return 'A4'


def write_csv_for_pair(t1, t2, caps, out_dir):
    """
    Enumerate all compositions for n = t2/t1, map with Expected-Overlap
    (fixed caps), and write a CSV:
      dt{t1}h_a0_count, ..., dt{t1}h_a4_count, dt{t2}h
    """
    if t2 <= t1 or t2 % t1 != 0:
        raise ValueError(f'invalid pair: {t1}->{t2}')
    if t1 not in caps or t2 not in caps:
        raise ValueError(f'caps missing for horizons in: {t1}->{t2}')

    n = t2 // t1
    cap_small = caps[t1]
    # Use mapping to decide each upperbound.
    cap_target = caps[t2]

    fname = f'map_dt{t1}h_to_dt{t2}h.csv'
    path = os.path.join(out_dir, fname)
    cols = [
        f'dt{t1}h_a0_count',
        f'dt{t1}h_a1_count',
        f'dt{t1}h_a2_count',
        f'dt{t1}h_a3_count',
        f'dt{t1}h_a4_count',
        f'dt{t2}h',
    ]

    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for x in enumerate_counts(n):
            L, U = sum_interval_for_counts(x, cap_small, t1)
            bucket = decide_bucket(L, U, t2, cap_target)
            writer.writerow(list(x) + [bucket])


def main():
    """
    Generate one CSV per (t1->t2) mapping specified in PAIRS.
    """
    for t1, t2 in PAIRS:
        write_csv_for_pair(t1, t2, CAPS, OUT_DIR)


if __name__ == '__main__':
    main()
