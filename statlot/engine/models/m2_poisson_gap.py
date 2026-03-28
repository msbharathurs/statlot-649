"""
m2_poisson_gap.py — Poisson Gap / Overdue Scoring for StatLot 649
Replaces dead EV/Kelly (M2, avg weight 0.048).

Core idea: in a fair lottery, each number appears ~every 49/6 ≈ 8.17 draws on average.
But draws are not uniform — the empirical gap distribution per number is the real signal.
A number that typically appears every 7 draws but hasn't appeared in 25 draws is MORE
due than a number with typical gap 20 that hasn't appeared in 25 draws.

We model per-number gap distributions empirically and score "how many standard deviations
overdue" each number is. Combo score = average z-score across its 6 numbers.
"""
import numpy as np
from collections import defaultdict


def train_m2_poisson(history: list[list[int]], n_numbers: int = 49) -> dict:
    """
    Compute per-number gap statistics from full draw history.
    Returns dict: number → {mean_gap, std_gap, draws_since, z_score}
    """
    N = len(history)
    last_seen = {}      # number → draw index of last appearance
    gaps = defaultdict(list)  # number → list of gap lengths

    for i, draw in enumerate(history):
        for num in draw:
            if num in last_seen:
                gaps[num].append(i - last_seen[num])
            last_seen[num] = i

    model = {}
    for num in range(1, n_numbers + 1):
        g = gaps[num]
        if len(g) >= 5:
            mean_gap = float(np.mean(g))
            std_gap  = float(np.std(g)) if np.std(g) > 0 else 1.0
        else:
            # Fallback: theoretical uniform gap
            mean_gap = N / max(1, len(g) + 1)
            std_gap  = mean_gap * 0.5

        # Draws since last appearance
        draws_since = (N - 1 - last_seen[num]) if num in last_seen else N

        # Z-score: how many SDs overdue is this number?
        z = (draws_since - mean_gap) / std_gap

        model[num] = {
            "mean_gap":   mean_gap,
            "std_gap":    std_gap,
            "draws_since": draws_since,
            "z_score":    float(z),
        }

    return model


def score_m2_poisson(model: dict, combo: list[int]) -> float:
    """
    Score a combo by average Poisson overdue z-score.
    High score = numbers are collectively overdue.
    Output bounded to [0, 1] via sigmoid.
    """
    if not model:
        return 0.0

    z_scores = [model[n]["z_score"] for n in combo if n in model]
    if not z_scores:
        return 0.0

    avg_z = float(np.mean(z_scores))
    # Sigmoid — z=0 → 0.5 (neutral), z=+2 → ~0.88 (overdue), z=-2 → ~0.12 (fresh)
    return float(1.0 / (1.0 + np.exp(-avg_z * 0.5)))
