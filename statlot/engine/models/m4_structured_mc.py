"""
m4_structured_mc.py — Structure-Constrained Monte Carlo for StatLot 649
Replaces blind M4 Monte Carlo (currently top weight 0.185 by default due to noise).

Key upgrade from 4D work: MC sampling is constrained to structurally valid combos only.
Rules enforced during sampling (not post-filter):
  1. Decade balance: at least 1 number from each of 3+ different decades
  2. Gap distribution: no more than 2 consecutive numbers
  3. Odd/even balance: 2–4 odd numbers
  4. Dual grid: must pass both gridA and gridB rules
  5. Sum range: 100–200 (historical sweet spot)

Then combos are scored by a frequency-weighted acceptance criterion (importance sampling).
"""
import numpy as np
from collections import defaultdict
import random


DECADES = [(1,9),(10,18),(19,27),(28,36),(37,45),(46,49)]

def _in_decade(n, lo, hi):
    return lo <= n <= hi

def _valid_structure(combo):
    s = sorted(combo)
    # Sum range
    if not (100 <= sum(s) <= 200):
        return False
    # Odd/even
    odd = sum(1 for n in s if n % 2 != 0)
    if not (2 <= odd <= 4):
        return False
    # Decades covered
    decades_hit = sum(1 for lo,hi in DECADES if any(_in_decade(n,lo,hi) for n in s))
    if decades_hit < 3:
        return False
    # Consecutive pairs
    gaps = [s[i+1]-s[i] for i in range(len(s)-1)]
    consec = sum(1 for g in gaps if g == 1)
    if consec > 2:
        return False
    return True

def _gridA_ok(combo):
    rows = set((n-1)//9+1 for n in combo)
    cols = set((n-1)%9+1 for n in combo)
    return 3 <= len(rows) <= 5 and 3 <= len(cols) <= 6

def _gridB_ok(combo):
    rows = set((n-1)//7+1 for n in combo)
    return 3 <= len(rows) <= 6


def train_m4_structured_mc(
    history: list[list[int]],
    n_numbers: int = 49,
    n_samples: int = 50000,
    lookback: int = 100,
    seed: int = 42
) -> dict:
    """
    Sample n_samples structurally valid combos, score each by
    frequency-weighted acceptance probability.
    Returns {combo_tuple: score} for top candidates.
    """
    rng = random.Random(seed)

    # Build frequency weights from recent history
    recent = history[-lookback:] if len(history) >= lookback else history
    freq = defaultdict(int)
    for draw in recent:
        for n in draw:
            freq[n] += 1

    total = sum(freq.values()) or 1
    # Probability weights — blend freq with uniform (alpha=0.3 smoothing)
    alpha = 0.3
    uniform_p = 1.0 / n_numbers
    weights = {}
    for n in range(1, n_numbers + 1):
        freq_p = freq[n] / total
        weights[n] = (1 - alpha) * freq_p + alpha * uniform_p

    numbers = list(range(1, n_numbers + 1))
    w_arr = np.array([weights[n] for n in numbers])
    w_arr /= w_arr.sum()

    combo_scores = defaultdict(float)
    accepted = 0
    attempts = 0
    max_attempts = n_samples * 20

    while accepted < n_samples and attempts < max_attempts:
        attempts += 1
        # Sample 6 numbers without replacement, frequency-biased
        chosen = np.random.choice(numbers, size=6, replace=False, p=w_arr).tolist()
        chosen_s = tuple(sorted(chosen))

        if not _valid_structure(chosen_s):
            continue
        if not _gridA_ok(chosen_s):
            continue
        if not _gridB_ok(chosen_s):
            continue

        # Acceptance weight = product of individual freq weights
        score = float(np.prod([weights[n] for n in chosen_s]))
        combo_scores[chosen_s] += score
        accepted += 1

    # Normalise scores
    if combo_scores:
        max_s = max(combo_scores.values())
        combo_scores = {k: v/max_s for k,v in combo_scores.items()}

    return {"combo_scores": dict(combo_scores), "weights": weights, "accepted": accepted}


def score_m4_structured_mc(model: dict, combo: list[int]) -> float:
    """
    Score a combo against the MC model.
    Exact match gets the sampled score; unsampled combos get a
    frequency-weighted fallback.
    """
    key = tuple(sorted(combo))
    cs = model.get("combo_scores", {})

    if key in cs:
        return float(cs[key])

    # Fallback: product of individual weights, normalised
    weights = model.get("weights", {})
    if not weights:
        return 0.5
    score = float(np.prod([weights.get(n, 1e-6) for n in combo]))
    # Rough normalisation — sampled combos average ~(mean_w)^6
    mean_w = float(np.mean(list(weights.values())))
    ref = mean_w ** 6
    normalised = min(score / max(ref, 1e-12), 1.0)
    return normalised
