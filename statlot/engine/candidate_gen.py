"""
Candidate generation engine.
Supports 6-number, 7-number, and 8-number pools.
"""
import random
import numpy as np
from collections import Counter
from itertools import combinations


def generate_candidates(
    history: list,
    pool_size: int = 6,       # 6, 7, or 8 numbers in the pool
    n_candidates: int = 200000,
    property_filters: dict = None,  # predicted properties from ML models
    weights: dict = None,
) -> list:
    """
    Generate candidate combinations using frequency + aging + pair scoring.
    pool_size=6: standard 6-number combo
    pool_size=7: generate pools of 7 numbers (C(7,6)=7 embedded combos)
    pool_size=8: generate pools of 8 numbers (C(8,6)=28 embedded combos)
    """
    if weights is None:
        weights = {"freq": 3.0, "pair": 2.0, "aging": 1.5}

    freq, pair_freq, aging = _compute_stats(history)
    last_nums = history[-1]["nums"] if history else []

    # Property filters from ML prediction
    filters = _build_filters(property_filters)

    pool = list(range(1, 50))
    w_arr = [freq.get(n, 0.1) + aging.get(n, 1) * 0.4 for n in pool]
    total_w = sum(w_arr)
    probs = [w / total_w for w in w_arr]

    seen = set()
    scored = []
    attempts = 0
    max_attempts = n_candidates * 6

    while len(scored) < n_candidates and attempts < max_attempts:
        attempts += 1
        try:
            combo = tuple(sorted(random.choices(pool, weights=probs, k=pool_size)))
        except Exception:
            combo = tuple(sorted(random.sample(pool, pool_size)))

        if len(set(combo)) < pool_size:
            continue
        if combo in seen:
            continue
        seen.add(combo)

        if not _passes_filters(combo, last_nums, filters, pool_size):
            continue

        sc = _score(combo, freq, pair_freq, aging, weights)
        scored.append((sc, combo))

    scored.sort(reverse=True)
    return [c for _, c in scored]


def _compute_stats(history):
    freq = Counter()
    pair_freq = Counter()
    n = len(history)
    for i, d in enumerate(history):
        w = 1.0 + (i / n) * 2.0
        nums = d["nums"]
        for num in nums:
            freq[num] += w
        for a in range(len(nums)):
            for b in range(a + 1, len(nums)):
                pair_freq[(nums[a], nums[b])] += w
    cur = n
    last_seen = {}
    for i, d in enumerate(history):
        for num in d["nums"]:
            last_seen[num] = i
    aging = {n: min(cur - last_seen.get(n, 0), 30) for n in range(1, 50)}
    return freq, pair_freq, aging


def _build_filters(preds: dict) -> dict:
    """Convert ML property predictions into filter ranges."""
    if not preds:
        # Default relaxed filters
        return {
            "sum_min": 70, "sum_max": 230,
            "odd_min": 1, "odd_max": 5,
            "consec_max": 4,
            "empty_dec_max": 3,
            "repeat_max": 4,
        }

    # Tighter filters based on ML predictions
    sum_bucket = preds.get("sum_bucket", 3)
    bucket_ranges = {0:(55,99), 1:(100,119), 2:(120,139), 3:(140,159),
                     4:(160,179), 5:(180,199), 6:(200,219), 7:(220,260)}
    lo, hi = bucket_ranges.get(sum_bucket, (70,230))
    # Allow ±1 bucket of slack
    lo = bucket_ranges.get(max(0, sum_bucket-1), (55,99))[0]
    hi = bucket_ranges.get(min(7, sum_bucket+1), (220,260))[1]

    pred_odd = preds.get("odd_count", 3)
    pred_empty = preds.get("empty_decades", 1)

    return {
        "sum_min": lo, "sum_max": hi,
        "odd_min": max(1, pred_odd - 1),
        "odd_max": min(5, pred_odd + 1),
        "consec_max": 3,
        "empty_dec_max": min(3, pred_empty + 1),
        "repeat_max": 4,
    }


def _passes_filters(combo, last_nums, filters, pool_size) -> bool:
    nums = sorted(combo)
    # For 7/8 pools, check inner combos are structurally valid
    check_nums = nums[:6] if pool_size > 6 else nums

    s = sum(check_nums)
    odd = sum(1 for n in check_nums if n % 2 != 0)
    consec = sum(1 for i in range(len(check_nums)-1) if check_nums[i+1]-check_nums[i]==1)
    dec = [sum(1 for n in check_nums if lo<=n<=hi) for lo,hi in [(1,10),(11,20),(21,30),(31,40),(41,49)]]
    empty_dec = sum(1 for d in dec if d == 0)
    repeat = len(set(check_nums) & set(last_nums))

    if not (filters["sum_min"] <= s <= filters["sum_max"]): return False
    if not (filters["odd_min"] <= odd <= filters["odd_max"]): return False
    if consec > filters["consec_max"]: return False
    if empty_dec > filters["empty_dec_max"]: return False
    if repeat > filters["repeat_max"]: return False
    return True


def _score(combo, freq, pair_freq, aging, weights) -> float:
    nums = sorted(combo)
    f = sum(freq.get(n, 0.1) for n in nums)
    p = sum(pair_freq.get((nums[a], nums[b]), 0)
            for a in range(len(nums)) for b in range(a+1, len(nums)))
    a = sum(aging.get(n, 1) for n in nums)
    return weights["freq"]*f + weights["pair"]*p + weights["aging"]*a


def expand_pool_to_combos(pool: tuple, size: int = 6) -> list:
    """Expand a 7 or 8 number pool into all C(pool,6) combinations."""
    return list(combinations(pool, size))
