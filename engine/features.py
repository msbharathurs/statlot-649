"""
Feature engineering for the data lake.
Produces 39 features per draw row.
"""
import numpy as np
from collections import Counter


def build_feature_row(draw_idx: int, draws: list[dict]) -> dict:
    """
    Given a list of draws (sorted ascending) and the current draw index,
    compute all 39 features for that draw using only PRIOR draws as context.
    """
    d = draws[draw_idx]
    nums = sorted([d["n1"], d["n2"], d["n3"], d["n4"], d["n5"], d["n6"]])
    s = sum(nums)

    # ── Repeat features: how many numbers appear in the last N draws ──────────
    def repeat_n(n_back):
        if draw_idx < n_back:
            return 0
        prev = set()
        for k in range(1, n_back + 1):
            p = draws[draw_idx - k]
            prev.update([p["n1"], p["n2"], p["n3"], p["n4"], p["n5"], p["n6"]])
        return len(set(nums) & prev)

    repeats = {f"repeat_{k}": repeat_n(k) for k in range(1, 11)}

    # ── Sum features ──────────────────────────────────────────────────────────
    past_sums = [sum([draws[j]["n1"], draws[j]["n2"], draws[j]["n3"],
                      draws[j]["n4"], draws[j]["n5"], draws[j]["n6"]])
                 for j in range(max(0, draw_idx - 10), draw_idx)]

    sum_delta = (s - past_sums[-1]) if past_sums else 0
    sum_ma3 = np.mean(past_sums[-3:]) if len(past_sums) >= 3 else (np.mean(past_sums) if past_sums else s)
    sum_ma5 = np.mean(past_sums[-5:]) if len(past_sums) >= 5 else (np.mean(past_sums) if past_sums else s)
    sum_ma10 = np.mean(past_sums) if past_sums else s

    # ── Structural features ───────────────────────────────────────────────────
    odd_count = sum(1 for n in nums if n % 2 != 0)
    low_count = sum(1 for n in nums if n <= 24)
    consec = sum(1 for i in range(5) if nums[i + 1] - nums[i] == 1)
    gaps = [nums[i + 1] - nums[i] for i in range(5)]
    max_gap = max(gaps)
    min_gap = min(gaps)
    avg_gap = np.mean(gaps)

    # ── Decade distribution ───────────────────────────────────────────────────
    decade_ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 49)]
    decades = [sum(1 for n in nums if lo <= n <= hi) for lo, hi in decade_ranges]
    empty_decades = sum(1 for dec in decades if dec == 0)

    # ── Rolling frequency (using prior draws only) ───────────────────────────
    def freq_in_window(window):
        if draw_idx == 0:
            return 0.0
        freq = Counter()
        for j in range(max(0, draw_idx - window), draw_idx):
            p = draws[j]
            for n in [p["n1"], p["n2"], p["n3"], p["n4"], p["n5"], p["n6"]]:
                freq[n] += 1
        return np.mean([freq.get(n, 0) for n in nums])

    avg_freq_10 = freq_in_window(10)
    avg_freq_20 = freq_in_window(20)
    avg_freq_50 = freq_in_window(50)

    # ── Hot / Cold ────────────────────────────────────────────────────────────
    if draw_idx >= 50:
        freq_50 = Counter()
        for j in range(draw_idx - 50, draw_idx):
            p = draws[j]
            for n in [p["n1"], p["n2"], p["n3"], p["n4"], p["n5"], p["n6"]]:
                freq_50[n] += 1
        sorted_nums = [n for n, _ in freq_50.most_common()]
        hot_set = set(sorted_nums[:10])
        cold_set = set(sorted_nums[-10:])
        hot_count = sum(1 for n in nums if n in hot_set)
        cold_count = sum(1 for n in nums if n in cold_set)
    else:
        hot_count = 0
        cold_count = 0

    # ── Pair strength ─────────────────────────────────────────────────────────
    if draw_idx >= 20:
        pair_freq = Counter()
        for j in range(max(0, draw_idx - 50), draw_idx):
            p = draws[j]
            pnums = sorted([p["n1"], p["n2"], p["n3"], p["n4"], p["n5"], p["n6"]])
            for a in range(6):
                for b in range(a + 1, 6):
                    pair_freq[(pnums[a], pnums[b])] += 1
        my_pairs = [(nums[a], nums[b]) for a in range(6) for b in range(a + 1, 6)]
        pair_scores = [pair_freq.get(p, 0) for p in my_pairs]
        avg_pair = np.mean(pair_scores)
        max_pair = max(pair_scores)
    else:
        avg_pair = 0.0
        max_pair = 0

    # ── Draw gap since any repeat ─────────────────────────────────────────────
    draws_since_repeat = 0
    for k in range(1, min(draw_idx + 1, 50)):
        p = draws[draw_idx - k]
        prev_set = {p["n1"], p["n2"], p["n3"], p["n4"], p["n5"], p["n6"]}
        if set(nums) & prev_set:
            break
        draws_since_repeat += 1

    nums_from_last2 = repeat_n(2)
    nums_from_last3 = repeat_n(3)

    return {
        "draw_number": d["draw_number"],
        # repeats
        **repeats,
        # sum
        "sum": s,
        "sum_delta": sum_delta,
        "sum_ma3": round(float(sum_ma3), 2),
        "sum_ma5": round(float(sum_ma5), 2),
        "sum_ma10": round(float(sum_ma10), 2),
        # structural
        "odd_count": odd_count,
        "even_count": 6 - odd_count,
        "low_count": low_count,
        "high_count": 6 - low_count,
        "consecutive_count": consec,
        "max_gap": max_gap,
        "min_gap": min_gap,
        "avg_gap": round(float(avg_gap), 2),
        # decades
        "decade_1": decades[0], "decade_2": decades[1],
        "decade_3": decades[2], "decade_4": decades[3], "decade_5": decades[4],
        "empty_decades": empty_decades,
        # rolling freq
        "avg_freq_last10": round(float(avg_freq_10), 3),
        "avg_freq_last20": round(float(avg_freq_20), 3),
        "avg_freq_last50": round(float(avg_freq_50), 3),
        # hot/cold
        "hot_count": hot_count,
        "cold_count": cold_count,
        # pair strength
        "avg_pair_freq": round(float(avg_pair), 3),
        "max_pair_freq": int(max_pair),
        # gap features
        "draws_since_any_repeat": draws_since_repeat,
        "numbers_from_last2": nums_from_last2,
        "numbers_from_last3": nums_from_last3,
    }


FEATURE_COLS = [
    "repeat_1","repeat_2","repeat_3","repeat_4","repeat_5",
    "repeat_6","repeat_7","repeat_8","repeat_9","repeat_10",
    "sum","sum_delta","sum_ma3","sum_ma5","sum_ma10",
    "odd_count","even_count","low_count","high_count",
    "consecutive_count","max_gap","min_gap","avg_gap",
    "decade_1","decade_2","decade_3","decade_4","decade_5","empty_decades",
    "avg_freq_last10","avg_freq_last20","avg_freq_last50",
    "hot_count","cold_count",
    "avg_pair_freq","max_pair_freq",
    "draws_since_any_repeat","numbers_from_last2","numbers_from_last3",
]
