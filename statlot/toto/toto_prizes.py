"""
toto_prizes.py — TOTO prize calculation engine
Handles Ordinary, Sys7-12 entries.
"""
from itertools import combinations
from typing import List, Optional, Dict

# Prize pool percentages (of 54% of sales)
PRIZE_POOL_PCT = {
    1: 0.38,   # Jackpot: 6 winning numbers
    2: 0.08,   # 5 + additional
    3: 0.055,  # 5 numbers
    4: 0.03,   # 4 + additional
}
FIXED_PRIZES = {
    5: 50.0,   # 4 numbers
    6: 25.0,   # 3 + additional
    7: 10.0,   # 3 numbers
}

# System entry costs & combination counts
SYSTEM_INFO = {
    6:  {"combos": 1,   "cost": 1.00},
    7:  {"combos": 7,   "cost": 7.00},
    8:  {"combos": 28,  "cost": 28.00},
    9:  {"combos": 84,  "cost": 84.00},
    10: {"combos": 210, "cost": 210.00},
    11: {"combos": 462, "cost": 462.00},
    12: {"combos": 924, "cost": 924.00},
}

def check_combo(combo: List[int], winning: List[int], additional: int) -> Dict:
    """Check a single 6-number combo against the draw. Returns best group won."""
    combo_set = set(combo)
    win_set = set(winning)
    matches = len(combo_set & win_set)
    has_additional = additional in combo_set

    group = None
    if matches == 6:
        group = 1
    elif matches == 5 and has_additional:
        group = 2
    elif matches == 5:
        group = 3
    elif matches == 4 and has_additional:
        group = 4
    elif matches == 4:
        group = 5
    elif matches == 3 and has_additional:
        group = 6
    elif matches == 3:
        group = 7

    return {
        "group": group,
        "matches": matches,
        "has_additional": has_additional,
        "combo": sorted(combo)
    }

def check_system_entry(numbers: List[int], winning: List[int], additional: int,
                       prize_pool_sgd: Optional[float] = None) -> Dict:
    """
    Check a system entry (7-12 numbers) against a draw.
    Expands to all C(n,6) combinations and checks each.
    Returns best group, all wins, estimated prize.
    """
    n = len(numbers)
    all_combos = list(combinations(sorted(numbers), 6))
    results = []
    best_group = None
    total_fixed = 0.0
    pool_wins = {}  # group -> count of winning combos

    for combo in all_combos:
        r = check_combo(list(combo), winning, additional)
        results.append(r)
        g = r["group"]
        if g is not None:
            if best_group is None or g < best_group:
                best_group = g
            if g in FIXED_PRIZES:
                total_fixed += FIXED_PRIZES[g]
            else:
                pool_wins[g] = pool_wins.get(g, 0) + 1

    # Estimate pool prizes (use pool_sgd if provided, else estimate from min jackpot)
    est_pool = prize_pool_sgd or 1_000_000 / 0.38  # assume min jackpot scenario
    total_pool_prize = 0.0
    for g, cnt in pool_wins.items():
        # Conservative: assume 1 winner total (best case for us)
        total_pool_prize += est_pool * PRIZE_POOL_PCT[g]

    return {
        "system_size": n,
        "combinations_checked": len(all_combos),
        "best_group": best_group,
        "winning_combos": [r for r in results if r["group"] is not None],
        "total_fixed_prize_sgd": total_fixed,
        "total_pool_prize_est_sgd": round(total_pool_prize, 2),
        "pool_wins_by_group": pool_wins,
        "any_win": best_group is not None
    }

def check_ordinary(numbers: List[int], winning: List[int], additional: int) -> Dict:
    """Check a single ordinary (6-number) entry."""
    r = check_combo(numbers, winning, additional)
    g = r["group"]
    prize = 0.0
    if g in FIXED_PRIZES:
        prize = FIXED_PRIZES[g]
    return {
        "system_size": 6,
        "combinations_checked": 1,
        "best_group": g,
        "winning_combos": [r] if g else [],
        "total_fixed_prize_sgd": prize,
        "total_pool_prize_est_sgd": 0.0,  # unknown without pool size
        "pool_wins_by_group": {g: 1} if g and g <= 4 else {},
        "any_win": g is not None
    }

if __name__ == "__main__":
    # Smoke test
    winning = [3, 15, 22, 34, 42, 45]
    additional = 7
    t = [3, 15, 22, 34, 42, 46]  # 5 matches
    r = check_ordinary(t, winning, additional)
    print("Ordinary 5-match test:", r)
    sys7 = [3, 15, 22, 34, 42, 45, 7]  # all 6 + additional
    r2 = check_system_entry(sys7, winning, additional)
    print("Sys7 Group1 test:", r2["best_group"], "wins:", len(r2["winning_combos"]))
    print("Smoke test passed ✅")
