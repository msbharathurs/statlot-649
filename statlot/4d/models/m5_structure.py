"""
M5 — Structure Distribution Model
Profiles the structural DNA of winning numbers per tier:
  - number_class (unique/single_pair/double_pair/triple/quad)
  - sum_band (small/medium/large)
  - parity_pattern (OOEE, OEOE, etc.)
  - hl_pattern (high/low split)
  - pair_type (AABB, ABAB, etc.)
Scores candidates by how well they match the historical structural profile of each tier.
"""

import duckdb
import numpy as np
import pickle
import os
from collections import defaultdict

DB_PATH   = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
MODEL_DIR = os.path.expanduser("~/statlot-649/models")
os.makedirs(MODEL_DIR, exist_ok=True)

ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]
PRIMES = {2, 3, 5, 7}


def analyze_candidate(num: str) -> dict:
    """Recompute structural features for a candidate number."""
    from collections import Counter
    d = [int(c) for c in num]
    freq = Counter(d)
    counts = sorted(freq.values(), reverse=True)
    unique_cnt = len(freq)

    if unique_cnt == 1:
        pair_type = "AAAA"; number_class = "quad"
    elif unique_cnt == 2:
        if counts == [3, 1]:
            pair_type = "AAAB"; number_class = "triple"
        else:
            if d[0]==d[1] and d[2]==d[3]: pair_type = "AABB"
            elif d[0]==d[2] and d[1]==d[3]: pair_type = "ABAB"
            elif d[0]==d[3] and d[1]==d[2]: pair_type = "ABBA"
            else: pair_type = "AABB_other"
            number_class = "double_pair"
    elif unique_cnt == 3:
        pair_type = "PAIR_OTHER"; number_class = "single_pair"
    else:
        pair_type = "ABCD"; number_class = "unique"

    digit_sum = sum(d)
    sum_band  = "small" if digit_sum <= 13 else ("medium" if digit_sum <= 22 else "large")
    odd_count = sum(x % 2 != 0 for x in d)
    parity_pat = "".join("O" if x % 2 != 0 else "E" for x in d)
    low_count  = sum(x <= 4 for x in d)
    hl_pat     = "".join("L" if x <= 4 else "H" for x in d)

    return {
        "number_class": number_class,
        "pair_type":    pair_type,
        "sum_band":     sum_band,
        "parity_pattern": parity_pat,
        "hl_pattern":   hl_pat,
        "digit_sum":    digit_sum,
        "odd_count":    odd_count,
        "low_count":    low_count,
    }


def train_m5(train_up_to_draw: int) -> dict:
    """
    Compute structural frequency profiles per tier.
    Returns nested dict: tier -> feature -> value -> probability
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    rows = con.execute(f"""
        SELECT tier, number_class, pair_type, sum_band, parity_pattern,
               hl_pattern, digit_sum, odd_count, low_count
        FROM draw_numbers
        WHERE draw_id <= {train_up_to_draw}
    """).fetchall()
    con.close()

    # tier groups
    tier_map = {}
    for row in rows:
        (tier, number_class, pair_type, sum_band, parity_pattern,
         hl_pattern, digit_sum, odd_count, low_count) = row
        # Map to tier groups
        if tier == "1st":
            tg = "1st"
        elif tier in ("2nd", "3rd"):
            tg = "top3"
        elif "starter" in tier:
            tg = "starter"
        else:
            tg = "consolation"
        for t in [tg, "all"]:
            if t not in tier_map:
                tier_map[t] = defaultdict(lambda: defaultdict(int))
            tier_map[t]["number_class"][number_class] += 1
            tier_map[t]["pair_type"][pair_type]       += 1
            tier_map[t]["sum_band"][sum_band]          += 1
            tier_map[t]["parity_pattern"][parity_pattern] += 1
            tier_map[t]["hl_pattern"][hl_pattern]     += 1
            tier_map[t]["digit_sum"][digit_sum]        += 1

    def normalize(d):
        total = sum(d.values())
        return {k: v/total for k, v in d.items()} if total else {}

    profiles = {}
    for tg, features in tier_map.items():
        profiles[tg] = {feat: normalize(counts) for feat, counts in features.items()}

    return profiles


def score_m5(profiles: dict, candidate: str, tier_group: str = "1st") -> float:
    """Score a candidate based on structural match to historical tier profile."""
    feats   = analyze_candidate(candidate)
    profile = profiles.get(tier_group, profiles.get("all", {}))
    if not profile:
        return 0.0

    score = 1.0
    for feat in ["number_class", "pair_type", "sum_band", "parity_pattern", "hl_pattern"]:
        val  = feats.get(feat)
        prob = profile.get(feat, {}).get(val, 1e-4)
        score *= prob

    return score ** (1.0 / 5)   # geometric mean


def predict_m5(profiles: dict, top_n: int = 100, tier_group: str = "1st") -> tuple:
    scores = {n: score_m5(profiles, n, tier_group=tier_group) for n in ALL_NUMBERS}
    max_s  = max(scores.values()) if scores else 1.0
    scores = {k: v / max_s for k, v in scores.items()}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in ranked[:top_n]], scores


def save_m5(profiles: dict, suffix: str = ""):
    path = os.path.join(MODEL_DIR, f"m5_structure{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(profiles, f)
    print(f"M5 saved → {path}")


def load_m5(suffix: str = "") -> dict:
    path = os.path.join(MODEL_DIR, f"m5_structure{suffix}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
    con.close()

    print(f"Training M5 Structure on draws 1–{max_draw} ...")
    profiles = train_m5(max_draw)

    print("\n-- 1st prize structural profile --")
    for feat, dist in profiles.get("1st", {}).items():
        top = sorted(dist.items(), key=lambda x: -x[1])[:5]
        print(f"  {feat}: {top}")

    top10, _ = predict_m5(profiles, top_n=10, tier_group="1st")
    print(f"\nTop-10 structurally likely (1st): {top10}")
    save_m5(profiles)
    print("M5 done.")
