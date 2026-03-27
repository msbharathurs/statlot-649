"""
M6 — Tier-Specific Bias Model
Discovers which numbers / ibox families / digit patterns are
statistically over-represented in EACH prize tier vs random baseline.
Uses chi-square style lift scoring.
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
TIERS_OF_INTEREST = ["1st", "2nd", "3rd", "starter", "consolation"]


def train_m6(train_up_to_draw: int) -> dict:
    """
    Compute lift scores per number per tier.
    lift = (observed_count_in_tier / total_in_tier) / (overall_count / total_all)
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    rows = con.execute(f"""
        SELECT number, tier
        FROM draw_numbers
        WHERE draw_id <= {train_up_to_draw}
    """).fetchall()
    con.close()

    # Bucket tiers
    def bucket(tier):
        if tier == "1st": return "1st"
        if tier in ("2nd","3rd"): return "top3_excl1st"
        if "starter" in tier: return "starter"
        return "consolation"

    overall = defaultdict(int)   # number -> total appearances
    tier_cnt = defaultdict(lambda: defaultdict(int))  # tier_bucket -> number -> count
    tier_total = defaultdict(int)  # tier_bucket -> total numbers
    total_all  = 0

    for number, tier in rows:
        b = bucket(tier)
        overall[number]     += 1
        tier_cnt[b][number] += 1
        tier_total[b]       += 1
        total_all           += 1

    # Compute lift per number per tier
    lift_scores = defaultdict(dict)
    for b in TIERS_OF_INTEREST + ["top3_excl1st", "starter", "consolation"]:
        if b not in tier_total:
            continue
        t_total = tier_total[b]
        for num in ALL_NUMBERS:
            t_obs      = tier_cnt[b].get(num, 0)
            overall_obs = overall.get(num, 0)
            # Expected = overall_rate * tier_total
            expected = (overall_obs / total_all) * t_total if total_all > 0 else 0.0
            if expected > 0:
                lift = (t_obs / t_total) / (overall_obs / total_all)
            else:
                lift = 0.0
            lift_scores[b][num] = lift

    # ibox lift (permutation family performance per tier)
    ibox_tier = defaultdict(lambda: defaultdict(int))
    ibox_total_tier = defaultdict(int)
    ibox_overall = defaultdict(int)

    for number, tier in rows:
        b    = bucket(tier)
        ibox = "".join(sorted(number))
        ibox_tier[b][ibox]    += 1
        ibox_total_tier[b]    += 1
        ibox_overall[ibox]    += 1

    ibox_lift = defaultdict(dict)
    for b in ibox_total_tier:
        t_total = ibox_total_tier[b]
        for ibox, overall_cnt in ibox_overall.items():
            t_obs    = ibox_tier[b].get(ibox, 0)
            expected = (overall_cnt / total_all) * t_total if total_all > 0 else 0.0
            lift     = (t_obs / t_total) / (overall_cnt / total_all) if expected > 0 else 0.0
            ibox_lift[b][ibox] = lift

    return {
        "lift_scores": dict(lift_scores),
        "ibox_lift":   dict(ibox_lift),
        "tier_total":  dict(tier_total),
        "total_all":   total_all,
    }


def score_m6(model: dict, candidate: str, tier_group: str = "1st") -> float:
    """Score a candidate by its lift in the target tier."""
    ls    = model["lift_scores"].get(tier_group, {})
    num_lift = ls.get(candidate, 1.0)   # 1.0 = neutral (average)

    ibox  = "".join(sorted(candidate))
    il    = model["ibox_lift"].get(tier_group, {})
    ibox_lift = il.get(ibox, 1.0)

    # Blend: 60% number lift, 40% ibox family lift
    score = 0.6 * num_lift + 0.4 * ibox_lift
    # Normalize: lift of 1.0 is baseline, >1 is good, <1 is bad
    # Cap at 5x to avoid extreme outliers
    return min(score, 5.0) / 5.0


def predict_m6(model: dict, top_n: int = 100, tier_group: str = "1st") -> tuple:
    scores = {n: score_m6(model, n, tier_group=tier_group) for n in ALL_NUMBERS}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in ranked[:top_n]], scores


def save_m6(model: dict, suffix: str = ""):
    path = os.path.join(MODEL_DIR, f"m6_tier_bias{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"M6 saved → {path}")


def load_m6(suffix: str = "") -> dict:
    path = os.path.join(MODEL_DIR, f"m6_tier_bias{suffix}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
    con.close()

    print(f"Training M6 Tier Bias on draws 1–{max_draw} ...")
    model = train_m6(max_draw)

    print("\n-- Top 10 highest lift for 1st prize --")
    ls = model["lift_scores"].get("1st", {})
    top = sorted(ls.items(), key=lambda x: -x[1])[:10]
    for num, lift in top:
        print(f"  {num}  lift={lift:.3f}")

    save_m6(model)
    print("M6 done.")
