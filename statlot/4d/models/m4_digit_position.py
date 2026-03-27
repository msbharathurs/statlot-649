"""
M4 — Digit-Position Frequency Model
Models each of the 4 digit positions independently AND jointly (d1d2, d3d4 pairs).
Generates candidates by sampling from position-specific distributions,
then scores full numbers by their joint positional probability.
Also tracks day-of-week bias per position.
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
DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


def train_m4(train_up_to_draw: int, tier_group: str = "top3") -> dict:
    """
    Train M4 on draws <= train_up_to_draw.
    tier_group: 'top3', '1st', 'starter', 'consolation', 'all'
    """
    con = duckdb.connect(DB_PATH, read_only=True)

    tier_filter = {
        "top3":        "tier IN ('1st','2nd','3rd')",
        "1st":         "tier = '1st'",
        "starter":     "tier LIKE 'starter_%'",
        "consolation": "tier LIKE 'consolation_%'",
        "all":         "1=1"
    }.get(tier_group, "tier IN ('1st','2nd','3rd')")

    rows = con.execute(f"""
        SELECT number, d1, d2, d3, d4, day_of_week
        FROM draw_numbers
        WHERE draw_id <= {train_up_to_draw}
          AND {tier_filter}
    """).fetchall()
    con.close()

    # ── per-position digit frequency ─────────────────────────────────────────
    pos_freq   = [defaultdict(int) for _ in range(4)]   # pos -> digit -> count
    pair_freq  = defaultdict(int)                         # (d1,d2) -> count
    pair_freq2 = defaultdict(int)                         # (d3,d4) -> count
    dow_pos_freq = defaultdict(lambda: [defaultdict(int) for _ in range(4)])

    for number, d1, d2, d3, d4, dow in rows:
        digits = [d1, d2, d3, d4]
        for pos, d in enumerate(digits):
            pos_freq[pos][d] += 1
            if dow:
                dow_pos_freq[dow][pos][d] += 1
        pair_freq[(d1, d2)] += 1
        pair_freq2[(d3, d4)] += 1

    # ── normalize ─────────────────────────────────────────────────────────────
    def normalize_dict(d):
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total else {}

    pos_probs    = [normalize_dict(pos_freq[p]) for p in range(4)]
    pair_probs   = normalize_dict(pair_freq)
    pair_probs2  = normalize_dict(pair_freq2)

    dow_pos_probs = {}
    for dow in DAYS:
        dow_pos_probs[dow] = [normalize_dict(dow_pos_freq[dow][p]) for p in range(4)]

    return {
        "pos_probs":    pos_probs,      # [pos][digit] = prob
        "pair_probs":   pair_probs,     # [(d1,d2)] = prob
        "pair_probs2":  pair_probs2,    # [(d3,d4)] = prob
        "dow_pos_probs": dow_pos_probs, # [dow][pos][digit] = prob
        "tier_group":   tier_group,
        "train_up_to":  train_up_to_draw,
    }


def score_m4(model: dict, candidate: str, dow: str = None) -> float:
    """Score a 4-digit candidate using digit-position probabilities."""
    if not candidate or len(candidate) != 4:
        return 0.0

    d = [int(c) for c in candidate]
    d1, d2, d3, d4 = d

    # ── independent positional score ─────────────────────────────────────────
    pos_score = 1.0
    for pos, digit in enumerate(d):
        p = model["pos_probs"][pos].get(digit, 1e-6)
        pos_score *= p
    # Geometric mean (avoid extreme skew)
    pos_score = pos_score ** 0.25

    # ── pair score ────────────────────────────────────────────────────────────
    pair_score  = model["pair_probs"].get((d1, d2), 1e-6)
    pair_score2 = model["pair_probs2"].get((d3, d4), 1e-6)

    # ── day-of-week adjustment ────────────────────────────────────────────────
    dow_score = 1.0
    if dow and dow in model.get("dow_pos_probs", {}):
        dow_probs = model["dow_pos_probs"][dow]
        for pos, digit in enumerate(d):
            p = dow_probs[pos].get(digit, 1e-6)
            dow_score *= p
        dow_score = dow_score ** 0.25

    # Blend
    score = 0.4 * pos_score + 0.2 * pair_score + 0.2 * pair_score2 + 0.2 * dow_score
    return score


def predict_m4(model: dict, top_n: int = 100, dow: str = None) -> tuple:
    scores = {n: score_m4(model, n, dow=dow) for n in ALL_NUMBERS}
    # Normalize to [0,1]
    max_s = max(scores.values()) if scores else 1.0
    scores = {k: v / max_s for k, v in scores.items()}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in ranked[:top_n]], scores


def save_m4(model: dict, suffix: str = ""):
    path = os.path.join(MODEL_DIR, f"m4_digit_pos{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"M4 saved → {path}")


def load_m4(suffix: str = "") -> dict:
    path = os.path.join(MODEL_DIR, f"m4_digit_pos{suffix}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
    last_dow = con.execute(
        "SELECT day_of_week FROM draws ORDER BY draw_id DESC LIMIT 1"
    ).fetchone()[0]
    con.close()

    print(f"Training M4 on draws 1–{max_draw} ...")
    for tg in ["top3", "1st", "all"]:
        model = train_m4(max_draw, tier_group=tg)
        top10, _ = predict_m4(model, top_n=10)
        print(f"  [{tg}] top-10: {top10}")
        save_m4(model, suffix=f"_{tg}")

    print(f"\nM4 done. (last draw day: {last_dow})")
