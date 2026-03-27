"""
M1 — Frequency + Recency Decay Model
Scores each candidate number based on how often it appeared historically,
with exponential decay so recent draws matter more than old ones.
Operates per tier (1st, 2nd, 3rd, starter, consolation-group).
"""

import duckdb
import numpy as np
import pickle
import os

DB_PATH    = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
MODEL_DIR  = os.path.expanduser("~/statlot-649/models")
os.makedirs(MODEL_DIR, exist_ok=True)

DECAY_HALF_LIFE = 200   # draws — score halves every 200 draws
ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]


def decay_weight(draw_id, current_draw_id, half_life=DECAY_HALF_LIFE):
    gap = current_draw_id - draw_id
    return np.exp(-np.log(2) * gap / half_life)


def train_m1(train_up_to_draw: int, tier_group: str = "all") -> dict:
    """
    Train M1 on draws <= train_up_to_draw.
    tier_group: 'top3' (1st/2nd/3rd), 'starter', 'consolation', 'all'
    Returns dict: number -> score
    """
    con = duckdb.connect(DB_PATH, read_only=True)

    tier_filter = {
        "top3":        "tier IN ('1st','2nd','3rd')",
        "1st":         "tier = '1st'",
        "starter":     "tier LIKE 'starter_%'",
        "consolation": "tier LIKE 'consolation_%'",
        "all":         "1=1"
    }.get(tier_group, "1=1")

    rows = con.execute(f"""
        SELECT number, draw_id
        FROM draw_numbers
        WHERE draw_id <= {train_up_to_draw}
          AND {tier_filter}
        ORDER BY draw_id
    """).fetchall()
    con.close()

    scores = {}
    for number, draw_id in rows:
        w = decay_weight(draw_id, train_up_to_draw)
        scores[number] = scores.get(number, 0.0) + w

    # Normalize to [0, 1]
    if scores:
        max_s = max(scores.values())
        scores = {k: v / max_s for k, v in scores.items()}

    # Assign 0 to numbers never seen
    for n in ALL_NUMBERS:
        if n not in scores:
            scores[n] = 0.0

    return scores


def predict_m1(scores: dict, top_n: int = 100) -> list:
    """Return top_n numbers sorted by M1 score descending."""
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in ranked[:top_n]]


def save_m1(scores: dict, suffix: str = ""):
    path = os.path.join(MODEL_DIR, f"m1_scores{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(scores, f)
    print(f"M1 saved → {path}")


def load_m1(suffix: str = "") -> dict:
    path = os.path.join(MODEL_DIR, f"m1_scores{suffix}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
    con.close()

    print(f"Training M1 on all {max_draw} draws ...")
    for tg in ["1st", "top3", "starter", "consolation", "all"]:
        scores = train_m1(max_draw, tier_group=tg)
        save_m1(scores, suffix=f"_{tg}")
        top10 = predict_m1(scores, top_n=10)
        print(f"  [{tg}] top-10: {top10}")
    print("M1 done.")
