"""
M2 — 2nd-Order Markov Chain Model
Models transition probabilities: given the last 2 draws' numbers,
what numbers are most likely to appear next?
Works at digit-position level (tractable) AND full-number level.
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


def train_m2(train_up_to_draw: int) -> dict:
    """
    Train 2nd-order Markov on draws <= train_up_to_draw.
    Returns a model dict with:
      - digit_transitions[pos][d_prev1][d_prev2] -> Counter of next digit
      - number_transitions[num_prev1][num_prev2] -> Counter of next number (top3 only)
    """
    con = duckdb.connect(DB_PATH, read_only=True)

    # Get ordered draws with all 23 numbers
    draws = con.execute(f"""
        SELECT draw_id, draw_date,
               prize_1st, prize_2nd, prize_3rd,
               starter_1, starter_2, starter_3, starter_4, starter_5,
               starter_6, starter_7, starter_8, starter_9, starter_10,
               consolation_1, consolation_2, consolation_3, consolation_4,
               consolation_5, consolation_6, consolation_7, consolation_8,
               consolation_9, consolation_10
        FROM draws
        WHERE draw_id <= {train_up_to_draw}
        ORDER BY draw_id
    """).fetchall()
    con.close()

    NUM_COLS = list(range(2, 27))  # indices of the 23 number columns in each row

    # ── digit-position transitions ────────────────────────────────────────────
    # For each position 0-3: count (prev1_digit, prev2_digit) -> next_digit
    # Use all 23 numbers per draw (each position independently)
    digit_trans = [[defaultdict(lambda: defaultdict(int)) for _ in range(10)]
                   for _ in range(4)]  # [pos][prev_digit] = {next_digit: count}

    # 1st order is enough for per-position (data is sparser at 2nd order)
    # We'll do 1st order per position, 2nd order for full 1st-prize number
    digit_trans_1st = [defaultdict(lambda: defaultdict(int)) for _ in range(4)]

    for i in range(1, len(draws)):
        prev_row = draws[i - 1]
        curr_row = draws[i]
        # Use ALL numbers from previous draw, map to all numbers in current draw
        # For digit transitions: compare same-tier numbers
        for col_idx in NUM_COLS:
            prev_num = prev_row[col_idx]
            curr_num = curr_row[col_idx]
            if prev_num and curr_num and len(prev_num) == 4 and len(curr_num) == 4:
                for pos in range(4):
                    d_prev = int(prev_num[pos])
                    d_curr = int(curr_num[pos])
                    digit_trans_1st[pos][d_prev][d_curr] += 1

    # ── full-number 2nd-order transitions (1st prize only) ───────────────────
    # (prev2_num, prev1_num) -> next_num
    num_trans_2nd = defaultdict(lambda: defaultdict(int))
    num_trans_1st = defaultdict(lambda: defaultdict(int))

    prize_col = 2  # prize_1st column index
    for i in range(1, len(draws)):
        p1 = draws[i - 1][prize_col]
        c  = draws[i][prize_col]
        if p1 and c:
            num_trans_1st[p1][c] += 1

    for i in range(2, len(draws)):
        p2 = draws[i - 2][prize_col]
        p1 = draws[i - 1][prize_col]
        c  = draws[i][prize_col]
        if p2 and p1 and c:
            num_trans_2nd[(p2, p1)][c] += 1

    # ── normalize to probabilities ────────────────────────────────────────────
    def normalize(counter):
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()} if total else {}

    digit_probs = []
    for pos in range(4):
        pos_dict = {}
        for d_prev, counts in digit_trans_1st[pos].items():
            pos_dict[d_prev] = normalize(counts)
        digit_probs.append(pos_dict)

    num_probs_1st = {k: normalize(v) for k, v in num_trans_1st.items()}
    num_probs_2nd = {k: normalize(v) for k, v in num_trans_2nd.items()}

    return {
        "digit_probs":    digit_probs,      # list of 4 dicts
        "num_probs_1st":  num_probs_1st,
        "num_probs_2nd":  num_probs_2nd,
        "last_2_draws":   [draws[-2][prize_col], draws[-1][prize_col]],
        "last_draw_id":   draws[-1][0],
    }


def score_m2(model: dict, candidate: str) -> float:
    """Score a 4-digit candidate using the Markov model."""
    if not candidate or len(candidate) != 4:
        return 0.0

    score = 0.0
    last2 = model["last_2_draws"]
    prev1, prev2 = last2[1], last2[0]

    # ── digit-position score ──────────────────────────────────────────────────
    digit_score = 0.0
    for pos in range(4):
        d_cand = int(candidate[pos])
        if prev1 and len(prev1) == 4:
            d_prev = int(prev1[pos])
            prob   = model["digit_probs"][pos].get(d_prev, {}).get(d_cand, 0.0)
            digit_score += prob
    digit_score /= 4.0

    # ── full-number score (1st prize context) ────────────────────────────────
    num_score_1st = model["num_probs_1st"].get(prev1, {}).get(candidate, 0.0)
    num_score_2nd = 0.0
    if prev2 and prev1:
        num_score_2nd = model["num_probs_2nd"].get((prev2, prev1), {}).get(candidate, 0.0)

    # Blend: digit_position (broad signal) + 1st-order number + 2nd-order number
    score = 0.5 * digit_score + 0.3 * num_score_1st + 0.2 * num_score_2nd
    return score


def predict_m2(model: dict, top_n: int = 100) -> list:
    scores = {n: score_m2(model, n) for n in ALL_NUMBERS}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in ranked[:top_n]], scores


def save_m2(model: dict, suffix: str = ""):
    path = os.path.join(MODEL_DIR, f"m2_markov{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"M2 saved → {path}")


def load_m2(suffix: str = "") -> dict:
    path = os.path.join(MODEL_DIR, f"m2_markov{suffix}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
    con.close()

    print(f"Training M2 Markov on draws 1–{max_draw} ...")
    model = train_m2(max_draw)
    top10, _ = predict_m2(model, top_n=10)
    print(f"  Top-10 predictions: {top10}")
    print(f"  Last 2 draws used: {model['last_2_draws']}")
    save_m2(model)
    print("M2 done.")
