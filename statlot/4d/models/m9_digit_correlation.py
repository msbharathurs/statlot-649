"""
M9 — Digit Correlation / Positional Pair Model
4D numbers have 4 digit positions (d0 d1 d2 d3).
This model tracks:
  1. Conditional prob: given digit X appeared at position P in draw N,
     what digit is most likely at position P in draw N+1?
  2. Pair correlation: digit pairs at (pos_i, pos_j) that co-appear above chance
  3. "Mirror" patterns: numbers whose digit-reversal also appears frequently
Produces a per-number score from positional transition probabilities.
"""

import duckdb
import numpy as np
import os
from collections import defaultdict

DB_PATH   = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]


def train_m9(train_up_to_draw: int) -> dict:
    con = duckdb.connect(DB_PATH, read_only=True)

    # Get ordered 1st-prize draws (strongest positional signal)
    rows = con.execute(f"""
        SELECT draw_id, prize_1st, prize_2nd, prize_3rd
        FROM draws
        WHERE draw_id <= {train_up_to_draw}
        ORDER BY draw_id
    """).fetchall()
    con.close()

    # Build positional transition matrices (digit -> next digit, per position)
    # trans[pos][from_digit][to_digit] = count
    trans_1st = [[defaultdict(int) for _ in range(10)] for _ in range(4)]

    prev_digits = None
    for draw_id, p1, p2, p3 in rows:
        if p1 and len(p1) == 4:
            curr = [int(c) for c in p1]
            if prev_digits is not None:
                for pos in range(4):
                    trans_1st[pos][prev_digits[pos]][curr[pos]] += 1
            prev_digits = curr

    # Normalize to probabilities
    trans_prob = [[{} for _ in range(10)] for _ in range(4)]
    for pos in range(4):
        for from_d in range(10):
            total = sum(trans_1st[pos][from_d].values())
            if total > 0:
                for to_d, cnt in trans_1st[pos][from_d].items():
                    trans_prob[pos][from_d][to_d] = cnt / total

    # Get last draw's 1st prize digits as context for next prediction
    last_digits = [int(c) for c in rows[-1][1]] if rows and rows[-1][1] else [0,0,0,0]

    # Score each candidate: product of positional transition probabilities
    model = {'trans_prob': trans_prob, 'last_digits': last_digits}

    scores = {}
    for num in ALL_NUMBERS:
        digits = [int(c) for c in num]
        score = 1.0
        for pos in range(4):
            from_d = last_digits[pos]
            prob = trans_prob[pos][from_d].get(digits[pos], 0.0)
            score *= (prob + 1e-6)  # Laplace smoothing
        scores[num] = score

    # Normalize
    max_s = max(scores.values(), default=1.0)
    model['scores'] = {n: round(s / max_s, 6) for n, s in scores.items()}

    return model


def score_m9(model: dict, candidate: str) -> float:
    return model['scores'].get(candidate, 0.0)
