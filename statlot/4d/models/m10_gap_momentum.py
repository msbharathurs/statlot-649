"""
M10 — Gap & Momentum Model
For each number, tracks:
  1. Current gap (draws since last appearance) vs historical average gap
     → "overdue" numbers score higher (gap > avg gap)
  2. Momentum: numbers that appeared in the last K draws with increasing frequency
     (hot streak) score higher
  3. "Regression to mean" signal: numbers that appeared WAY too recently
     (gap << avg_gap) are penalised (cool-down period)
This is fundamentally different from M1 (which uses decay) — M10 uses
gap statistics to identify numbers at the inflection point of their cycle.
"""

import duckdb
import numpy as np
import os
from collections import defaultdict

DB_PATH   = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]

MOMENTUM_WINDOW = 30   # draws to look back for momentum
RECENT_WINDOW   = 10   # draws for cool-down check


def train_m10(train_up_to_draw: int, tier_group: str = "1st") -> dict:
    con = duckdb.connect(DB_PATH, read_only=True)

    tier_filter = {
        "1st": "tier = '1st'",
        "top3": "tier IN ('1st','2nd','3rd')",
        "all": "1=1"
    }.get(tier_group, "tier = '1st'")

    rows = con.execute(f"""
        SELECT number, draw_id
        FROM draw_numbers
        WHERE draw_id <= {train_up_to_draw}
          AND {tier_filter}
        ORDER BY number, draw_id
    """).fetchall()

    max_draw = con.execute(f"SELECT MAX(draw_id) FROM draws WHERE draw_id <= {train_up_to_draw}").fetchone()[0]
    total_draws = con.execute(f"SELECT COUNT(*) FROM draws WHERE draw_id <= {train_up_to_draw}").fetchone()[0]
    con.close()

    # Build appearance list per number
    appearances = defaultdict(list)
    for number, draw_id in rows:
        appearances[number].append(draw_id)

    # Count appearances in momentum + recent windows
    momentum_counts = defaultdict(int)
    recent_counts   = defaultdict(int)
    for number, draw_ids in appearances.items():
        for d in draw_ids:
            gap = max_draw - d
            if gap < MOMENTUM_WINDOW:
                momentum_counts[number] += 1
            if gap < RECENT_WINDOW:
                recent_counts[number] += 1

    scores = {}
    for num in ALL_NUMBERS:
        app_list = appearances.get(num, [])
        n_apps   = len(app_list)

        if n_apps < 3:
            # Too rare — neutral score
            scores[num] = 0.5
            continue

        # Average gap between appearances
        gaps = [app_list[i+1] - app_list[i] for i in range(len(app_list)-1)]
        avg_gap = np.mean(gaps)
        std_gap = np.std(gaps) + 1.0

        # Current gap since last appearance
        last_app  = app_list[-1]
        curr_gap  = max_draw - last_app

        # Overdue score: how far past avg_gap are we? Normalised by std
        z_overdue = (curr_gap - avg_gap) / std_gap  # positive = overdue

        # Momentum score: appearances in last MOMENTUM_WINDOW draws
        expected_in_window = n_apps * MOMENTUM_WINDOW / total_draws
        momentum_score = momentum_counts.get(num, 0) / max(expected_in_window, 0.1)

        # Cool-down penalty: appeared too recently
        cool_penalty = 1.0
        if recent_counts.get(num, 0) > 0:
            cool_penalty = 0.5

        raw = (np.tanh(z_overdue * 0.5) + 1) * 0.5  # maps to [0,1]
        raw = raw * momentum_score * cool_penalty
        scores[num] = max(0.0, raw)

    # Normalize
    max_s = max(scores.values(), default=1.0)
    model = {n: round(s / max_s, 6) for n, s in scores.items()}
    return model


def score_m10(model: dict, candidate: str) -> float:
    return model.get(candidate, 0.0)
