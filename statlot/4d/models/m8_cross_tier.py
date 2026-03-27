"""
M8 — Cross-Tier Pattern Model
Observes which numbers appear across MULTIPLE tiers in the same draw
(e.g. a number that appears in 1st AND starter in the same draw = "sticky" number).
Also tracks: which numbers historically co-appear with recent 1st-prize numbers.
Hypothesis: certain numbers cluster with prize-winners — finding them early gives an edge.
"""

import duckdb
import numpy as np
import os
from collections import defaultdict

DB_PATH   = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]


def train_m8(train_up_to_draw: int) -> dict:
    con = duckdb.connect(DB_PATH, read_only=True)

    # All numbers per draw, with tier
    rows = con.execute(f"""
        SELECT draw_id, number, tier
        FROM draw_numbers
        WHERE draw_id <= {train_up_to_draw}
        ORDER BY draw_id
    """).fetchall()

    # Recent 1st-prize numbers (last 100 draws) for co-occurrence
    recent_1st = set(con.execute(f"""
        SELECT prize_1st FROM draws
        WHERE draw_id <= {train_up_to_draw}
        ORDER BY draw_id DESC LIMIT 100
    """).fetchdf()['prize_1st'].tolist())

    con.close()

    # Build per-draw tier map
    draw_tiers = defaultdict(lambda: defaultdict(set))  # draw_id -> number -> set of tiers
    for draw_id, number, tier in rows:
        draw_tiers[draw_id][number].add(tier)

    # Score 1: how often does each number appear in multiple tiers in the same draw?
    multi_tier_count = defaultdict(int)
    total_draws = len(draw_tiers)
    for draw_id, num_tiers in draw_tiers.items():
        for number, tiers in num_tiers.items():
            if len(tiers) > 1:
                multi_tier_count[number] += 1

    # Score 2: co-occurrence with recent 1st-prize numbers
    # Build co-occurrence: which numbers appear in same draw as each 1st-prize number?
    cooc = defaultdict(int)
    prize1_draws = defaultdict(set)  # 1st prize number -> draw_ids it won
    for draw_id, num_tiers in draw_tiers.items():
        draw_nums = set(num_tiers.keys())
        for num in draw_nums:
            if num in recent_1st:
                # All other numbers in this draw co-occur with this prize winner
                for other in draw_nums:
                    if other != num:
                        cooc[other] += 1

    # Normalize scores
    max_multi = max(multi_tier_count.values(), default=1)
    max_cooc  = max(cooc.values(), default=1)

    model = {}
    for num in ALL_NUMBERS:
        s1 = multi_tier_count.get(num, 0) / max_multi
        s2 = cooc.get(num, 0) / max_cooc
        model[num] = round(0.4 * s1 + 0.6 * s2, 6)

    return model


def score_m8(model: dict, candidate: str) -> float:
    return model.get(candidate, 0.0)
