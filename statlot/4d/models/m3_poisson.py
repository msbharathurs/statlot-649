"""
M3 — Poisson Gap Model
Models the gap (draws between appearances) for each number as a Poisson process.
If a number's actual gap >> expected gap, it's "overdue" → higher score.
Score = P(gap >= observed) under the fitted Poisson/exponential distribution.
"""

import duckdb
import numpy as np
import pickle
import os
from scipy.stats import poisson, expon

DB_PATH   = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
MODEL_DIR = os.path.expanduser("~/statlot-649/models")
os.makedirs(MODEL_DIR, exist_ok=True)

ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]


def train_m3(train_up_to_draw: int) -> dict:
    """
    Fit Poisson gap model on draws <= train_up_to_draw.
    For each number: compute mean gap between appearances, last seen draw.
    Returns model dict.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    rows = con.execute(f"""
        SELECT number, draw_id
        FROM draw_numbers
        WHERE draw_id <= {train_up_to_draw}
        ORDER BY number, draw_id
    """).fetchall()
    con.close()

    from collections import defaultdict
    appearances = defaultdict(list)
    for number, draw_id in rows:
        appearances[number].append(draw_id)

    model = {}
    for num in ALL_NUMBERS:
        apps = appearances.get(num, [])
        if len(apps) < 2:
            model[num] = {
                "mean_gap":      None,
                "last_seen":     apps[-1] if apps else None,
                "times_seen":    len(apps),
                "current_gap":   (train_up_to_draw - apps[-1]) if apps else None,
                "overdue_score": 0.0,
            }
            continue

        gaps = [apps[i+1] - apps[i] for i in range(len(apps)-1)]
        mean_gap    = np.mean(gaps)
        std_gap     = np.std(gaps) if len(gaps) > 1 else mean_gap
        last_seen   = apps[-1]
        current_gap = train_up_to_draw - last_seen

        # Overdue score: how many "standard deviations" overdue is this number?
        # P(X >= current_gap) under exponential with mean=mean_gap
        # Higher = more overdue = higher score
        if mean_gap > 0:
            # Survival function of exponential: P(X >= t) = exp(-t/mean)
            overdue_prob = np.exp(-current_gap / mean_gap)
            # We INVERT this — lower survival = more overdue → higher score
            overdue_score = 1.0 - overdue_prob
        else:
            overdue_score = 0.0

        model[num] = {
            "mean_gap":      round(mean_gap, 2),
            "std_gap":       round(std_gap, 2),
            "last_seen":     last_seen,
            "times_seen":    len(apps),
            "current_gap":   current_gap,
            "overdue_score": round(overdue_score, 6),
        }

    return model


def score_m3(model: dict, candidate: str) -> float:
    """Return the overdue score for a candidate number."""
    entry = model.get(candidate)
    if entry is None:
        return 0.0
    return entry["overdue_score"]


def predict_m3(model: dict, top_n: int = 100) -> tuple:
    scores = {n: score_m3(model, n) for n in ALL_NUMBERS}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in ranked[:top_n]], scores


def save_m3(model: dict, suffix: str = ""):
    path = os.path.join(MODEL_DIR, f"m3_poisson{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"M3 saved → {path}")


def load_m3(suffix: str = "") -> dict:
    path = os.path.join(MODEL_DIR, f"m3_poisson{suffix}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
    con.close()

    print(f"Training M3 Poisson on draws 1–{max_draw} ...")
    model = train_m3(max_draw)

    # Stats
    scored = [(n, v["overdue_score"], v["current_gap"], v["mean_gap"])
              for n, v in model.items() if v["mean_gap"] is not None]
    scored.sort(key=lambda x: -x[1])

    print(f"  Numbers with gap data: {len(scored)}")
    print(f"  Top-10 most overdue (1st prize tier context):")
    for num, score, gap, mean_gap in scored[:10]:
        print(f"    {num}  overdue_score={score:.4f}  current_gap={gap}  mean_gap={mean_gap}")

    save_m3(model)
    print("M3 done.")
