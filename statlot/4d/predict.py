"""
predict.py — Final Prediction Engine
Trains ALL models on complete historical data (draws 1–4471),
then generates final ranked list of top-50 numbers to play.
Outputs: ranked predictions per tier + confidence scores.
"""

import duckdb
import numpy as np
import pickle
import json
import os
from datetime import datetime

DB_PATH    = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
MODEL_DIR  = os.path.expanduser("~/statlot-649/models")
RESULT_DIR = os.path.expanduser("~/statlot-649/predictions")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.m1_frequency      import train_m1
from models.m2_markov         import train_m2, score_m2
from models.m3_poisson        import train_m3, score_m3
from models.m4_digit_position import train_m4, score_m4
from models.m5_structure      import train_m5, score_m5
from models.m6_tier_bias      import train_m6, score_m6
from models.m7_fft_cycle      import train_m7, score_m7

ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]


# ── Final tuned weights (update these after Optuna tuning in Phase 2) ─────────
FINAL_WEIGHTS = {
    "m1": 0.25,
    "m2": 0.20,
    "m3": 0.15,
    "m4": 0.15,
    "m5": 0.10,
    "m6": 0.10,
    "m7": 0.05,
}


def train_all_models(max_draw: int) -> dict:
    print(f"\nTraining all models on draws 1–{max_draw} ...")
    models = {}

    print("  [1/7] M1 Frequency+Decay ...")
    models["m1_1st"]    = train_m1(max_draw, tier_group="1st")
    models["m1_top3"]   = train_m1(max_draw, tier_group="top3")
    models["m1_all"]    = train_m1(max_draw, tier_group="all")

    print("  [2/7] M2 Markov Chain ...")
    models["m2"] = train_m2(max_draw)

    print("  [3/7] M3 Poisson Gap ...")
    models["m3"] = train_m3(max_draw)

    print("  [4/7] M4 Digit Position ...")
    models["m4_1st"]  = train_m4(max_draw, tier_group="1st")
    models["m4_all"]  = train_m4(max_draw, tier_group="all")

    print("  [5/7] M5 Structure Distribution ...")
    models["m5"] = train_m5(max_draw)

    print("  [6/7] M6 Tier Bias ...")
    models["m6"] = train_m6(max_draw)

    print("  [7/7] M7 FFT Cycle ...")
    models["m7"] = train_m7(max_draw)

    # Save all models
    for name, obj in models.items():
        path = os.path.join(MODEL_DIR, f"{name}_final.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    print(f"  All models saved to {MODEL_DIR}/")

    return models


def score_all(models: dict, candidate: str, tier_group: str = "1st",
              dow: str = None, weights: dict = None) -> float:
    w = weights or FINAL_WEIGHTS
    m1_key = f"m1_{tier_group}" if f"m1_{tier_group}" in models else "m1_all"
    m4_key = f"m4_{tier_group}" if f"m4_{tier_group}" in models else "m4_all"

    s  = w.get("m1", 0) * models[m1_key].get(candidate, 0.0)
    s += w.get("m2", 0) * score_m2(models["m2"], candidate)
    s += w.get("m3", 0) * score_m3(models["m3"], candidate)
    s += w.get("m4", 0) * score_m4(models[m4_key], candidate, dow=dow)
    s += w.get("m5", 0) * score_m5(models["m5"], candidate, tier_group=tier_group)
    s += w.get("m6", 0) * score_m6(models["m6"], candidate, tier_group=tier_group)
    s += w.get("m7", 0) * score_m7(models["m7"], candidate)
    return s


def generate_predictions(models: dict, next_draw_dow: str = None,
                          top_n: int = 50, weights: dict = None) -> dict:
    print(f"\nScoring all 10,000 candidates ...")
    predictions = {}

    for tier_group in ["1st", "top3", "all"]:
        scores = {}
        for num in ALL_NUMBERS:
            scores[num] = score_all(models, num, tier_group=tier_group,
                                    dow=next_draw_dow, weights=weights)
        # Normalize
        max_s = max(scores.values()) if scores else 1.0
        scores = {k: round(v / max_s, 6) for k, v in scores.items()}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        predictions[tier_group] = {
            "top_numbers": [n for n, _ in ranked[:top_n]],
            "scores":      dict(ranked[:top_n]),
        }
        print(f"  [{tier_group}] Top-10: {[n for n, _ in ranked[:10]]}")

    return predictions


def format_output(predictions: dict, next_draw_info: dict) -> dict:
    """Format final output for display / saving."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "next_draw":    next_draw_info,
        "predictions":  {},
        "play_list":    [],
    }

    # Primary play list = 1st prize tier top-50
    top50 = predictions["1st"]["top_numbers"]
    for rank, num in enumerate(top50, 1):
        output["play_list"].append({
            "rank":    rank,
            "number":  num,
            "score_1st":  predictions["1st"]["scores"].get(num, 0),
            "score_top3": predictions["top3"]["scores"].get(num, 0),
            "score_all":  predictions["all"]["scores"].get(num, 0),
        })

    output["predictions"] = predictions
    return output


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    max_draw  = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
    last_row  = con.execute(
        "SELECT draw_id, draw_date, day_of_week FROM draws ORDER BY draw_id DESC LIMIT 1"
    ).fetchone()
    con.close()

    print(f"Last draw: #{last_row[0]} on {last_row[1]} ({last_row[2]})")

    # Days of week cycling
    dow_cycle = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    last_dow  = last_row[2]
    # Singapore Pools draws: Wed, Sat, Sun
    draw_days = ["Wed","Sat","Sun"]
    last_idx  = draw_days.index(last_dow) if last_dow in draw_days else 0
    next_dow  = draw_days[(last_idx + 1) % len(draw_days)]

    print(f"Predicting for next draw day: {next_dow}")

    models = train_all_models(max_draw)
    preds  = generate_predictions(models, next_draw_dow=next_dow, top_n=50)
    output = format_output(preds, {
        "after_draw_id": max_draw,
        "after_draw_date": str(last_row[1]),
        "expected_day": next_dow,
    })

    # Save
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULT_DIR, f"predictions_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL PREDICTIONS — Top 20 numbers to play:")
    print(f"{'='*60}")
    for entry in output["play_list"][:20]:
        print(f"  #{entry['rank']:2d}  {entry['number']}   "
              f"score_1st={entry['score_1st']:.4f}  "
              f"score_all={entry['score_all']:.4f}")
    print(f"\nFull list saved → {path}")
