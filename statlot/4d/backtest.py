"""
backtest.py — Walk-forward backtesting harness for 4D prediction engine
Trains M1–M7 on historical window, predicts top-N for next draw,
scores against actual, accumulates lift metrics.
"""

import duckdb
import numpy as np
import os
import json
from datetime import datetime

DB_PATH   = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
MODEL_DIR = os.path.expanduser("~/statlot-649/models")
RESULT_DIR = os.path.expanduser("~/statlot-649/backtest_results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Import models
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.m1_frequency     import train_m1
from models.m2_markov        import train_m2, score_m2
from models.m3_poisson       import train_m3, score_m3
from models.m4_digit_position import train_m4, score_m4
from models.m5_structure     import train_m5, score_m5
from models.m6_tier_bias     import train_m6, score_m6
from models.m7_fft_cycle     import train_m7, score_m7

ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]

# Default ensemble weights (will be tuned by Optuna in Phase 2)
DEFAULT_WEIGHTS = {
    "m1": 0.25,
    "m2": 0.20,
    "m3": 0.15,
    "m4": 0.15,
    "m5": 0.10,
    "m6": 0.10,
    "m7": 0.05,
}


def ensemble_score(models: dict, candidate: str, weights: dict = None,
                   tier_group: str = "1st", dow: str = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    s = 0.0
    if "m1" in models:
        s += w.get("m1", 0) * models["m1"].get(candidate, 0.0)
    if "m2" in models:
        s += w.get("m2", 0) * score_m2(models["m2"], candidate)
    if "m3" in models:
        s += w.get("m3", 0) * score_m3(models["m3"], candidate)
    if "m4" in models:
        s += w.get("m4", 0) * score_m4(models["m4"], candidate, dow=dow)
    if "m5" in models:
        s += w.get("m5", 0) * score_m5(models["m5"], candidate, tier_group=tier_group)
    if "m6" in models:
        s += w.get("m6", 0) * score_m6(models["m6"], candidate, tier_group=tier_group)
    if "m7" in models:
        s += w.get("m7", 0) * score_m7(models["m7"], candidate)
    return s


def get_all_numbers_in_draw(draw_row: tuple) -> set:
    """Extract all 23 numbers from a draw row."""
    return {v for v in draw_row[2:] if v and len(v) == 4}


def run_backtest(
    train_start: int = 1,
    train_end: int   = 4200,
    test_end: int    = 4471,
    top_n: int       = 50,
    step: int        = 10,       # re-train every N draws
    weights: dict    = None,
    tier_group: str  = "1st",
    verbose: bool    = True,
):
    """
    Walk-forward backtest:
    - Train on draws [train_start, train_end]
    - Predict top_n numbers for each test draw
    - Score: did any prediction appear in the actual draw?
    - Re-train every `step` draws
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    test_draws = con.execute(f"""
        SELECT draw_id, day_of_week,
               prize_1st, prize_2nd, prize_3rd,
               starter_1, starter_2, starter_3, starter_4, starter_5,
               starter_6, starter_7, starter_8, starter_9, starter_10,
               consolation_1, consolation_2, consolation_3, consolation_4,
               consolation_5, consolation_6, consolation_7, consolation_8,
               consolation_9, consolation_10
        FROM draws
        WHERE draw_id > {train_end} AND draw_id <= {test_end}
        ORDER BY draw_id
    """).fetchall()
    con.close()

    results = []
    models  = {}
    current_train_end = train_end

    print(f"\n{'='*60}")
    print(f"Backtest: train[{train_start}–{train_end}] → test[{train_end+1}–{test_end}]")
    print(f"Test draws: {len(test_draws)}, top_n={top_n}, step={step}")
    print(f"{'='*60}")

    for i, draw_row in enumerate(test_draws):
        draw_id = draw_row[0]
        dow     = draw_row[1]
        actual  = get_all_numbers_in_draw(draw_row)
        actual_1st = draw_row[2]

        # Re-train every `step` draws
        if i == 0 or i % step == 0:
            if verbose:
                print(f"\n  [Re-training at draw {draw_id}, train_end={current_train_end}]")
            models["m1"] = train_m1(current_train_end, tier_group=tier_group)
            models["m2"] = train_m2(current_train_end)
            models["m3"] = train_m3(current_train_end)
            models["m4"] = train_m4(current_train_end, tier_group=tier_group)
            models["m5"] = train_m5(current_train_end)
            models["m6"] = train_m6(current_train_end)
            models["m7"] = train_m7(current_train_end)

        # Score all 10,000 candidates
        scores = {}
        for num in ALL_NUMBERS:
            scores[num] = ensemble_score(models, num, weights=weights,
                                          tier_group=tier_group, dow=dow)

        ranked    = sorted(scores.items(), key=lambda x: -x[1])
        top_picks = [n for n, _ in ranked[:top_n]]

        # Scoring
        hit_any    = any(n in actual for n in top_picks)      # any of 23 numbers
        hit_1st    = actual_1st in top_picks if actual_1st else False
        n_hits     = sum(1 for n in top_picks if n in actual)  # hits across all 23

        results.append({
            "draw_id":   draw_id,
            "hit_any":   hit_any,
            "hit_1st":   hit_1st,
            "n_hits":    n_hits,
            "top_picks": top_picks,
            "actual_1st": actual_1st,
        })

        current_train_end = draw_id  # slide window

        if verbose and (i + 1) % 10 == 0:
            recent  = results[-10:]
            hit_rate = sum(r["hit_any"] for r in recent) / len(recent)
            avg_hits = np.mean([r["n_hits"] for r in recent])
            print(f"  [{i+1}/{len(test_draws)}] draw {draw_id} — "
                  f"last-10 hit_rate={hit_rate:.1%}  avg_hits={avg_hits:.2f}")

    # ── summary ───────────────────────────────────────────────────────────────
    n_test     = len(results)
    hit_rate   = sum(r["hit_any"] for r in results) / n_test
    hit_1st    = sum(r["hit_1st"] for r in results) / n_test
    avg_hits   = np.mean([r["n_hits"] for r in results])
    random_baseline_any = 1 - ((10000 - 23) / 10000) ** top_n   # P(at least 1 hit)
    lift_any   = hit_rate / random_baseline_any if random_baseline_any > 0 else 0

    summary = {
        "run_date":      datetime.now().isoformat(),
        "train_range":   f"{train_start}–{train_end}",
        "test_range":    f"{train_end+1}–{test_end}",
        "n_test_draws":  n_test,
        "top_n":         top_n,
        "step":          step,
        "tier_group":    tier_group,
        "hit_rate_any":  round(hit_rate, 4),
        "hit_rate_1st":  round(hit_1st, 4),
        "avg_hits_per_draw": round(avg_hits, 4),
        "random_baseline_any": round(random_baseline_any, 4),
        "lift_any":      round(lift_any, 4),
        "weights":       weights or DEFAULT_WEIGHTS,
    }

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"  Hit rate (any of 23):   {hit_rate:.2%}  (random baseline: {random_baseline_any:.2%})")
    print(f"  Hit rate (1st prize):   {hit_1st:.2%}")
    print(f"  Avg hits per draw:      {avg_hits:.3f}")
    print(f"  Lift over random:       {lift_any:.3f}x")
    print(f"{'='*60}")

    # Save results
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULT_DIR, f"backtest_{ts}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "draws": results}, f, indent=2)
    print(f"Results saved → {path}")

    return summary, results


if __name__ == "__main__":
    summary, _ = run_backtest(
        train_start=1,
        train_end=4200,
        test_end=4471,
        top_n=50,
        step=10,
        tier_group="1st",
        verbose=True,
    )
