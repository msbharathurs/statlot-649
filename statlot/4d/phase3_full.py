"""
Phase 3 — Full M1-M10 pipeline
Step 1: Walk-forward backtest (4201-4471, 271 draws) with default weights
Step 2: Optuna weight tuning (train 1-4400, val 4401-4471, 150 trials)
Step 3: Full backtest with tuned weights
Step 4: Final predictions on all 5461 draws → top-50 for next draw (Saturday)
"""

import sys, os, json, optuna
import numpy as np
from datetime import datetime
import duckdb

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.expanduser("~/statlot-649/statlot/4d"))

from models.m1_frequency         import train_m1
from models.m2_markov            import train_m2, score_m2
from models.m3_poisson           import train_m3, score_m3
from models.m4_digit_position    import train_m4, score_m4
from models.m5_structure         import train_m5, score_m5
from models.m6_tier_bias         import train_m6, score_m6
from models.m7_fft_cycle         import train_m7, score_m7
from models.m8_cross_tier        import train_m8, score_m8
from models.m9_digit_correlation import train_m9, score_m9
from models.m10_gap_momentum     import train_m10, score_m10
from backtest import run_backtest

DB_PATH    = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
RESULT_DIR = os.path.expanduser("~/statlot-649/backtest_results")
PRED_DIR   = os.path.expanduser("~/statlot-649/predictions")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]
M_KEYS      = ["m1","m2","m3","m4","m5","m6","m7","m8","m9","m10"]

TRAIN_END   = 4400
VAL_START   = 4401
VAL_END     = 4471
TOP_N       = 50
N_TRIALS    = 150

DEFAULT_W = {
    "m1": 0.15, "m2": 0.18, "m3": 0.10, "m4": 0.10,
    "m5": 0.10, "m6": 0.08, "m7": 0.07,
    "m8": 0.10, "m9": 0.07, "m10": 0.05,
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# ══════════════════════════════════════════════════════════════
# STEP 1 — Walk-forward backtest with default weights
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1 — Backtest (default weights, 4201-4471)")
print("="*60)
summary_default = run_backtest(
    train_end=4200, test_end=VAL_END,
    top_n=TOP_N, step=10, tier_group="1st",
    weights=DEFAULT_W
)
print(f"Default  hit_rate_any={summary_default['hit_rate_any']:.4f}  "
      f"hit_rate_1st={summary_default['hit_rate_1st']:.4f}  "
      f"lift={summary_default['lift_any']:.3f}x")

# ══════════════════════════════════════════════════════════════
# STEP 2 — Optuna weight tuning
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"STEP 2 — Optuna weight tuning ({N_TRIALS} trials, val {VAL_START}-{VAL_END})")
print("="*60)

print("Training all 10 models on draws 1-4400...")
models = {}
models["m1"]     = train_m1(TRAIN_END, tier_group="1st");   print("  M1 done")
models["m1_top3"]= train_m1(TRAIN_END, tier_group="top3")
models["m1_all"] = train_m1(TRAIN_END, tier_group="all")
models["m2"]     = train_m2(TRAIN_END);                      print("  M2 done")
models["m3"]     = train_m3(TRAIN_END);                      print("  M3 done")
models["m4"]     = train_m4(TRAIN_END, tier_group="1st");    print("  M4 done")
models["m5"]     = train_m5(TRAIN_END);                      print("  M5 done")
models["m6"]     = train_m6(TRAIN_END);                      print("  M6 done")
models["m7"]     = train_m7(TRAIN_END);                      print("  M7 done")
models["m8"]     = train_m8(TRAIN_END);                      print("  M8 done")
models["m9"]     = train_m9(TRAIN_END);                      print("  M9 done")
models["m10_1st"]= train_m10(TRAIN_END, tier_group="1st");   print("  M10 done")
print("All models trained.\n")

# Pre-score all 10k candidates (vectorised Optuna)
print("Pre-scoring 10,000 candidates...")
score_mat = np.zeros((10000, len(M_KEYS)))
for j, key in enumerate(M_KEYS):
    for i, num in enumerate(ALL_NUMBERS):
        if key == "m1":   score_mat[i,j] = models["m1"].get(num, 0.0)
        elif key == "m2": score_mat[i,j] = score_m2(models["m2"], num)
        elif key == "m3": score_mat[i,j] = score_m3(models["m3"], num)
        elif key == "m4": score_mat[i,j] = score_m4(models["m4"], num)
        elif key == "m5": score_mat[i,j] = score_m5(models["m5"], num, tier_group="1st")
        elif key == "m6": score_mat[i,j] = score_m6(models["m6"], num, tier_group="1st")
        elif key == "m7": score_mat[i,j] = score_m7(models["m7"], num)
        elif key == "m8": score_mat[i,j] = score_m8(models["m8"], num)
        elif key == "m9": score_mat[i,j] = score_m9(models["m9"], num)
        elif key == "m10":score_mat[i,j] = score_m10(models["m10_1st"], num)

nums_arr = np.array(ALL_NUMBERS)
print("Pre-scoring done.\n")

con = duckdb.connect(DB_PATH, read_only=True)
val_draws = con.execute(f"""
    SELECT draw_id, prize_1st FROM draws
    WHERE draw_id BETWEEN {VAL_START} AND {VAL_END}
    ORDER BY draw_id
""").fetchall()
con.close()
print(f"Val draws: {len(val_draws)}\n")

def evaluate(w_vec):
    w = np.array([w_vec[k] for k in M_KEYS])
    w = w / w.sum()
    combo = score_mat @ w
    top_idx = np.argpartition(combo, -TOP_N)[-TOP_N:]
    top_set = set(nums_arr[top_idx])
    return sum(1 for _, p1 in val_draws if p1 in top_set) / len(val_draws)

def objective(trial):
    w = {k: trial.suggest_float(k, 0.01, 1.0) for k in M_KEYS}
    return evaluate(w)

study = optuna.create_study(direction="maximize")
study.enqueue_trial(DEFAULT_W)
print(f"Running {N_TRIALS} Optuna trials...")
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

best_params = study.best_trial.params
total_w     = sum(best_params.values())
best_w      = {k: round(best_params[k] / total_w, 4) for k in M_KEYS}
default_val = evaluate(DEFAULT_W)
tuned_val   = study.best_trial.value

print(f"\nVal hit_rate — default: {default_val:.4f}  tuned: {tuned_val:.4f}")
print(f"Best weights: {best_w}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Full backtest with tuned weights
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3 — Full backtest with tuned weights (4201-4471)")
print("="*60)
summary_tuned = run_backtest(
    train_end=4200, test_end=VAL_END,
    top_n=TOP_N, step=10, tier_group="1st",
    weights=best_w
)
print(f"Tuned    hit_rate_any={summary_tuned['hit_rate_any']:.4f}  "
      f"hit_rate_1st={summary_tuned['hit_rate_1st']:.4f}  "
      f"lift={summary_tuned['lift_any']:.3f}x")

# ══════════════════════════════════════════════════════════════
# STEP 4 — Final predictions on ALL 5461 draws
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4 — Final predictions (trained on all 5461 draws)")
print("="*60)

con = duckdb.connect(DB_PATH, read_only=True)
max_draw  = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
last_row  = con.execute(
    "SELECT draw_id, draw_date, day_of_week FROM draws ORDER BY draw_id DESC LIMIT 1"
).fetchone()
con.close()
print(f"Last draw: #{last_row[0]} ({last_row[2]} {last_row[1]})")

# Choose best weights: tuned if better than default, else default
use_weights = best_w if summary_tuned['hit_rate_any'] >= summary_default['hit_rate_any'] else DEFAULT_W
print(f"Using {'tuned' if use_weights is best_w else 'default'} weights for final predictions")

print("Training all 10 models on full history...")
fm = {}
fm["m1"]     = train_m1(max_draw, tier_group="1st")
fm["m2"]     = train_m2(max_draw)
fm["m3"]     = train_m3(max_draw)
fm["m4"]     = train_m4(max_draw, tier_group="1st")
fm["m5"]     = train_m5(max_draw)
fm["m6"]     = train_m6(max_draw)
fm["m7"]     = train_m7(max_draw)
fm["m8"]     = train_m8(max_draw)
fm["m9"]     = train_m9(max_draw)
fm["m10_1st"]= train_m10(max_draw, tier_group="1st")
print("All 10 models trained on full history.")

# Score
print("Scoring 10,000 candidates with final models...")
final_scores = {}
for num in ALL_NUMBERS:
    w = use_weights
    s  = w["m1"]  * fm["m1"].get(num, 0.0)
    s += w["m2"]  * score_m2(fm["m2"], num)
    s += w["m3"]  * score_m3(fm["m3"], num)
    s += w["m4"]  * score_m4(fm["m4"], num)
    s += w["m5"]  * score_m5(fm["m5"], num, tier_group="1st")
    s += w["m6"]  * score_m6(fm["m6"], num, tier_group="1st")
    s += w["m7"]  * score_m7(fm["m7"], num)
    s += w["m8"]  * score_m8(fm["m8"], num)
    s += w["m9"]  * score_m9(fm["m9"], num)
    s += w["m10"] * score_m10(fm["m10_1st"], num)
    final_scores[num] = s

max_s = max(final_scores.values())
ranked = sorted(final_scores.items(), key=lambda x: -x[1])
top50  = ranked[:50]

draw_days   = ["Wed","Sat","Sun"]
last_dow    = last_row[2]
next_dow    = draw_days[(draw_days.index(last_dow)+1) % 3] if last_dow in draw_days else "Sat"

output = {
    "generated_at": datetime.now().isoformat(),
    "phase": 3,
    "models_used": 10,
    "weights_used": use_weights,
    "backtest_default": {"hit_rate_any": summary_default["hit_rate_any"],
                         "hit_rate_1st": summary_default["hit_rate_1st"],
                         "lift": summary_default["lift_any"]},
    "backtest_tuned":   {"hit_rate_any": summary_tuned["hit_rate_any"],
                         "hit_rate_1st": summary_tuned["hit_rate_1st"],
                         "lift": summary_tuned["lift_any"]},
    "next_draw": {"after": max_draw, "expected_day": next_dow},
    "top_50": [{"rank": i+1, "number": n, "score": round(s/max_s, 6)}
               for i,(n,s) in enumerate(top50)],
}

path = os.path.join(PRED_DIR, f"phase3_predictions_{ts}.json")
with open(path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print("PHASE 3 COMPLETE — TOP 20 PICKS")
print(f"{'='*60}")
for entry in output["top_50"][:20]:
    print(f"  #{entry['rank']:2d}  {entry['number']}   score={entry['score']:.4f}")
print(f"\nDefault backtest  → hit_rate_any={summary_default['hit_rate_any']:.4f}  1st={summary_default['hit_rate_1st']:.4f}  lift={summary_default['lift_any']:.3f}x")
print(f"Tuned   backtest  → hit_rate_any={summary_tuned['hit_rate_any']:.4f}  1st={summary_tuned['hit_rate_1st']:.4f}  lift={summary_tuned['lift_any']:.3f}x")
print(f"\nResults → {path}")
