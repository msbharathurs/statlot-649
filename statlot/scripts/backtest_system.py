"""
backtest_system.py — Walk-forward backtest for System 7 and System 8 predictors.
Separate training/testing suite from the 6-number system.

Splits by INDEX (draws sorted by draw_number, range 2341-4167):
  Iter 1: train[:1650]  test[1650:]  → ~177 test draws
  Iter 2: train[:1750]  test[1750:]  → ~77 test draws
  Iter 3: train[:1810]  test[1810:]  → ~17 test draws

Metric: P(3+ drawn numbers land inside our system set)
Triplets excluded. Doublets kept (45% historical rate).
Saves: results/backtest_system.json
"""

import sys, os, json, datetime
from collections import defaultdict
from math import comb

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from engine.candidate_gen_v2 import generate_candidates
from engine.features_v2 import build_features_batch
from engine.ensemble import EnsembleScorer
from engine.models.m1_bayes import BayesianFreqScorer
from engine.models.m2_ev_kelly import EVKellyScorer
from engine.models.m3_rf import RFScorer
from engine.models.m4_monte_carlo import MonteCarloScorer
from engine.models.m5_xgb import XGBScorer
from engine.models.m6_dqn import DQNAgent
from engine.models.m7_markov import MarkovScorer
from engine.models.m8_gmm import GMMScorer
from engine.models.m9_lstm import LSTMScorer
from engine.models.additional import AdditionalPredictor

DRAWS_PATH   = os.path.join(_ROOT, "draws_clean.csv")
RESULTS_DIR  = os.path.join(_ROOT, "results")
N_CANDIDATES = 20000
VAL_DRAWS    = 50
N_TRIALS     = 100

# Index-based splits (draws sorted by draw_number)
ITERATIONS = [
    {"name": "Iter1", "train_idx": 1650},
    {"name": "Iter2", "train_idx": 1750},
    {"name": "Iter3", "train_idx": 1810},
]


def load_draws():
    import csv
    draws = []
    with open(DRAWS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nums = sorted([int(row[f"n{i}"]) for i in range(1, 7)])
                add  = int(row["additional"]) if row.get("additional","").strip() else None
                draws.append({"draw_number": int(row["draw_number"]), "nums": nums, "additional": add})
            except:
                pass
    return sorted(draws, key=lambda x: x["draw_number"])


def has_triplet(nums):
    s = sorted(nums)
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            run += 1
            if run >= 3:
                return True
        else:
            run = 1
    return False


def build_number_scores(scored, top_k=5000):
    num_score = defaultdict(float)
    num_count = defaultdict(int)
    for combo, score in scored[:top_k]:
        for n in combo:
            num_score[int(n)] += score
            num_count[int(n)] += 1
    return {n: num_score.get(n, 0.0) / max(num_count.get(n, 0), 1) for n in range(1, 50)}


def select_system_numbers(num_scores, n_select, history):
    decades      = [(1,7),(8,14),(15,21),(22,28),(29,35),(36,42),(43,49)]
    ranked       = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
    pool         = [n for n, _ in ranked[:30]]
    recent       = [n for d in history[-20:] for n in d["nums"]]
    hot_set      = set(sorted(set(recent), key=recent.count, reverse=True)[:15])
    selected     = []
    used_decades = defaultdict(int)

    for _ in range(n_select):
        best_n, best_score = None, -1
        for n in pool:
            if n in selected:
                continue
            if has_triplet(sorted(selected + [n])):
                continue
            decade_idx = next((i for i,(lo,hi) in enumerate(decades) if lo<=n<=hi), 0)
            diversity_penalty = 0.3 * max(0, used_decades[decade_idx] - 1)
            hot_bonus = 0.05 if n in hot_set else 0
            adj_score = num_scores[n] - diversity_penalty + hot_bonus
            if adj_score > best_score:
                best_score = adj_score
                best_n = n
        if best_n:
            selected.append(best_n)
            decade_idx = next((i for i,(lo,hi) in enumerate(decades) if lo<=best_n<=hi), 0)
            used_decades[decade_idx] += 1

    return sorted(selected)


def coverage_stats(sys_nums, draws):
    stats = {3:0, 4:0, 5:0, 6:0}
    for draw in draws:
        ov = len(set(draw["nums"]) & set(sys_nums))
        for k in [3,4,5,6]:
            if ov >= k:
                stats[k] += 1
    n = max(len(draws), 1)
    return {k: round(v/n, 4) for k,v in stats.items()}


def theoretical_coverage(n_select):
    total = comb(49, n_select)
    return sum(comb(6,k)*comb(43,n_select-k)/total for k in range(3, min(7, n_select+1)))


def run_iteration(cfg, all_draws):
    name      = cfg["name"]
    t_idx     = cfg["train_idx"]

    train_draws = all_draws[:t_idx]
    test_draws  = all_draws[t_idx:]

    print(f"\n{chr(61)*60}")
    print(f"  {name}: train {len(train_draws)} draws "
          f"(dn {train_draws[0]['draw_number']}-{train_draws[-1]['draw_number']}) | "
          f"test {len(test_draws)} draws "
          f"(dn {test_draws[0]['draw_number']}-{test_draws[-1]['draw_number']})")
    print("="*60)

    suffix = f"_sys_{name.lower()}"

    print("[TRAIN] Training 9 models from scratch...")
    m1 = BayesianFreqScorer(); m1.fit(train_draws)
    m2 = EVKellyScorer();      m2.fit(train_draws)
    m3 = RFScorer();           m3.fit(train_draws, len(train_draws)); m3.save(suffix)
    m4 = MonteCarloScorer();   m4.fit(train_draws)
    m5 = XGBScorer();          m5.fit(train_draws, len(train_draws)); m5.save(suffix)
    m6 = DQNAgent();           m6.fit(train_draws, len(train_draws), n_episodes=2); m6.save(suffix)
    m7 = MarkovScorer();       m7.fit(train_draws)
    m8 = GMMScorer();          m8.fit(train_draws, len(train_draws)); m8.save(suffix)
    m9 = LSTMScorer(epochs=15);m9.fit(train_draws, len(train_draws)); m9.save(suffix)
    add_pred = AdditionalPredictor(); add_pred.fit(train_draws)
    print("[TRAIN] Done")

    print(f"[TUNE]  Tuning weights on last {VAL_DRAWS} train draws...")
    ensemble = EnsembleScorer()
    for nm, sc in [("m1",m1),("m2",m2),("m3",m3),("m4",m4),("m5",m5),
                   ("m6",m6),("m7",m7),("m8",m8),("m9",m9)]:
        ensemble.register(nm, sc)

    val_d   = train_draws[-VAL_DRAWS:]
    val_h   = train_draws[:-VAL_DRAWS]
    val_c   = generate_candidates(val_h, n_candidates=5000)
    add_bv  = add_pred.get_bias_scores(val_h, strength=0.15)
    ensemble.tune_weights(val_c, val_h, val_d, n_trials=N_TRIALS, seed=42, add_bias_scores=add_bv)
    top_w = {k:round(v,3) for k,v in sorted(ensemble.weights.items(), key=lambda x:x[1], reverse=True)[:5]}
    print(f"  Top weights: {top_w}")

    print(f"[GEN]   Generating {N_CANDIDATES} candidates...")
    candidates  = generate_candidates(train_draws, n_candidates=N_CANDIDATES)
    feat_matrix = build_features_batch(candidates, train_draws)
    add_bias    = add_pred.get_bias_scores(train_draws, strength=0.15)
    scored      = ensemble.score_batch(candidates, train_draws,
                                       feat_matrix=feat_matrix, add_bias_scores=add_bias)

    num_scores = build_number_scores(scored, top_k=5000)
    top15 = [n for n,_ in sorted(num_scores.items(), key=lambda x:x[1], reverse=True)[:15]]
    print(f"  Top 15 numbers: {top15}")

    s7 = select_system_numbers(num_scores, n_select=7, history=train_draws)
    s8 = select_system_numbers(num_scores, n_select=8, history=train_draws)

    s7_cov  = coverage_stats(s7, test_draws)
    s8_cov  = coverage_stats(s8, test_draws)
    s7_theo = theoretical_coverage(7)
    s8_theo = theoretical_coverage(8)

    print(f"\n  System 7: {s7}")
    print(f"    3+: {s7_cov[3]:.1%}  (theory {s7_theo:.1%})  lift: {s7_cov[3]/s7_theo:.2f}x  4+: {s7_cov[4]:.1%}  5+: {s7_cov[5]:.1%}")
    print(f"\n  System 8: {s8}")
    print(f"    3+: {s8_cov[3]:.1%}  (theory {s8_theo:.1%})  lift: {s8_cov[3]/s8_theo:.2f}x  4+: {s8_cov[4]:.1%}  5+: {s8_cov[5]:.1%}")

    return {
        "iteration":    name,
        "train_draws":  len(train_draws),
        "test_draws":   len(test_draws),
        "train_dn_range": [int(train_draws[0]["draw_number"]), int(train_draws[-1]["draw_number"])],
        "test_dn_range":  [int(test_draws[0]["draw_number"]),  int(test_draws[-1]["draw_number"])],
        "system7": {
            "numbers":    [int(x) for x in s7],
            "test_3plus": s7_cov[3], "test_4plus": s7_cov[4],
            "test_5plus": s7_cov[5], "test_6":     s7_cov[6],
            "theoretical_3plus": round(s7_theo, 4),
            "lift_3plus": round(s7_cov[3]/s7_theo, 3),
        },
        "system8": {
            "numbers":    [int(x) for x in s8],
            "test_3plus": s8_cov[3], "test_4plus": s8_cov[4],
            "test_5plus": s8_cov[5], "test_6":     s8_cov[6],
            "theoretical_3plus": round(s8_theo, 4),
            "lift_3plus": round(s8_cov[3]/s8_theo, 3),
        },
        "ensemble_weights": {k: round(v,3) for k,v in ensemble.weights.items()},
        "top15_numbers":    [int(n) for n in top15],
    }


def main():
    print("=" * 60)
    print("  SYSTEM 7 & 8 — WALK-FORWARD BACKTEST (3 ITERATIONS)")
    print("  Triplets excluded | Doublets kept | Index-based splits")
    print("=" * 60)

    all_draws = load_draws()
    print(f"Loaded {len(all_draws)} draws | dn range: {all_draws[0]['draw_number']}–{all_draws[-1]['draw_number']}")

    results = []
    for cfg in ITERATIONS:
        res = run_iteration(cfg, all_draws)
        results.append(res)

    print(f"\n{chr(61)*60}")
    print("  BACKTEST SUMMARY")
    print("="*60)
    for res in results:
        s7 = res["system7"]; s8 = res["system8"]
        print(f"  {res[iteration]} ({res[test_draws]} test draws) | "
              f"S7: {s7[test_3plus]:.1%} ({s7[lift_3plus]:.2f}x lift) | "
              f"S8: {s8[test_3plus]:.1%} ({s8[lift_3plus]:.2f}x lift)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "run_date": datetime.datetime.now(datetime.UTC).isoformat(),
        "triplet_exclusion": True,
        "doublet_exclusion": False,
        "notes": "Triplets excluded (5.1% of draws). Doublets kept (45.3% of draws).",
        "iterations": results,
    }
    out_path = os.path.join(RESULTS_DIR, "backtest_system.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("Next: run predict_system_final.py to train on all 1827 draws and generate final picks.")


if __name__ == "__main__":
    main()
