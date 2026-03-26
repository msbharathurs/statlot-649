"""
predict_system_final.py — Final System 7 & System 8 prediction after backtest.

Trains on ALL 1827 draws, tunes weights on last 50, applies backtest learnings.
Triplets excluded. Doublets kept (45.3% historical rate — too common to exclude).
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
SUFFIX       = "_sysfinal"


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
            except: pass
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
    decades     = [(1,7),(8,14),(15,21),(22,28),(29,35),(36,42),(43,49)]
    ranked      = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
    pool        = [n for n, _ in ranked[:30]]
    recent      = [n for d in history[-20:] for n in d["nums"]]
    hot_set     = set(sorted(set(recent), key=recent.count, reverse=True)[:15])
    selected    = []
    used_decades = defaultdict(int)

    for _ in range(n_select):
        best_n, best_score = None, -1
        for n in pool:
            if n in selected: continue
            if has_triplet(sorted(selected + [n])): continue
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


def coverage_stats(sys_nums, draws, val=200):
    stats = {3:0, 4:0, 5:0, 6:0}
    for draw in draws[-val:]:
        ov = len(set(draw["nums"]) & set(sys_nums))
        for k in [3,4,5,6]:
            if ov >= k: stats[k] += 1
    return {k: round(v/val, 4) for k,v in stats.items()}


def theoretical_coverage(n_select):
    total = comb(49, n_select)
    return sum(comb(6,k)*comb(43, n_select-k)/total for k in range(3, min(7, n_select+1)))


def main():
    draw_label = "Draw 4168"
    args = sys.argv[1:]
    if "--draw_label" in args:
        draw_label = args[args.index("--draw_label") + 1]

    print("=" * 60)
    print(f"  FINAL SYSTEM 7 & 8 — {draw_label}")
    print(f"  Trained on ALL draws | Triplets excluded | Doublets kept")
    print("=" * 60)

    draws = load_draws()
    print(f"Loaded {len(draws)} draws")

    # Train all models on full history
    print(f"\n[TRAIN] Training on {len(draws)} draws...")
    m1 = BayesianFreqScorer(); m1.fit(draws)
    m2 = EVKellyScorer();      m2.fit(draws)
    m3 = RFScorer();           m3.fit(draws, len(draws)); m3.save(SUFFIX)
    m4 = MonteCarloScorer();   m4.fit(draws)
    m5 = XGBScorer();          m5.fit(draws, len(draws)); m5.save(SUFFIX)
    m6 = DQNAgent();           m6.fit(draws, len(draws), n_episodes=2); m6.save(SUFFIX)
    m7 = MarkovScorer();       m7.fit(draws)
    m8 = GMMScorer();          m8.fit(draws, len(draws)); m8.save(SUFFIX)
    m9 = LSTMScorer(epochs=15);m9.fit(draws, len(draws)); m9.save(SUFFIX)
    add_pred = AdditionalPredictor(); add_pred.fit(draws)
    print("[TRAIN] Done ✓")

    # Tune weights
    ensemble = EnsembleScorer()
    for nm, sc in [("m1",m1),("m2",m2),("m3",m3),("m4",m4),("m5",m5),
                   ("m6",m6),("m7",m7),("m8",m8),("m9",m9)]:
        ensemble.register(nm, sc)

    val_draws   = draws[-50:]
    val_history = draws[:-50]
    val_cands   = generate_candidates(val_history, n_candidates=5000)
    add_bias_v  = add_pred.get_bias_scores(val_history, strength=0.15)
    ensemble.tune_weights(val_cands, val_history, val_draws, n_trials=100, seed=42,
                          add_bias_scores=add_bias_v)
    weights_top = sorted(ensemble.weights.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top weights: { {k:round(v,3) for k,v in weights_top} }")

    # Score candidates
    print(f"\n[GEN]  Generating {N_CANDIDATES} candidates...")
    candidates  = generate_candidates(draws, n_candidates=N_CANDIDATES)
    feat_matrix = build_features_batch(candidates, draws)
    add_bias    = add_pred.get_bias_scores(draws, strength=0.15)
    scored      = ensemble.score_batch(candidates, draws, feat_matrix=feat_matrix,
                                       add_bias_scores=add_bias)

    # Number scores
    num_scores = build_number_scores(scored, top_k=5000)
    top15 = [n for n,_ in sorted(num_scores.items(), key=lambda x: x[1], reverse=True)[:15]]
    print(f"  Top 15 numbers: {top15}")

    # Select
    s7 = select_system_numbers(num_scores, n_select=7, history=draws)
    s8 = select_system_numbers(num_scores, n_select=8, history=draws)

    # Historical validation (last 200)
    s7_cov  = coverage_stats(s7, draws, val=200)
    s8_cov  = coverage_stats(s8, draws, val=200)
    s7_theo = theoretical_coverage(7)
    s8_theo = theoretical_coverage(8)

    # Print
    print(f"\n{'='*60}")
    print(f"  SYSTEM 7 — {draw_label}")
    print(f"{'='*60}")
    print(f"  Numbers : {s7}")
    print(f"  C(7,6)  = {comb(7,6)} embedded ordinary combos")
    print(f"  3+ cov  : {s7_cov[3]:.1%}  (random: {s7_theo:.1%})  Lift: {s7_cov[3]/s7_theo:.2f}x")
    print(f"  4+ cov  : {s7_cov[4]:.1%}    5+ cov: {s7_cov[5]:.1%}")

    print(f"\n{'='*60}")
    print(f"  SYSTEM 8 — {draw_label}")
    print(f"{'='*60}")
    print(f"  Numbers : {s8}")
    print(f"  C(8,6)  = {comb(8,6)} embedded ordinary combos")
    print(f"  3+ cov  : {s8_cov[3]:.1%}  (random: {s8_theo:.1%})  Lift: {s8_cov[3]/s8_theo:.2f}x")
    print(f"  4+ cov  : {s8_cov[4]:.1%}    5+ cov: {s8_cov[5]:.1%}")
    print(f"{'='*60}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "draw_label":    draw_label,
        "generated_date": datetime.datetime.now(datetime.UTC).isoformat(),
        "train_draws":   len(draws),
        "triplet_exclusion": True,
        "doublet_exclusion": False,
        "system7": {
            "numbers":        [int(x) for x in s7],
            "embedded_combos": comb(7,6),
            "hist_3plus_200": s7_cov[3],
            "hist_4plus_200": s7_cov[4],
            "hist_5plus_200": s7_cov[5],
            "theoretical_3plus": round(s7_theo, 4),
            "lift_3plus": round(s7_cov[3]/s7_theo, 3),
        },
        "system8": {
            "numbers":        [int(x) for x in s8],
            "embedded_combos": comb(8,6),
            "hist_3plus_200": s8_cov[3],
            "hist_4plus_200": s8_cov[4],
            "hist_5plus_200": s8_cov[5],
            "theoretical_3plus": round(s8_theo, 4),
            "lift_3plus": round(s8_cov[3]/s8_theo, 3),
        },
        "ensemble_weights": {k: round(v,3) for k,v in ensemble.weights.items()},
        "top15_numbers": [int(n) for n in top15],
    }
    with open(os.path.join(RESULTS_DIR, "system_prediction_final.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → results/system_prediction_final.json")


if __name__ == "__main__":
    main()
