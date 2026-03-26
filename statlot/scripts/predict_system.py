"""
predict_system.py — System 7 and System 8 predictor for Singapore TOTO
Strategy: 
  - Score ALL individual numbers 1-49 using ensemble models (marginal frequency)
  - Build a per-number "value" score = how often each number appears in top-scored combos
  - Select top-7 / top-8 by value score + diversity (avoid clustering)
  - Validate: compute P(3+ drawn numbers in our set) via historical simulation

Usage: python3 scripts/predict_system.py --draw_label "Draw 4168"
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

DRAWS_PATH  = os.path.join(_ROOT, "draws_clean.csv")
RESULTS_DIR = os.path.join(_ROOT, "results")
SUFFIX      = "_prod"
N_CANDIDATES = 20000

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

def build_number_scores(scored, top_k=3000):
    """Aggregate per-number score from top-k combos weighted by combo score."""
    num_score = defaultdict(float)
    num_count = defaultdict(int)
    for combo, score in scored[:top_k]:
        for n in combo:
            num_score[int(n)] += score
            num_count[int(n)] += 1
    # Normalize by frequency in pool
    result = {}
    for n in range(1, 50):
        count = num_count.get(n, 0)
        raw   = num_score.get(n, 0.0)
        result[n] = raw / max(count, 1)  # average score when this number appears
    return result

def select_system_numbers(num_scores, n_select, history, top_pct=0.5):
    """
    Greedy + diversity selection:
    Pick n_select numbers that maximise:
      - High individual ensemble score
      - Cover spread across different frequency/decade bands
      - Avoid all-hot or all-cold clustering
    """
    # Sort candidates by score
    ranked = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
    pool = [n for n, _ in ranked[:25]]  # top 25 by score as candidate pool
    
    # Compute recent hot numbers (last 20 draws)
    recent = [n for d in history[-20:] for n in d["nums"]]
    hot_set = set(sorted(set(recent), key=recent.count, reverse=True)[:15])
    
    # Compute decade distribution of historical draws (want representation)
    # Decades: 1-7,8-14,15-21,22-28,29-35,36-42,43-49
    decades = [(1,7),(8,14),(15,21),(22,28),(29,35),(36,42),(43,49)]
    
    selected = []
    used_decades = defaultdict(int)
    
    for _ in range(n_select):
        best_n, best_score = None, -1
        for n in pool:
            if n in selected: continue
            decade_idx = next((i for i,(lo,hi) in enumerate(decades) if lo<=n<=hi), 0)
            # Penalise if this decade already has 2+ numbers (diversity)
            diversity_penalty = 0.3 * max(0, used_decades[decade_idx] - 1)
            # Bonus if number is in hot set
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

def backtest_system(system_nums, history, val_draws=100):
    """
    Simulate: for last val_draws, check if 3+ of drawn numbers are in our system set.
    This is the key metric for system entries.
    """
    hits = 0
    n = len(system_nums)
    for draw in history[-val_draws:]:
        drawn = set(draw["nums"])
        overlap = len(drawn & set(system_nums))
        if overlap >= 3:
            hits += 1
    
    hit_rate = hits / val_draws
    
    # Theoretical baseline
    total_n = comb(49, n)
    p_3plus = sum(comb(6, k) * comb(43, n-k) / total_n for k in range(3, min(7, n+1)))
    
    return hit_rate, p_3plus, hits, val_draws

def main():
    draw_label = "Next Draw"
    args = sys.argv[1:]
    if "--draw_label" in args:
        draw_label = args[args.index("--draw_label") + 1]

    print("=" * 60)
    print(f"  SYSTEM 7 & 8 PREDICTION — {draw_label}")
    print("=" * 60)

    draws = load_draws()
    print(f"Loaded {len(draws)} draws")

    history = draws
    train_end_idx = len(draws)
    suffix = SUFFIX

    # Load/train models (reuse prod suffix if already trained same session, else train fresh)
    print(f"\n[TRAIN] Training models on {train_end_idx} draws...")
    m1 = BayesianFreqScorer(); m1.fit(draws)
    m2 = EVKellyScorer();      m2.fit(draws)
    m3 = RFScorer();           m3.fit(draws, train_end_idx); m3.save(suffix)
    m4 = MonteCarloScorer();   m4.fit(draws)
    m5 = XGBScorer();          m5.fit(draws, train_end_idx); m5.save(suffix)
    m6 = DQNAgent();           m6.fit(draws, train_end_idx, n_episodes=2); m6.save(suffix)
    m7 = MarkovScorer();       m7.fit(draws)
    m8 = GMMScorer();          m8.fit(draws, train_end_idx); m8.save(suffix)
    m9 = LSTMScorer(epochs=15);m9.fit(draws, train_end_idx); m9.save(suffix)
    add_pred = AdditionalPredictor(); add_pred.fit(draws)
    print("[TRAIN] Done ✓")

    # Ensemble
    print(f"\n[ENSEMBLE] Loading tuned weights...")
    ensemble = EnsembleScorer()
    for name, scorer in [("m1",m1),("m2",m2),("m3",m3),("m4",m4),("m5",m5),
                          ("m6",m6),("m7",m7),("m8",m8),("m9",m9)]:
        ensemble.register(name, scorer)

    val_draws   = draws[-50:]
    val_history = draws[:-50]
    val_candidates = generate_candidates(val_history, n_candidates=5000)
    add_bias_scores = add_pred.get_bias_scores(val_history, strength=0.15)
    ensemble.tune_weights(val_candidates, val_history, val_draws,
                          n_trials=100, seed=42, add_bias_scores=add_bias_scores)

    # Generate + score candidates
    print(f"\n[GEN] Generating {N_CANDIDATES} candidates...")
    candidates = generate_candidates(history, n_candidates=N_CANDIDATES)
    feat_matrix = build_features_batch(candidates, history)
    add_bias    = add_pred.get_bias_scores(history, strength=0.15)
    scored      = ensemble.score_batch(candidates, history,
                                       feat_matrix=feat_matrix, add_bias_scores=add_bias)
    print(f"  Scored {len(scored)} candidates")

    # Build per-number scores
    print(f"\n[ANALYZE] Building per-number value scores...")
    num_scores = build_number_scores(scored, top_k=5000)
    
    # Top 15 numbers by score
    top15 = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"  Top 15 numbers: {[n for n,_ in top15]}")

    # Select System 7
    s7 = select_system_numbers(num_scores, n_select=7, history=history)
    # Select System 8
    s8 = select_system_numbers(num_scores, n_select=8, history=history)

    # Backtest each
    s7_hit, s7_theo, s7_hits, s7_val = backtest_system(s7, history, val_draws=200)
    s8_hit, s8_theo, s8_hits, s8_val = backtest_system(s8, history, val_draws=200)

    # Also compute P(4+) and P(5+) for transparency
    def coverage_stats(sys_nums, history, val=200):
        stats = {3:0, 4:0, 5:0, 6:0}
        for draw in history[-val:]:
            ov = len(set(draw["nums"]) & set(sys_nums))
            for k in [3,4,5,6]:
                if ov >= k: stats[k] += 1
        return {k: v/val for k,v in stats.items()}

    s7_stats = coverage_stats(s7, history)
    s8_stats = coverage_stats(s8, history)

    # Print results
    print(f"\n{'='*60}")
    print(f"  SYSTEM 7 — {draw_label}")
    print(f"{'='*60}")
    print(f"  Numbers: {s7}")
    print(f"  Embedded combos: C(7,6) = {comb(7,6)} ordinary entries")
    print()
    print(f"  Historical (last 200 draws):")
    print(f"    3+ coverage: {s7_stats[3]:.1%}  (theory: {s7_theo:.1%})")
    print(f"    4+ coverage: {s7_stats[4]:.1%}")
    print(f"    5+ coverage: {s7_stats[5]:.1%}")
    print(f"    6  coverage: {s7_stats[6]:.1%}  ← jackpot coverage")
    print(f"  Lift vs random 3+: {s7_stats[3]/s7_theo:.2f}x")

    print(f"\n{'='*60}")
    print(f"  SYSTEM 8 — {draw_label}")
    print(f"{'='*60}")
    print(f"  Numbers: {s8}")
    print(f"  Embedded combos: C(8,6) = {comb(8,6)} ordinary entries")
    print()
    print(f"  Historical (last 200 draws):")
    print(f"    3+ coverage: {s8_stats[3]:.1%}  (theory: {s8_theo:.1%})")
    print(f"    4+ coverage: {s8_stats[4]:.1%}")
    print(f"    5+ coverage: {s8_stats[5]:.1%}")
    print(f"    6  coverage: {s8_stats[6]:.1%}  ← jackpot coverage")
    print(f"  Lift vs random 3+: {s8_stats[3]/s8_theo:.2f}x")
    print(f"{'='*60}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "draw_label": draw_label,
        "generated_date": datetime.datetime.now(datetime.UTC).isoformat(),
        "train_draws": int(train_end_idx),
        "system7": {
            "numbers": [int(x) for x in s7],
            "embedded_combos": comb(7,6),
            "hist_3plus_rate": round(s7_stats[3], 4),
            "hist_4plus_rate": round(s7_stats[4], 4),
            "hist_5plus_rate": round(s7_stats[5], 4),
            "hist_6_rate": round(s7_stats[6], 4),
            "theoretical_3plus": round(s7_theo, 4),
        },
        "system8": {
            "numbers": [int(x) for x in s8],
            "embedded_combos": comb(8,6),
            "hist_3plus_rate": round(s8_stats[3], 4),
            "hist_4plus_rate": round(s8_stats[4], 4),
            "hist_5plus_rate": round(s8_stats[5], 4),
            "hist_6_rate": round(s8_stats[6], 4),
            "theoretical_3plus": round(s8_theo, 4),
        },
        "top15_numbers": [int(n) for n,_ in top15],
    }
    out_path = os.path.join(RESULTS_DIR, "system_prediction.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
