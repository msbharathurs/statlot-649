"""
predict_final.py — Production prediction generator
Trains on ALL draws, Optuna val on last 50, generates 10 tickets for next draw.
"""
import sys, os, json, datetime

# PYTHONPATH fix — ensure engine imports work from scripts/ subdir
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from engine.candidate_gen_v2 import generate_candidates
from engine.features_v2 import build_features_batch
from engine.ensemble import EnsembleScorer
from engine.diversity_select import select_diverse_tickets
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
N_TICKETS    = 10

def load_draws():
    import csv
    draws = []
    with open(DRAWS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nums = sorted([int(row[f"n{i}"]) for i in range(1, 7)])
                add  = int(row["additional"]) if row.get("additional","").strip() else None
                draws.append({
                    "draw_number": int(row["draw_number"]),
                    "nums": nums,
                    "additional": add,
                })
            except Exception as e:
                pass
    return sorted(draws, key=lambda x: x["draw_number"])

def main():
    draw_label = "Next Draw"
    args = sys.argv[1:]
    if "--draw_label" in args:
        idx = args.index("--draw_label")
        draw_label = args[idx + 1]

    print("=" * 60)
    print(f"  PRODUCTION PREDICTION — {draw_label}")
    print("=" * 60)

    draws = load_draws()
    print(f"Loaded {len(draws)} draws ({draws[0]['draw_number']} → {draws[-1]['draw_number']})")

    history       = draws
    train_end_idx = len(draws)
    suffix        = SUFFIX

    # --- Train all 9 models ---
    print(f"\n[TRAIN] Training all 9 models on {train_end_idx} draws...")
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
    print("[TRAIN] All models trained ✓")

    # --- Tune ensemble on last 50 draws ---
    print(f"\n[ENSEMBLE] Tuning weights on last 50 draws...")
    ensemble = EnsembleScorer()
    for name, scorer in [("m1",m1),("m2",m2),("m3",m3),("m4",m4),("m5",m5),
                          ("m6",m6),("m7",m7),("m8",m8),("m9",m9)]:
        ensemble.register(name, scorer)

    val_draws       = draws[-50:]
    val_history     = draws[:-50]
    val_candidates  = generate_candidates(val_history, n_candidates=5000)
    add_bias_scores = add_pred.get_bias_scores(val_history, strength=0.15)
    ensemble.tune_weights(val_candidates, val_history, val_draws,
                          n_trials=100, seed=42, add_bias_scores=add_bias_scores)
    ensemble.save(suffix)

    # --- Generate + score candidates ---
    print(f"\n[GEN] Generating {N_CANDIDATES} candidates...")
    candidates = generate_candidates(history, n_candidates=N_CANDIDATES)
    print(f"  Candidates: {len(candidates)}")

    print(f"[SCORE] Scoring...")
    feat_matrix = build_features_batch(candidates, history)
    add_bias    = add_pred.get_bias_scores(history, strength=0.15)
    scored      = ensemble.score_batch(candidates, history,
                                       feat_matrix=feat_matrix, add_bias_scores=add_bias)

    # --- 10 tickets: 2 diverse passes of 5 ---
    print(f"[SELECT] Selecting {N_TICKETS} tickets...")
    predicted_add = add_pred.predict(history, top_n=7)
    tickets_a     = select_diverse_tickets(scored, n_tickets=5, bonus_candidates=predicted_add)
    used          = set(tuple(t) for t in tickets_a)
    scored_rest   = [(c, s) for c, s in scored if tuple(c) not in used]
    tickets_b     = select_diverse_tickets(scored_rest, n_tickets=5, bonus_candidates=predicted_add)
    all_tickets   = tickets_a + tickets_b

    # --- Output ---
    print(f"\n{'='*60}")
    print(f"  FINAL {N_TICKETS} TICKETS FOR {draw_label}")
    print(f"{'='*60}")
    for i, t in enumerate(all_tickets, 1):
        print(f"  T{i:02d}: {sorted(t)}")
    print(f"\n  Predicted additional (top 5): {sorted(predicted_add[:5])}")
    top5w = sorted(ensemble.weights.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top ensemble weights: { {k: round(v,3) for k,v in top5w} }")
    print(f"{'='*60}")

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "draw_label": draw_label,
        "generated_date": datetime.datetime.utcnow().isoformat(),
        "train_draws": train_end_idx,
        "tickets": [sorted(t) for t in all_tickets],
        "predicted_additional": sorted(predicted_add[:5]),
        "ensemble_weights": ensemble.weights,
    }
    # Convert any numpy types to native Python
    import numpy as np
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        raise TypeError(f"Not serializable: {type(o)}")
    out_path = os.path.join(RESULTS_DIR, "final_prediction.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=convert)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
