"""
backtest_sys8.py — Full walk-forward backtest for System 8 (8 numbers per ticket).
COMPLETELY SEPARATE from 6num and sys7.

Same architecture as sys7 but:
  - N_SELECT = 8 (8 numbers per ticket)
  - MAX_SHARED = 3 (slightly more sharing ok — 8 numbers, 49 range)
  - Dual-grid rules further relaxed for 8 numbers
  - T9 bonus slot: bonus candidate appended as 9th number to T5
  - All metrics, SHAP, separate prod models: saved_models/sys8/

Results: results/backtest_sys8.json
"""

import sys, os, json, datetime, pickle
from collections import defaultdict
from math import comb

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from engine.candidate_gen_v2 import generate_candidates
from engine.features_v2 import build_features_batch
from engine.ensemble import EnsembleScorer
from engine.diversity_select import select_diverse_tickets as diversity_select
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
MODELS_DIR   = os.path.join(_ROOT, "saved_models", "sys8")
N_CANDIDATES = 20000
VAL_DRAWS    = 50
N_TRIALS     = 100
POOL_SIZE    = 500
MAX_SHARED   = 3
LAMBDA       = 0.3
ADD_BIAS     = 0.15
N_TICKETS    = 5
N_SELECT     = 8
N_EPISODES   = 2
LSTM_EPOCHS  = 15

ITERATIONS = [
    {"name": "Iter1", "train_idx": 1650},
    {"name": "Iter2", "train_idx": 1750},
    {"name": "Iter3", "train_idx": 1810},
]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def grid_a_row(n):
    for i,(lo,hi) in enumerate(GRID_A_ROWS):
        if lo<=n<=hi: return i
    return -1

def grid_b_row(n):
    for i,(lo,hi) in enumerate(GRID_B_ROWS):
        if lo<=n<=hi: return i
    return -1


def dual_grid_ok_sys8(nums):
    """
    Strict dual-grid elimination for System 8 — same logic as 6num should_eliminate(),
    adapted for 8 numbers. Same physical ticket, same grid, same distribution rules.
    sum range: 8num * (100/6) = 134 min, 8num * (210/6) = 280 max.
    """
    nums = sorted(nums)
    rA = [rowA(n) for n in nums]; cA = [colA(n) for n in nums]
    rB = [rowB(n) for n in nums]; cB = [colB(n) for n in nums]
    s = sum(nums)
    consec = sum(1 for i in range(len(nums)-1) if nums[i+1]-nums[i]==1)
    if len(set(rA)) <= 2 or len(set(cA)) <= 2: return False
    if (max(rA)-min(rA)) <= 1 or (max(cA)-min(cA)) <= 3: return False
    if len(set(rB)) <= 2 or len(set(cB)) <= 2: return False
    if (max(rB)-min(rB)) <= 1 or (max(cB)-min(cB)) <= 3: return False
    if sum(1 for n in nums if n >= 46) >= 4: return False
    if sum(1 for n in nums if n >= 43) >= 5: return False
    if all(n % 2 != 0 for n in nums) or all(n % 2 == 0 for n in nums): return False
    if s < 134 or s > 280: return False
    if consec >= 4: return False
    return True

def rowA(n): return (n - 1) // 9 + 1
def colA(n): return (n - 1) % 9 + 1
def rowB(n): return (n - 1) // 7 + 1
def colB(n): return (n - 1) % 7 + 1

def has_triplet(nums):
    s = sorted(nums)
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]+1:
            run += 1
            if run >= 3: return True
        else: run = 1
    return False


def ticket_matches(ticket_nums, draw_nums, draw_add):
    main  = len(set(ticket_nums) & set(draw_nums))
    bonus = 1 if draw_add and draw_add in ticket_nums else 0
    return main, bonus


def evaluate_tickets(tickets, test_draws):
    counters = {k: 0 for k in ["3plus","4plus","5plus","jackpot","3plus_bonus","4plus_bonus"]}
    n = len(test_draws)
    for draw in test_draws:
        best_main = 0; best_bonus = 0
        for t in tickets:
            m, b = ticket_matches(t, draw["nums"], draw["additional"])
            if m > best_main or (m == best_main and b > best_bonus):
                best_main = m; best_bonus = b
        if best_main >= 3: counters["3plus"] += 1
        if best_main >= 4: counters["4plus"] += 1
        if best_main >= 5: counters["5plus"] += 1
        if best_main == 6: counters["jackpot"] += 1
        if best_main >= 3 and best_bonus == 1: counters["3plus_bonus"] += 1
        if best_main >= 4 and best_bonus == 1: counters["4plus_bonus"] += 1
    return {k: round(v/max(n,1), 4) for k,v in counters.items()}


def theoretical_rate(n_select, match_k):
    total = comb(49, n_select)
    return sum(comb(6,k)*comb(43,n_select-k)/total for k in range(match_k, min(7,n_select+1)))


def run_shap_analysis(m5, candidates, train_draws, suffix, n_sample=200):
    try:
        import shap
        feat_matrix = build_features_batch(candidates[:n_sample], train_draws)
        explainer   = shap.TreeExplainer(m5.model)
        shap_vals   = explainer.shap_values(feat_matrix)
        mean_abs    = np.abs(shap_vals).mean(axis=0)
        feat_names  = [f"feat_{i}" for i in range(feat_matrix.shape[1])]
        importance  = sorted(zip(feat_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)[:20]
        out = os.path.join(RESULTS_DIR, f"shap_sys8_{suffix}.json")
        with open(out, "w") as f:
            json.dump({"top20_features": importance}, f, indent=2)
        print(f"  [SHAP] Saved → {out}")
    except Exception as e:
        print(f"  [SHAP] Skipped: {e}")


def run_iteration(cfg, all_draws):
    name   = cfg["name"]
    t_idx  = cfg["train_idx"]
    suffix = name.lower()

    train_draws = all_draws[:t_idx]
    test_draws  = all_draws[t_idx:]

    print(f"\n{'='*60}")
    print(f"  SYS8 {name}: train={len(train_draws)} "
          f"(dn {train_draws[0]['draw_number']}-{train_draws[-1]['draw_number']}) | "
          f"test={len(test_draws)} "
          f"(dn {test_draws[0]['draw_number']}-{test_draws[-1]['draw_number']})")
    print(f"{'='*60}")

    print("[TRAIN] Training 9 models from scratch...")
    m1 = BayesianFreqScorer(); m1.fit(train_draws)
    m2 = EVKellyScorer();      m2.fit(train_draws)
    m3 = RFScorer();           m3.fit(train_draws, len(train_draws)); m3.save(f"_sys8_{suffix}")
    m4 = MonteCarloScorer();   m4.fit(train_draws)
    m5 = XGBScorer();          m5.fit(train_draws, len(train_draws)); m5.save(f"_sys8_{suffix}")
    m6 = DQNAgent();           m6.fit(train_draws, len(train_draws), n_episodes=N_EPISODES); m6.save(f"_sys8_{suffix}")
    m7 = MarkovScorer();       m7.fit(train_draws)
    m8 = GMMScorer();          m8.fit(train_draws, len(train_draws)); m8.save(f"_sys8_{suffix}")
    m9 = LSTMScorer(epochs=LSTM_EPOCHS); m9.fit(train_draws, len(train_draws)); m9.save(f"_sys8_{suffix}")
    add_pred = AdditionalPredictor(); add_pred.fit(train_draws)
    print("[TRAIN] Done ✓")

    print("[SHAP]  Running SHAP on XGBoost...")
    shap_candidates = generate_candidates(train_draws, n_candidates=300)
    run_shap_analysis(m5, shap_candidates, train_draws, suffix)

    print(f"[TUNE]  Tuning ensemble weights ({N_TRIALS} trials, {VAL_DRAWS} val draws)...")
    ensemble = EnsembleScorer()
    for nm, sc in [("m1",m1),("m2",m2),("m3",m3),("m4",m4),("m5",m5),
                   ("m6",m6),("m7",m7),("m8",m8),("m9",m9)]:
        ensemble.register(nm, sc)

    val_draws = train_draws[-VAL_DRAWS:]
    val_hist  = train_draws[:-VAL_DRAWS]
    val_cands = generate_candidates(val_hist, n_candidates=5000)
    add_bv    = add_pred.get_bias_scores(val_hist, strength=ADD_BIAS)
    ensemble.tune_weights(val_cands, val_hist, val_draws,
                          n_trials=N_TRIALS, seed=42, add_bias_scores=add_bv)
    top_w = sorted(ensemble.weights.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top weights: { {k:round(v,3) for k,v in top_w} }")

    wpath = os.path.join(MODELS_DIR, f"ensemble_weights_{suffix}.pkl")
    with open(wpath, "wb") as f:
        pickle.dump(ensemble.weights, f)

    print(f"[GEN]   Generating {N_CANDIDATES} candidates...")
    candidates  = generate_candidates(train_draws, n_candidates=N_CANDIDATES)
    feat_matrix = build_features_batch(candidates, train_draws)
    add_bias    = add_pred.get_bias_scores(train_draws, strength=ADD_BIAS)
    scored      = ensemble.score_batch(candidates, train_draws,
                                       feat_matrix=feat_matrix, add_bias_scores=add_bias)

    print("[FILTER] Applying triplet filter...")
    filtered = [(c, s) for c, s in scored if not has_triplet(list(c))]
    print(f"  After triplet filter: {len(filtered)} combos")

    print(f"[SELECT] diversity_select: pool={POOL_SIZE}, MAX_SHARED={MAX_SHARED}, LAMBDA={LAMBDA}...")
    pool      = filtered[:POOL_SIZE]
    tickets_6 = diversity_select(pool, n_tickets=N_TICKETS, max_shared=MAX_SHARED, lambda_=LAMBDA)
    print(f"  Selected {len(tickets_6)} base tickets (6-num each)")

    # Build per-number scores
    num_score = defaultdict(float)
    num_count = defaultdict(int)
    for combo, score in filtered[:5000]:
        for n in combo:
            num_score[int(n)] += score
            num_count[int(n)] += 1
    num_scores = {n: num_score[n]/max(num_count[n],1) for n in range(1,50)}

    # Expand each 6-num ticket to 8-num (add 2 more numbers)
    tickets_8 = []
    for base in tickets_6:
        used = set(base)
        # Add 7th number
        cands_7 = sorted([(n, num_scores.get(n,0)) for n in range(1,50) if n not in used],
                          key=lambda x: x[1], reverse=True)
        n7 = None
        for n, _ in cands_7:
            c7 = sorted(list(used)+[n])
            if not has_triplet(c7):
                n7 = n; break
        if n7 is None: n7 = cands_7[0][0]
        used.add(n7)
        # Add 8th number
        cands_8 = sorted([(n, num_scores.get(n,0)) for n in range(1,50) if n not in used],
                          key=lambda x: x[1], reverse=True)
        n8 = None
        for n, _ in cands_8:
            c8 = sorted(list(used)+[n])
            if not has_triplet(c8) and dual_grid_ok_sys8(c8):
                n8 = n; break
        if n8 is None:
            for n, _ in cands_8:
                c8 = sorted(list(used)+[n])
                if not has_triplet(c8):
                    n8 = n; break
        if n8 is None: n8 = cands_8[0][0]
        used.add(n8)
        tickets_8.append(sorted(list(used)))

    # T9 bonus slot
    bonus_scores = add_pred.get_bias_scores(train_draws, strength=1.0)
    bonus_sorted = sorted(bonus_scores.items(), key=lambda x: x[1], reverse=True)
    bonus_num = None
    if len(tickets_8) >= N_TICKETS:
        for bn, _ in bonus_sorted:
            bn = int(bn)
            cand = sorted(tickets_8[N_TICKETS-1] + [bn])
            if not has_triplet(cand):
                bonus_num = bn; break

    tickets_final = tickets_8[:N_TICKETS]
    bonus_ticket  = None
    if bonus_num:
        bonus_ticket = sorted(tickets_final[N_TICKETS-1] + [bonus_num])
        print(f"  T9 bonus slot: {bonus_ticket} (bonus num: {bonus_num})")

    print(f"\n  Final {N_TICKETS} System-8 tickets:")
    for i, t in enumerate(tickets_final):
        print(f"    T{i+1}: {t}")

    stats = evaluate_tickets(tickets_final, test_draws)
    bonus_stats = {}
    if bonus_ticket:
        all_t = tickets_final[:-1] + [bonus_ticket]
        bonus_stats = evaluate_tickets(all_t, test_draws)

    theo_3 = theoretical_rate(N_SELECT, 3)
    theo_4 = theoretical_rate(N_SELECT, 4)
    theo_5 = theoretical_rate(N_SELECT, 5)

    print(f"\n  === SYS8 {name} RESULTS ({len(test_draws)} test draws) ===")
    print(f"  3+:      {stats['3plus']:.1%}  (theory {theo_3:.1%})  lift: {stats['3plus']/theo_3:.2f}x")
    print(f"  4+:      {stats['4plus']:.1%}  (theory {theo_4:.1%})  lift: {stats['4plus']/max(theo_4,0.0001):.2f}x")
    print(f"  5+:      {stats['5plus']:.1%}")
    print(f"  Jackpot: {stats['jackpot']:.1%}")
    print(f"  3+bonus: {stats['3plus_bonus']:.1%}")
    print(f"  4+bonus: {stats['4plus_bonus']:.1%}")
    if bonus_stats:
        print(f"  [With T9 bonus]: 3+: {bonus_stats['3plus']:.1%}  3+bonus: {bonus_stats['3plus_bonus']:.1%}")

    return {
        "iteration":       name,
        "train_draws":     len(train_draws),
        "test_draws":      len(test_draws),
        "train_dn_range":  [int(train_draws[0]["draw_number"]), int(train_draws[-1]["draw_number"])],
        "test_dn_range":   [int(test_draws[0]["draw_number"]),  int(test_draws[-1]["draw_number"])],
        "tickets":         [[int(n) for n in t] for t in tickets_final],
        "bonus_ticket":    [int(n) for n in bonus_ticket] if bonus_ticket else None,
        "results":         {k: v for k,v in stats.items()},
        "results_with_bonus": bonus_stats if bonus_stats else None,
        "theoretical":     {"3plus": round(theo_3,4), "4plus": round(theo_4,4), "5plus": round(theo_5,4)},
        "lift_3plus":      round(stats["3plus"]/max(theo_3,0.0001), 3),
        "lift_4plus":      round(stats["4plus"]/max(theo_4,0.0001), 3),
        "ensemble_weights": {k: round(v,3) for k,v in ensemble.weights.items()},
        "config":          {"N_CANDIDATES":N_CANDIDATES,"VAL_DRAWS":VAL_DRAWS,"N_TRIALS":N_TRIALS,
                            "POOL_SIZE":POOL_SIZE,"MAX_SHARED":MAX_SHARED,"LAMBDA":LAMBDA,
                            "ADD_BIAS":ADD_BIAS,"N_TICKETS":N_TICKETS,"N_SELECT":N_SELECT},
    }


def main():
    print("=" * 60)
    print("  SYSTEM 8 — FULL WALK-FORWARD BACKTEST (3 ITERATIONS)")
    print("  Separate from 6num and sys7. Full rigor.")
    print("  Triplets excluded | Dual-grid applied | T9 bonus slot")
    print("  diversity_select | SHAP per iter | Full metrics")
    print("=" * 60)

    all_draws = load_draws()
    print(f"Loaded {len(all_draws)} draws | dn {all_draws[0]['draw_number']}-{all_draws[-1]['draw_number']}")

    results = []
    for cfg in ITERATIONS:
        res = run_iteration(cfg, all_draws)
        results.append(res)

    print(f"\n{'='*60}")
    print("  SYS8 BACKTEST SUMMARY")
    print(f"{'='*60}")
    for res in results:
        r = res["results"]
        print(f"  {res['iteration']} ({res['test_draws']} test) | "
              f"3+: {r['3plus']:.1%} (lift {res['lift_3plus']:.2f}x) | "
              f"4+: {r['4plus']:.1%} (lift {res['lift_4plus']:.2f}x) | "
              f"3+bonus: {r['3plus_bonus']:.1%}")

    out = {
        "system": "sys8", "n_select": N_SELECT,
        "run_date": datetime.datetime.now(datetime.UTC).isoformat(),
        "triplet_exclusion": True, "doublet_exclusion": False,
        "dual_grid": True, "diversity_select": True, "bonus_slot": True,
        "notes": "Fully separate from 6num and sys7. Max rigor.",
        "iterations": results,
    }
    out_path = os.path.join(RESULTS_DIR, "backtest_sys8.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
