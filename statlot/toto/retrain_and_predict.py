"""
toto/retrain_and_predict.py — Full retrain + generate predictions → save to DuckDB
Generates: 3×Sys6, 1×Sys7 (mandatory) + Sys8-12 optional (supersets of Sys7)
"""
import json, sys, uuid, duckdb
from datetime import datetime, date
from itertools import combinations
import numpy as np
sys.path.insert(0, "/home/ubuntu/statlot-649/statlot")

from engine.candidate_gen_v2 import generate_candidates
from engine.diversity_select import select_diverse_tickets
from engine.models.m1_bayes import BayesianFreqScorer
from engine.models.m2_poisson_gap import train_m2_poisson, score_m2_poisson
from engine.models.m6_dqn import DQNAgent
from engine.models.m7_markov import MarkovScorer
from engine.models.additional import AdditionalPredictor

DB_PATH     = "/home/ubuntu/statlot-649/statlot_toto.duckdb"
HISTORY_FILE= "/home/ubuntu/statlot-649/statlot/sp_historical_draws.json"
DRAW_FILE   = "/home/ubuntu/statlot-649/statlot/toto/latest_draw.json"
STATUS_FILE = "/home/ubuntu/statlot-649/logs/toto_pipeline_status.json"

def write_status(step, status):
    with open(STATUS_FILE, "w") as f:
        json.dump({"step": step, "status": status, "ts": datetime.utcnow().isoformat()}, f)

def norm(arr):
    mn,mx = arr.min(), arr.max()
    return (arr-mn)/(mx-mn+1e-9)

def expand_to_sys(base_numbers: list, target_size: int, all_candidates: list,
                  final_scores: np.ndarray) -> list:
    """Expand a 6-num ticket to sys-N by adding highest-scoring candidates."""
    base_set = set(base_numbers)
    pool_size = target_size - 6
    # Score all numbers not in base ticket by how often they appear in top candidates
    num_freq = {}
    top_idx = np.argsort(final_scores)[::-1][:500]
    for i in top_idx:
        for n in all_candidates[i]:
            if n not in base_set:
                num_freq[n] = num_freq.get(n, 0) + final_scores[i]
    extras = sorted(num_freq, key=lambda x: -num_freq[x])[:pool_size]
    return sorted(base_numbers + extras)

def run():
    write_status("retrain", "running")

    # Load history + append latest draw if available
    with open(HISTORY_FILE) as f:
        raw = json.load(f)

    try:
        with open(DRAW_FILE) as f:
            latest = json.load(f)
        # Check if already in history
        existing = {d["draw_number"] for d in raw}
        if latest["draw_number"] not in existing:
            nums = latest["numbers"]
            raw.append({
                "draw_number": latest["draw_number"],
                "draw_date": latest["draw_date"],
                "n1":nums[0],"n2":nums[1],"n3":nums[2],
                "n4":nums[3],"n5":nums[4],"n6":nums[5],
                "additional": latest["additional"]
            })
            # Save updated history
            with open(HISTORY_FILE, "w") as f:
                json.dump(raw, f, indent=2)
            print(f"Appended draw #{latest['draw_number']} to history")
    except FileNotFoundError:
        print("No latest draw file — predicting for next unknown draw")
        latest = None

    clean = [d for d in raw if all(d.get(f'n{i}') is not None for i in range(1,7))]
    draws = [{"draw_number": d["draw_number"],
              "nums": sorted([d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]]),
              "additional": d.get("additional", 0)} for d in clean]
    print(f"Training on {len(draws)} draws...")

    # === TRAIN ===
    W = {'m1':0.265,'m2':0.176,'m6':0.118,'m7':0.294,'add_bias':0.147}
    m1 = BayesianFreqScorer(); m1.fit(draws)
    m2_model = train_m2_poisson([{"nums": d["nums"]} for d in draws])
    m7 = MarkovScorer(); m7.fit(draws)
    add_pred = AdditionalPredictor(); add_pred.fit(draws)
    m6 = DQNAgent()
    try: m6.load("_iter1")
    except: m6.fit(draws[-500:])  # fallback: train on recent 500

    # === SCORE ===
    write_status("retrain", "scoring")
    print("Generating 8k candidates...")
    cands_raw = generate_candidates(draws, n_candidates=8000)
    candidates = [list(int(x) for x in c) for c in cands_raw]

    add_bias = add_pred.get_bias_scores(draws, strength=0.15)
    add_top = [int(n) for n,_ in sorted(add_bias.items(), key=lambda x:-x[1])[:5]]

    S = {}
    S['m1'] = norm(np.array(m1.score_batch(candidates, draws), dtype=float))
    S['m2'] = norm(np.array([score_m2_poisson(m2_model, c) for c in candidates], dtype=float))
    S['m7'] = norm(np.array(m7.score_batch(candidates, draws), dtype=float))
    try: S['m6'] = norm(np.array(m6.score_batch(candidates, draws), dtype=float))
    except: S['m6'] = np.zeros(len(candidates))
    S['add_bias'] = norm(np.array([np.mean([add_bias.get(n,0.0) for n in c]) for c in candidates], dtype=float))

    final_scores = sum(W[k]*S[k] for k in W)
    top_idx = np.argsort(final_scores)[::-1][:500]
    scored = [(tuple(candidates[i]), float(final_scores[i])) for i in top_idx]

    # === PICK TICKETS ===
    write_status("retrain", "selecting")
    # 3x Sys6
    tickets_raw = select_diverse_tickets(scored, n_tickets=3, lambda_=0.3,
                                          max_shared=2, bonus_candidates=add_top[:3])
    sys6_tickets = [sorted(list(t)) for t in tickets_raw]

    # 1x Sys7 — expand best Sys6 to 7 numbers
    sys7 = expand_to_sys(sys6_tickets[0], 7, candidates, final_scores)

    # Sys8-12 — expand sys7 progressively
    sys8  = expand_to_sys(sys7,  8,  candidates, final_scores)
    sys9  = expand_to_sys(sys8,  9,  candidates, final_scores)
    sys10 = expand_to_sys(sys9,  10, candidates, final_scores)
    sys11 = expand_to_sys(sys10, 11, candidates, final_scores)
    sys12 = expand_to_sys(sys11, 12, candidates, final_scores)

    # Bonus ticket (additional-biased)
    bonus_sc = 0.5*S['m1'] + 0.5*S['add_bias']
    b_idx = np.argsort(bonus_sc)[::-1][:200]
    bonus_scored = [(tuple(candidates[i]), float(bonus_sc[i])) for i in b_idx]
    bonus_raw = select_diverse_tickets(bonus_scored, n_tickets=1, lambda_=0.2, max_shared=3)
    bonus_t6 = sorted(list(bonus_raw[0]))

    # === SAVE TO DUCKDB ===
    write_status("retrain", "saving")
    from toto.toto_db import init_schema
    init_schema()

    next_draw_num = (latest["draw_number"] + 1) if latest else None
    pred_id = f"pred_{next_draw_num or 'next'}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    gen_at = datetime.utcnow().isoformat()

    con = duckdb.connect(DB_PATH)
    con.execute("""
        INSERT OR REPLACE INTO toto_predictions
            (id, draw_number, draw_date, generated_at,
             sys6_t1, sys6_t2, sys6_t3, sys7_t1,
             sys8_t1, sys9_t1, sys10_t1, sys11_t1, sys12_t1,
             bonus_t6, additional_picks,
             cost_mandatory, cost_with_sys8, cost_with_sys12,
             engine_weights, draws_trained,
             backtest_3plus, backtest_lift, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        pred_id, next_draw_num, None, gen_at,
        json.dumps(sys6_tickets[0]), json.dumps(sys6_tickets[1]), json.dumps(sys6_tickets[2]),
        json.dumps(sys7), json.dumps(sys8), json.dumps(sys9), json.dumps(sys10),
        json.dumps(sys11), json.dumps(sys12),
        json.dumps(bonus_t6), json.dumps(add_top),
        10.00,   # 3*$1 + $7
        38.00,   # + $28 for sys8
        934.00,  # + $924 for sys12
        json.dumps(W), len(draws),
        "10.4%", "1.16x",
        f"Auto-generated after draw #{latest['draw_number'] if latest else 'N/A'}"
    ])
    con.close()

    write_status("retrain", "done")

    print("\n" + "="*58)
    print(f"  PREDICTION SAVED — pred_id: {pred_id}")
    print(f"  For draw: #{next_draw_num} | Trained on {len(draws)} draws")
    print("="*58)
    print(f"  Sys6 T1: {sys6_tickets[0]}")
    print(f"  Sys6 T2: {sys6_tickets[1]}")
    print(f"  Sys6 T3: {sys6_tickets[2]}")
    print(f"  Sys7 T1: {sys7}")
    print(f"  Sys8 T1: {sys8}")
    print(f"  Sys9 T1: {sys9}")
    print(f"  Sys10 T1:{sys10}")
    print(f"  Sys11 T1:{sys11}")
    print(f"  Sys12 T1:{sys12}")
    print(f"  Bonus T6:{bonus_t6}")
    print(f"  Additional picks: {add_top}")
    print(f"  Cost mandatory (3×Sys6 + 1×Sys7): $10")
    print(f"  Cost with Sys8: $38  |  Cost with Sys12: $934")

    return {
        "pred_id": pred_id,
        "sys6": sys6_tickets,
        "sys7": sys7,
        "sys8": sys8,
        "sys9": sys9,
        "sys10": sys10,
        "sys11": sys11,
        "sys12": sys12,
        "bonus_t6": bonus_t6,
        "additional_picks": add_top
    }

if __name__ == "__main__":
    run()
