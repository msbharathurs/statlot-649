"""
toto/retrain_and_predict.py — Full retrain + generate predictions → save to DuckDB

Training window: draws >= 2341 only (post-2341 is the clean, verified window).
Pre-2341 data exists in toto_draws but is excluded from all training and backtesting
due to unreliable data quality. This is a permanent rule — do not change without
explicit instruction from Bharath.

On each run this writes to TWO tables:
  - toto_predictions   : one row per draw prediction (the "latest" record, may be replaced)
  - toto_predictions_log: append-only audit trail — one row per ticket per run, NEVER deleted

Generates: 3×Sys6, 1×Sys7 (mandatory) + Sys8-12 optional (supersets of Sys7)
"""
import json
import os
import subprocess
import sys
import uuid
import duckdb
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

DB_PATH      = "/home/ubuntu/statlot-649/statlot_toto.duckdb"
DRAW_FILE    = "/home/ubuntu/statlot-649/statlot/toto/latest_draw.json"
STATUS_FILE  = "/home/ubuntu/statlot-649/logs/toto_pipeline_status.json"
LOG_PATH     = "/home/ubuntu/statlot-649/logs/toto_retrain.log"

# ── Training window — PERMANENT RULE ─────────────────────────────────────────
MIN_TRAIN_DRAW = 2341  # Do NOT lower this without Bharath's explicit approval


def write_status(step, status):
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump({"step": step, "status": status, "ts": datetime.utcnow().isoformat()}, f)


def norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def get_git_commit() -> str:
    """Return short git commit hash for model_version tracking."""
    try:
        result = subprocess.run(
            ["git", "-C", "/home/ubuntu/statlot-649", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def expand_to_sys(base_numbers: list, target_size: int, all_candidates: list,
                  final_scores: np.ndarray) -> list:
    """Expand a 6-num ticket to sys-N by adding highest-scoring extra numbers."""
    base_set = set(base_numbers)
    pool_size = target_size - 6
    num_freq = {}
    top_idx = np.argsort(final_scores)[::-1][:500]
    for i in top_idx:
        for n in all_candidates[i]:
            if n not in base_set:
                num_freq[n] = num_freq.get(n, 0) + final_scores[i]
    extras = sorted(num_freq, key=lambda x: -num_freq[x])[:pool_size]
    return sorted(base_numbers + extras)


def load_training_draws() -> tuple:
    """
    Load clean draws from toto_draws WHERE draw_no >= MIN_TRAIN_DRAW.
    Returns (draws_list, latest_draw_no_in_db).
    Never touches pre-2341 data.
    """
    con = duckdb.connect(DB_PATH)
    rows = con.execute(f"""
        SELECT draw_no, draw_date, n1, n2, n3, n4, n5, n6, additional
        FROM toto_draws
        WHERE draw_no >= {MIN_TRAIN_DRAW}
          AND n1 IS NOT NULL AND n2 IS NOT NULL AND n3 IS NOT NULL
          AND n4 IS NOT NULL AND n5 IS NOT NULL AND n6 IS NOT NULL
          AND draw_date IS NOT NULL
        ORDER BY draw_no ASC
    """).fetchall()
    max_draw_no = con.execute(f"SELECT MAX(draw_no) FROM toto_draws WHERE draw_no >= {MIN_TRAIN_DRAW}").fetchone()[0]
    con.close()

    draws = []
    for row in rows:
        draw_no, draw_date, n1, n2, n3, n4, n5, n6, additional = row
        draws.append({
            "draw_number": draw_no,
            "nums": sorted([n1, n2, n3, n4, n5, n6]),
            "additional": additional or 0
        })

    return draws, max_draw_no


def write_predictions_log(
    con,
    draw_no: int,
    retrain_draw_no: int,
    model_version: str,
    n_training_draws: int,
    tickets: list,      # list of (system_type: str, numbers: list, scores: list | None)
    dry_run: bool = False
):
    """
    Append one row per ticket to toto_predictions_log.
    This is ADDITIVE — never replaced, never deleted.
    Each row captures: which draw it's for, which system, what numbers,
    confidence scores if available, how many draws the model trained on.
    """
    ts = datetime.utcnow().isoformat()
    rows_written = []

    for system_type, numbers, confidence_scores in tickets:
        note_parts = [f"trained on {n_training_draws} draws (>= draw #{MIN_TRAIN_DRAW})"]
        if dry_run:
            note_parts.append("DRY RUN")

        con.execute("""
            INSERT INTO toto_predictions_log
                (draw_no, predicted_at, model_version, predicted_numbers,
                 confidence_scores, retrain_draw_no, system_type, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            draw_no,
            ts,
            model_version,
            numbers,
            confidence_scores if confidence_scores is not None else [],
            retrain_draw_no,
            system_type,
            "; ".join(note_parts)
        ])
        rows_written.append({
            "draw_no": draw_no,
            "system_type": system_type,
            "numbers": numbers,
            "retrain_draw_no": retrain_draw_no,
            "model_version": model_version
        })

    return rows_written


def run(dry_run: bool = False):
    write_status("retrain", "running")
    git_hash = get_git_commit()
    print(f"Git commit: {git_hash}")
    print(f"Training window: draws >= {MIN_TRAIN_DRAW} only (pre-2341 excluded by rule)")

    # === LOAD TRAINING DATA FROM DB (post-2341 only) ===
    draws, max_draw_in_db = load_training_draws()
    n_training = len(draws)
    print(f"Training draws loaded from toto_draws: {n_training} draws (#{MIN_TRAIN_DRAW}–#{max_draw_in_db})")

    if n_training < 100:
        raise RuntimeError(f"Insufficient training data: only {n_training} draws >= {MIN_TRAIN_DRAW}. Aborting.")

    # === OPTIONALLY APPEND LATEST DRAW FROM FILE ===
    latest = None
    try:
        with open(DRAW_FILE) as f:
            latest = json.load(f)
        existing_draw_nos = {d["draw_number"] for d in draws}
        if latest["draw_number"] not in existing_draw_nos and latest["draw_number"] >= MIN_TRAIN_DRAW:
            nums = latest["numbers"]
            draws.append({
                "draw_number": latest["draw_number"],
                "nums": sorted([nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]]),
                "additional": latest.get("additional", 0)
            })
            print(f"Appended draw #{latest['draw_number']} from latest_draw.json to training set")
            n_training = len(draws)
            max_draw_in_db = latest["draw_number"]
    except FileNotFoundError:
        print("No latest_draw.json — predicting for next unknown draw")
        latest = None

    # === TRAIN ===
    W = {'m1': 0.265, 'm2': 0.176, 'm6': 0.118, 'm7': 0.294, 'add_bias': 0.147}
    m1 = BayesianFreqScorer()
    m1.fit(draws)
    m2_model = train_m2_poisson([{"nums": d["nums"]} for d in draws])
    m7 = MarkovScorer()
    m7.fit(draws)
    add_pred = AdditionalPredictor()
    add_pred.fit(draws)
    m6 = DQNAgent()
    try:
        m6.load("_iter1")
    except Exception:
        m6.fit(draws[-500:])

    # === SCORE ===
    write_status("retrain", "scoring")
    print(f"Generating candidates (trained on {n_training} draws)...")
    cands_raw  = generate_candidates(draws, n_candidates=8000)
    candidates = [list(int(x) for x in c) for c in cands_raw]

    add_bias = add_pred.get_bias_scores(draws, strength=0.15)
    add_top  = [int(n) for n, _ in sorted(add_bias.items(), key=lambda x: -x[1])[:5]]

    S = {}
    S['m1']       = norm(np.array(m1.score_batch(candidates, draws), dtype=float))
    S['m2']       = norm(np.array([score_m2_poisson(m2_model, c) for c in candidates], dtype=float))
    S['m7']       = norm(np.array(m7.score_batch(candidates, draws), dtype=float))
    try:
        S['m6']   = norm(np.array(m6.score_batch(candidates, draws), dtype=float))
    except Exception:
        S['m6']   = np.zeros(len(candidates))
    S['add_bias'] = norm(np.array([np.mean([add_bias.get(n, 0.0) for n in c]) for c in candidates], dtype=float))

    final_scores = sum(W[k] * S[k] for k in W)
    top_idx  = np.argsort(final_scores)[::-1][:500]
    scored   = [(tuple(candidates[i]), float(final_scores[i])) for i in top_idx]

    # Build per-number confidence from top-500 ensemble scores
    num_score_acc = {}
    num_count     = {}
    for i in top_idx:
        sc = float(final_scores[i])
        for n in candidates[i]:
            num_score_acc[n] = num_score_acc.get(n, 0.0) + sc
            num_count[n]     = num_count.get(n, 0) + 1
    num_confidence = {n: num_score_acc[n] / num_count[n] for n in num_score_acc}
    max_conf = max(num_confidence.values()) if num_confidence else 1.0

    def ticket_confidence(numbers: list) -> list:
        """Return per-number confidence scores normalised to [0, 1]."""
        return [round(num_confidence.get(n, 0.0) / max_conf, 4) for n in numbers]

    # === PICK TICKETS ===
    write_status("retrain", "selecting")
    tickets_raw  = select_diverse_tickets(scored, n_tickets=3, lambda_=0.3,
                                          max_shared=2, bonus_candidates=add_top[:3])
    sys6_tickets = [sorted(list(t)) for t in tickets_raw]

    sys7  = expand_to_sys(sys6_tickets[0], 7,  candidates, final_scores)
    sys8  = expand_to_sys(sys7,            8,  candidates, final_scores)
    sys9  = expand_to_sys(sys8,            9,  candidates, final_scores)
    sys10 = expand_to_sys(sys9,           10,  candidates, final_scores)
    sys11 = expand_to_sys(sys10,          11,  candidates, final_scores)
    sys12 = expand_to_sys(sys11,          12,  candidates, final_scores)

    bonus_sc     = 0.5 * S['m1'] + 0.5 * S['add_bias']
    b_idx        = np.argsort(bonus_sc)[::-1][:200]
    bonus_scored = [(tuple(candidates[i]), float(bonus_sc[i])) for i in b_idx]
    bonus_raw    = select_diverse_tickets(bonus_scored, n_tickets=1, lambda_=0.2, max_shared=3)
    bonus_t6     = sorted(list(bonus_raw[0]))

    # Determine draw number this prediction is FOR
    next_draw_num = (max_draw_in_db + 1) if max_draw_in_db else None

    # === SAVE TO DB ===
    write_status("retrain", "saving")
    from toto.toto_db import init_schema
    init_schema()

    pred_id = f"pred_{next_draw_num or 'next'}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    gen_at  = datetime.utcnow().isoformat()

    con = duckdb.connect(DB_PATH)

    # ── toto_predictions (one row per draw, REPLACE on re-run) ────────────────
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
        10.00, 38.00, 934.00,
        json.dumps(W), n_training,
        "pending", "pending",
        f"Trained on draws >={MIN_TRAIN_DRAW}; last draw in training: #{max_draw_in_db}"
    ])

    # ── toto_predictions_log (append-only — one row per ticket) ───────────────
    log_tickets = [
        ("sys6_t1", sys6_tickets[0], ticket_confidence(sys6_tickets[0])),
        ("sys6_t2", sys6_tickets[1], ticket_confidence(sys6_tickets[1])),
        ("sys6_t3", sys6_tickets[2], ticket_confidence(sys6_tickets[2])),
        ("sys7_t1", sys7,            ticket_confidence(sys7)),
        ("sys8_t1", sys8,            ticket_confidence(sys8)),
        ("sys9_t1", sys9,            ticket_confidence(sys9)),
        ("sys10_t1", sys10,          ticket_confidence(sys10)),
        ("sys11_t1", sys11,          ticket_confidence(sys11)),
        ("sys12_t1", sys12,          ticket_confidence(sys12)),
        ("bonus_t6", bonus_t6,       ticket_confidence(bonus_t6)),
    ]

    log_rows = write_predictions_log(
        con=con,
        draw_no=next_draw_num,
        retrain_draw_no=max_draw_in_db,
        model_version=git_hash,
        n_training_draws=n_training,
        tickets=log_tickets,
        dry_run=dry_run
    )

    con.close()

    write_status("retrain", "done")

    # === PRINT RESULTS ===
    print("\n" + "=" * 60)
    print(f"  PREDICTION SAVED — pred_id: {pred_id}")
    print(f"  For draw: #{next_draw_num} | Trained on {n_training} draws (>= #{MIN_TRAIN_DRAW})")
    print(f"  Git version: {git_hash} | Dry run: {dry_run}")
    print("=" * 60)
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
    print(f"\n  toto_predictions_log rows written: {len(log_rows)}")
    for row in log_rows:
        print(f"    → {row['system_type']:10s}  draw#{row['draw_no']}  "
              f"numbers={row['numbers']}  model={row['model_version']}")

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
        "additional_picks": add_top,
        "log_rows": log_rows
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but mark predictions as dry-run in log")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
