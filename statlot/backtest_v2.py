"""
backtest_v2.py — Clean Walk-Forward Backtest, Zero Leakage
3 iterations: Train 1-1650/Test 1651-1823, Train 1-1750/Test 1751-1823, Train 1-1810/Test 1811-1823
Metric: P(3+ match OR 3+bonus in best of 5 tickets). Nothing below 3-match reported.
Auto-saves models + results to S3 after each iteration (if S3_BUCKET env var is set).
"""
import json, os, sys, time, numpy as np

from engine.features_v2 import build_features
from engine.candidate_gen_v2 import generate_candidates
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
from engine.ensemble import EnsembleScorer
from engine.diversity_select import select_diverse_tickets, coverage_report

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
S3_BUCKET = os.environ.get("S3_BUCKET", "")          # e.g. statlot-649
S3_PREFIX = os.environ.get("S3_PREFIX", "statlot-649") # folder prefix inside bucket

ITERATIONS = [
    {"name": "Iter1", "train_end": 1650, "test_start": 1651, "test_end": 1823},
    {"name": "Iter2", "train_end": 1750, "test_start": 1751, "test_end": 1823},
    {"name": "Iter3", "train_end": 1810, "test_start": 1811, "test_end": 1823},
]
RANDOM_BASELINE = 0.0898


# ─── S3 HELPERS ──────────────────────────────────────────────────────────────

def _s3_upload_file(local_path, s3_key):
    """Upload a single file to S3. Silently skips if boto3/bucket not configured."""
    if not S3_BUCKET:
        return
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"  [S3] ✅ uploaded → s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"  [S3] ⚠️  upload failed ({s3_key}): {e}")


def _s3_upload_dir(local_dir, s3_prefix):
    """Recursively upload all files in a directory to S3."""
    if not S3_BUCKET:
        return
    for root, _, files in os.walk(local_dir):
        for fname in files:
            if fname == ".gitkeep":
                continue
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{rel}".replace("\\", "/")
            _s3_upload_file(local_path, s3_key)


def save_iter_to_s3(iter_name, results_path):
    """After each iteration: upload models + results for that iter."""
    if not S3_BUCKET:
        print("  [S3] S3_BUCKET not set — skipping upload (set env var to enable)")
        return
    suffix = f"_{iter_name.lower()}"
    prefix = f"{S3_PREFIX}/models/{iter_name.lower()}"
    print(f"\n  [S3] Uploading {iter_name} artifacts → s3://{S3_BUCKET}/{prefix}/")

    # Upload saved model files matching this iteration's suffix
    if os.path.isdir(SAVED_MODELS_DIR):
        for fname in os.listdir(SAVED_MODELS_DIR):
            if suffix in fname or fname.startswith("ensemble"):
                local = os.path.join(SAVED_MODELS_DIR, fname)
                _s3_upload_file(local, f"{prefix}/{fname}")

    # Upload results JSON
    if os.path.exists(results_path):
        fname = os.path.basename(results_path)
        _s3_upload_file(results_path, f"{S3_PREFIX}/results/{fname}")


def save_final_to_s3(all_results_path):
    if not S3_BUCKET:
        return
    _s3_upload_file(all_results_path, f"{S3_PREFIX}/results/all_results.json")
    print(f"  [S3] Final summary uploaded.")


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_draws(csv_path):
    import csv
    draws = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nums = sorted([
                int(row["1st Number"].strip()), int(row["2nd Number"].strip()),
                int(row["3"].strip()),          int(row["4"].strip()),
                int(row["5"].strip()),           int(row["6th Number"].strip()),
            ])
            add_raw = row.get("Additional Number", "").strip()
            draws.append({
                "draw_number": int(row["Draw"].strip()),
                "nums": nums,
                "additional": int(add_raw) if add_raw else None,
            })
    draws.sort(key=lambda x: x["draw_number"])
    return draws


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_all_models(draws, train_end_idx, iter_name):
    train = draws[:train_end_idx]
    suffix = f"_{iter_name.lower()}"
    print(f"\n{'='*60}\n  TRAINING {iter_name} — {train_end_idx} draws\n{'='*60}")

    print("\n[M1] Bayesian Freq+Decay")
    m1 = BayesianFreqScorer(); m1.fit(train)

    print("[M2] EV/Kelly")
    m2 = EVKellyScorer(); m2.fit(train)

    print("[M3] Random Forest + SMOTE + Platt")
    m3 = RFScorer(); m3.fit(draws, train_end_idx); m3.save(suffix)

    print("[M4] Monte Carlo importance sampling")
    m4 = MonteCarloScorer(); m4.fit(train)

    print("[M5] XGBoost + Optuna")
    m5 = XGBScorer(); m5.fit(draws, train_end_idx); m5.save(suffix)

    print("[M6] DQN RL")
    m6 = DQNAgent(); m6.fit(draws, train_end_idx, n_episodes=2); m6.save(suffix)

    print("[M7] 2nd-order Markov")
    m7 = MarkovScorer(); m7.fit(train)

    print("[M8] GMM density")
    m8 = GMMScorer(); m8.fit(draws, train_end_idx); m8.save(suffix)

    print("[M9] LSTM sequence model")
    m9 = LSTMScorer(epochs=15); m9.fit(draws, train_end_idx); m9.save(suffix)

    print("[ADD] Additional number predictor")
    add_pred = AdditionalPredictor(); add_pred.fit(train)

    # Build ensemble + tune weights on last 50 train draws
    ensemble = EnsembleScorer()
    for name, scorer in [("m1",m1),("m2",m2),("m3",m3),("m4",m4),
                          ("m5",m5),("m6",m6),("m7",m7),("m8",m8),("m9",m9)]:
        ensemble.register(name, scorer)

    print("\n[ENSEMBLE] Tuning weights on last 50 train draws...")
    val_draws   = train[-50:]
    val_history = train[:-50]
    val_candidates = generate_candidates(val_history, n_candidates=5000)
    if val_candidates:
        ensemble.tune_weights(val_candidates, val_history, val_draws, n_trials=50)
    ensemble.save(suffix)

    return ensemble, add_pred


# ─── TESTING ─────────────────────────────────────────────────────────────────

def run_test(draws, ensemble, add_pred, test_start_idx, test_end_idx, iter_name):
    test_draws = draws[test_start_idx - 1 : test_end_idx]
    n_test = len(test_draws)
    print(f"\n{'='*60}\n  TESTING {iter_name} — {n_test} draws\n{'='*60}\n")

    results = []
    hit_3plus = hit_3bonus = hit_4plus = hit_4bonus = hit_5plus = 0

    for i, test_draw in enumerate(test_draws):
        draw_idx  = test_start_idx - 1 + i
        history   = draws[:draw_idx]
        actual    = set(test_draw["nums"])
        actual_add = test_draw.get("additional")

        candidates = generate_candidates(history, n_candidates=20000)
        scored     = ensemble.score_batch(candidates, history)

        predicted_add = add_pred.predict(history, top_n=3)
        tickets = select_diverse_tickets(scored, n_tickets=5)

        best_match = 0; best_bonus = False; ticket_details = []
        for ticket in tickets:
            m     = len(set(ticket) & actual)
            bonus = bool(actual_add and actual_add in ticket)
            if m > best_match or (m == best_match and bonus and not best_bonus):
                best_match = m; best_bonus = bonus
            ticket_details.append({"combo": sorted(ticket), "match": m, "has_bonus": bonus})

        if best_match >= 3:
            hit_3plus += 1
            if best_bonus: hit_3bonus += 1
        if best_match >= 4:
            hit_4plus += 1
            if best_bonus: hit_4bonus += 1
        if best_match >= 5:
            hit_5plus += 1

        results.append({
            "draw_number":       test_draw["draw_number"],
            "actual":            sorted(actual),
            "actual_additional": actual_add,
            "tickets":           ticket_details,
            "best_match":        best_match,
            "best_has_bonus":    best_bonus,
            "add_predictor_hit": add_pred.score_hit(predicted_add, actual_add),
            "predicted_additional": predicted_add,
        })

        if (i + 1) % 10 == 0 or (i + 1) == n_test:
            print(
                f"  draw {i+1:3d}/{n_test} | "
                f"3+: {hit_3plus/(i+1):.1%} | "
                f"3+bonus: {hit_3bonus/(i+1):.1%} | "
                f"4+: {hit_4plus/(i+1):.1%}",
                flush=True,
            )

    return {
        "iter":           iter_name,
        "n_test":         n_test,
        "n_tickets":      5,
        "hit_3plus":      hit_3plus,
        "hit_3bonus":     hit_3bonus,
        "hit_4plus":      hit_4plus,
        "hit_4bonus":     hit_4bonus,
        "hit_5plus":      hit_5plus,
        "rate_3plus":     round(hit_3plus / n_test, 4),
        "rate_3bonus":    round(hit_3bonus / n_test, 4),
        "rate_4plus":     round(hit_4plus / n_test, 4),
        "rate_4bonus":    round(hit_4bonus / n_test, 4),
        "rate_5plus":     round(hit_5plus / n_test, 4),
        "random_baseline": RANDOM_BASELINE,
        "lift_vs_random": round(hit_3plus / n_test / RANDOM_BASELINE, 3),
        "per_draw":       results,
    }


# ─── SUMMARY PRINT ───────────────────────────────────────────────────────────

def print_summary(s):
    n  = s["n_test"]
    bl = s["random_baseline"]
    print(f"\n{'='*60}\n  RESULTS — {s['iter']} ({n} test draws)\n{'='*60}")
    print(f"  Random baseline (5 tickets): {bl:.1%}\n")
    print(f"  3+ match:       {s['hit_3plus']:3d}/{n}  ({s['rate_3plus']:.1%})")
    print(f"  3+ with bonus:  {s['hit_3bonus']:3d}/{n}  ({s['rate_3bonus']:.1%})")
    print(f"  4+ match:       {s['hit_4plus']:3d}/{n}  ({s['rate_4plus']:.1%})")
    print(f"  4+ with bonus:  {s['hit_4bonus']:3d}/{n}  ({s['rate_4bonus']:.1%})")
    print(f"  5+ match:       {s['hit_5plus']:3d}/{n}  ({s['rate_5plus']:.1%})")
    print(f"\n  Lift vs random: {s['lift_vs_random']:.2f}x")
    print(f"  Improvement:    {(s['rate_3plus'] - bl) / bl * 100:+.1f}% over random")
    print(f"{'='*60}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def run_all(csv_path):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    draws = load_draws(csv_path)
    print(f"\nLoaded {len(draws)} draws ({draws[0]['draw_number']} → {draws[-1]['draw_number']})")
    print(f"S3 bucket: {S3_BUCKET or '(not set — local only)'}\n")

    all_summaries = []

    for iteration in ITERATIONS:
        t0 = time.time()

        ensemble, add_pred = train_all_models(draws, iteration["train_end"], iteration["name"])
        summary = run_test(
            draws, ensemble, add_pred,
            iteration["test_start"], iteration["test_end"], iteration["name"],
        )
        summary["train_time_sec"] = round(time.time() - t0, 1)

        print_summary(summary)
        all_summaries.append(summary)

        # Save results JSON locally
        out_path = os.path.join(RESULTS_DIR, f"{iteration['name'].lower()}_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved locally: {out_path}")

        # Push to S3
        save_iter_to_s3(iteration["name"], out_path)

    # Final comparison table
    print(f"\n{'='*60}\n  WALK-FORWARD COMPARISON\n{'='*60}")
    print(f"  {'Iter':<8} {'Train':>7} {'Test':>5} {'3+':>7} {'3+bonus':>9} {'4+':>7} {'Lift':>7}")
    print(f"  {'-'*55}")
    for s in all_summaries:
        tr = next(x["train_end"] for x in ITERATIONS if x["name"] == s["iter"])
        print(
            f"  {s['iter']:<8} {tr:>7} {s['n_test']:>5} "
            f"{s['rate_3plus']:>7.1%} {s['rate_3bonus']:>9.1%} "
            f"{s['rate_4plus']:>7.1%} {s['lift_vs_random']:>6.2f}x"
        )
    print(f"\n  Random baseline: {RANDOM_BASELINE:.1%}\n{'='*60}")

    all_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(all_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    save_final_to_s3(all_path)

    return all_summaries


if __name__ == "__main__":
    run_all(sys.argv[1] if len(sys.argv) > 1 else "data/draws.csv")
