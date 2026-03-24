"""
Full Backtest Runner — StatLot 649
Runs all 6 models × 3 pool sizes (6, 7, 8) = 18 backtest configurations.
Produces a comprehensive comparison report with win rates, lift vs random.

Usage:
  python3 scripts/run_full_backtest.py
  python3 scripts/run_full_backtest.py --model rf --pool 7
  python3 scripts/run_full_backtest.py --train-rl  # also trains RL agent
"""
import sys
import os
import argparse
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db, init_db
from db.models import DrawRecord, BacktestRun
from engine.backtest import run_backtest
from engine.m6_rl import RLAgent
from engine.m2_ev_kelly import compute_base_rate

MODELS = ["baseline", "m2_ev", "rf", "monte_carlo", "xgb", "rl"]
POOLS = [6, 7, 8]


def load_draws_from_db():
    """Load all draws from local SQLite DB."""
    db = next(get_db())
    rows = db.query(DrawRecord).order_by(DrawRecord.draw_number).all()
    draws = []
    seen = set()
    for r in rows:
        if r.draw_number in seen:
            continue
        seen.add(r.draw_number)
        nums = sorted([r.n1, r.n2, r.n3, r.n4, r.n5, r.n6])
        draws.append({
            "draw_number": r.draw_number,
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "additional": r.additional,
            "nums": nums,
        })
    return draws


def random_baseline(draws, n_test=200, n_preds=10):
    """Pure random baseline — how often does random get 3+?"""
    import random
    random.seed(99)
    match3, match4, match5 = 0, 0, 0
    for test in draws[-n_test:]:
        actual = set(test["nums"])
        best = 0
        for _ in range(n_preds):
            combo = set(random.sample(range(1, 50), 6))
            best = max(best, len(combo & actual))
        if best >= 3: match3 += 1
        if best >= 4: match4 += 1
        if best >= 5: match5 += 1
    return {
        "3plus_rate": match3 / n_test,
        "4plus_rate": match4 / n_test,
        "5plus_rate": match5 / n_test,
    }


def print_results_table(all_results):
    """Pretty print comparison table."""
    header = f"{'Model':<15} {'Pool':<6} {'3+%':<8} {'4+%':<8} {'5+%':<8} {'Avg Match':<10} {'Lift vs Rand':<14}"
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS — ALL MODELS × ALL POOL SIZES")
    print("=" * 70)
    print(header)
    print("-" * 70)

    rand_3 = all_results.get("random_baseline", {}).get("3plus_rate", 0.05)

    for key, r in sorted(all_results.items()):
        if key == "random_baseline":
            continue
        model = r.get("model_type", "?")
        pool = r.get("pool_size", 6)
        r3 = r.get("three_plus_rate", 0)
        r4 = r.get("four_plus_rate", 0)
        r5 = r.get("five_plus_rate", 0)
        avg = r.get("avg_match", 0)
        lift = ((r3 - rand_3) / rand_3 * 100) if rand_3 > 0 else 0
        print(f"{model:<15} {pool:<6} {r3:.1%}   {r4:.1%}   {r5:.1%}   {avg:.3f}     +{lift:.1f}%")

    print("=" * 70)
    print(f"{'RANDOM':<15} {'6':<6} {rand_3:.1%}   baseline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--pool", type=int, choices=[6, 7, 8, 0], default=0,
                        help="0 = all pools")
    parser.add_argument("--train-rl", action="store_true")
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--n-preds", type=int, default=10)
    parser.add_argument("--mc-sims", type=int, default=3000)
    parser.add_argument("--output", default="backtest_results.json")
    args = parser.parse_args()

    print("Initializing DB...")
    init_db()

    print("Loading draws...")
    draws = load_draws_from_db()
    print(f"  {len(draws)} unique draws loaded (#{draws[0]['draw_number']} - #{draws[-1]['draw_number']})")

    # Base rate analysis
    print("\nBase Rate Analysis:")
    base_rate = compute_base_rate(draws, lookback=300)
    print(f"  Historical 3+ match rate: {base_rate['base_rate_3plus']:.4f}")
    print(f"  Sum range (68%): {base_rate['sum_range_68pct']}")
    print(f"  Dominant odd count: {base_rate['dominant_odd_count']}")

    # Train RL if requested
    if args.train_rl:
        print("\nTraining RL Agent (50,000 iterations)...")
        agent = RLAgent()
        if not agent.load():
            agent.train(draws, n_iterations=50000, progress_every=5000)
        else:
            print("Loaded existing RL agent.")

    # Random baseline
    print("\nComputing random baseline...")
    rand = random_baseline(draws, n_test=min(300, len(draws)//3))
    print(f"  Random 3+: {rand['3plus_rate']:.2%}, 4+: {rand['4plus_rate']:.2%}")

    all_results = {"random_baseline": rand}

    models_to_run = MODELS if args.model == "all" else [args.model]
    pools_to_run = POOLS if args.pool == 0 else [args.pool]

    total_runs = len(models_to_run) * len(pools_to_run)
    run_num = 0

    for model in models_to_run:
        for pool in pools_to_run:
            run_num += 1
            key = f"{model}_pool{pool}"
            print(f"\n[{run_num}/{total_runs}] Running: {model.upper()} | Pool={pool}")
            t0 = time.time()

            try:
                result = run_backtest(
                    draws=draws,
                    model_type=model,
                    pool_size=pool,
                    n_preds=args.n_preds,
                    min_history=args.min_history,
                    mc_simulations=args.mc_sims,
                )
                elapsed = time.time() - t0
                result["elapsed_sec"] = round(elapsed, 1)

                r3 = result.get("three_plus_rate", 0)
                r4 = result.get("four_plus_rate", 0)
                r5 = result.get("five_plus_rate", 0)
                avg = result.get("avg_match", 0)
                lift = ((r3 - rand["3plus_rate"]) / rand["3plus_rate"] * 100) if rand["3plus_rate"] > 0 else 0
                print(f"  3+: {r3:.2%} | 4+: {r4:.2%} | 5+: {r5:.2%} | avg: {avg:.3f} | lift: +{lift:.1f}% | {elapsed:.1f}s")

                all_results[key] = result

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    print_results_table(all_results)

    # Save results
    with open(args.output, "w") as f:
        # Make JSON serializable
        clean = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                clean[k] = {kk: (vv if not hasattr(vv, 'item') else vv.item())
                             for kk, vv in v.items() if kk not in ("per_draw_results", "five_plus_events")}
            else:
                clean[k] = v
        json.dump(clean, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Find best model
    best_key = max(
        (k for k in all_results if k != "random_baseline"),
        key=lambda k: all_results[k].get("three_plus_rate", 0)
    )
    best = all_results[best_key]
    print(f"\n🏆 BEST MODEL: {best_key}")
    print(f"   3+ Rate: {best.get('three_plus_rate', 0):.2%}")
    print(f"   4+ Rate: {best.get('four_plus_rate', 0):.2%}")
    print(f"   Lift vs Random: +{((best.get('three_plus_rate',0) - rand['3plus_rate']) / rand['3plus_rate'] * 100):.1f}%")


if __name__ == "__main__":
    main()
