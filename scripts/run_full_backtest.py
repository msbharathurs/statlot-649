"""
Run all backtest variants and print comparison table.
Usage: python scripts/run_full_backtest.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from db.database import SessionLocal
from db.models import DrawRecord
from engine.backtest import run_backtest


def main():
    db = SessionLocal()
    draws_raw = db.query(DrawRecord).order_by(DrawRecord.draw_number).all()
    draws = [r.to_dict() for r in draws_raw]
    db.close()
    print(f"Loaded {len(draws)} draws from DB\n")

    configs = [
        {"model_type": "baseline",    "pool_size": 6, "label": "Baseline     6-num"},
        {"model_type": "baseline",    "pool_size": 7, "label": "Baseline     7-num"},
        {"model_type": "baseline",    "pool_size": 8, "label": "Baseline     8-num"},
        {"model_type": "rf",          "pool_size": 6, "label": "RF(M3)       6-num"},
        {"model_type": "rf",          "pool_size": 7, "label": "RF(M3)       7-num"},
        {"model_type": "xgb",         "pool_size": 6, "label": "XGBoost(M5)  6-num"},
        {"model_type": "monte_carlo", "pool_size": 6, "label": "MonteCarlo(M4) 6-num"},
    ]

    results = []
    for cfg in configs:
        print(f"Running: {cfg['label']}...")
        t0 = time.time()
        res = run_backtest(
            draws=draws,
            model_type=cfg["model_type"],
            pool_size=cfg["pool_size"],
            n_preds=10,
            n_candidates=150000,
            min_history=100,
            mc_simulations=3000,
        )
        elapsed = time.time() - t0
        res["label"] = cfg["label"]
        res["elapsed_s"] = round(elapsed, 1)
        results.append(res)
        print(f"  Done {elapsed:.1f}s  avg={res['avg_match']:.3f}  "
              f"3+={res['three_plus_rate']}%  4+={res['four_plus_rate']}%  "
              f"5+={res['five_plus_count']}  lift={res['lift_pct']:+.1f}%\n")

    print("\n" + "="*95)
    print(f"{'Model':<26} {'Tested':>6} {'AvgMatch':>9} {'RandAvg':>8} {'Lift':>7} "
          f"{'3+%':>6} {'4+%':>6} {'5+cnt':>6} {'Time':>6}")
    print("="*95)
    for r in results:
        print(f"  {r['label']:<24} {r['total_tested']:>6} {r['avg_match']:>9.4f} "
              f"{r['rand_avg']:>8.4f} {r['lift_pct']:>+6.1f}% "
              f"{r['three_plus_rate']:>5.1f}% {r['four_plus_rate']:>5.1f}% "
              f"{r['five_plus_count']:>6} {r['elapsed_s']:>5.1f}s")

    print("\n\n=== 5+ MATCH EVENTS ===")
    for r in results:
        if r["five_plus_events"]:
            print(f"\n[{r['label']}]")
            for ev in r["five_plus_events"]:
                print(f"  Draw #{ev['draw']}: actual={ev['actual']}  "
                      f"predicted={ev['predicted']}  matched={ev['matched_nums']}")

    with open("backtest_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → backtest_summary.json")


if __name__ == "__main__":
    main()
