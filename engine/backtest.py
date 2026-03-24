"""
Rolling-window backtest engine.
Supports per-model backtesting: baseline, RF, XGBoost, Monte Carlo.
Tracks 3+, 4+, 5+ match rates independently.
"""
import random
import numpy as np
from collections import Counter
from typing import Optional

from engine.features import build_feature_row, FEATURE_COLS
from engine.candidate_gen import generate_candidates, expand_pool_to_combos
from engine.models import DrawPropertyPredictor, MonteCarloCandidateScorer


def run_backtest(
    draws: list,
    model_type: str = "baseline",   # baseline | rf | xgb | monte_carlo
    pool_size: int = 6,              # 6, 7, 8
    n_preds: int = 10,
    n_candidates: int = 150000,
    min_history: int = 100,
    mc_simulations: int = 5000,
    progress_callback=None,
) -> dict:
    """
    Run a full rolling-window backtest.
    Returns result dict with match distribution, 5+ events, etc.
    """
    draws = sorted(draws, key=lambda d: d["draw_number"])
    N = len(draws)
    test_draws = draws[min_history:]
    total = len(test_draws)

    # Pre-build feature rows if using ML models
    predictor = None
    if model_type in ("rf", "xgb"):
        predictor = DrawPropertyPredictor(model_type=model_type)
        # Try to load saved model, else train fresh
        predictor.load()
        if not predictor.models:
            print(f"  Training {model_type.upper()} model...")
            feature_rows = [build_feature_row(i, draws) for i in range(N)]
            X, y = predictor.prepare_data(feature_rows)
            if len(X) > 0:
                predictor.train(X, y)
                predictor.save()

    mc_scorer = MonteCarloCandidateScorer(n_simulations=mc_simulations) if model_type == "monte_carlo" else None

    results = []
    match_dist = Counter()
    five_plus = []
    four_plus = []

    random.seed(42)

    for i, test in enumerate(test_draws):
        history = draws[:min_history + i]

        # Get property predictions if using ML
        property_preds = None
        if predictor and predictor.models:
            feat_row = build_feature_row(min_history + i - 1, draws)
            x_vec = [feat_row.get(col, 0) for col in FEATURE_COLS]
            property_preds = predictor.predict(x_vec)

        # Generate candidates
        raw_candidates = generate_candidates(
            history,
            pool_size=pool_size,
            n_candidates=n_candidates,
            property_filters=property_preds,
        )

        # Pick top N predictions
        if pool_size == 6:
            top_combos = raw_candidates[:n_preds]
        else:
            # For 7/8-pool: expand to 6-combos and pick top unique ones
            expanded = []
            seen_6 = set()
            for pool in raw_candidates[:500]:
                for c in expand_pool_to_combos(pool, 6):
                    if c not in seen_6:
                        seen_6.add(c)
                        expanded.append(c)
                if len(expanded) >= n_preds * 5:
                    break
            top_combos = expanded[:n_preds]

        if mc_scorer and top_combos:
            mc_results = mc_scorer.score_candidates(top_combos, history[-50:])
            top_combos = [tuple(r["combo"]) for r in mc_results[:n_preds]]

        # Score against actual draw
        actual = set(test["nums"])
        best_match, best_combo = 0, None
        all_matches = []
        for combo in top_combos:
            m = len(set(combo) & actual)
            all_matches.append(m)
            if m > best_match:
                best_match, best_combo = m, combo

        match_dist[best_match] += 1
        results.append({
            "draw_number": test["draw_number"],
            "actual": test["nums"],
            "best_match": best_match,
            "best_combo": list(best_combo) if best_combo else [],
            "all_matches": all_matches,
            "property_preds": property_preds or {},
        })

        if best_match >= 5:
            five_plus.append({
                "draw": test["draw_number"],
                "actual": test["nums"],
                "predicted": list(best_combo),
                "matches": best_match,
                "matched_nums": sorted(set(best_combo) & actual),
            })
        if best_match >= 4:
            four_plus.append({
                "draw": test["draw_number"],
                "actual": test["nums"],
                "predicted": list(best_combo),
                "matches": best_match,
                "matched_nums": sorted(set(best_combo) & actual),
            })

        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, total, match_dist, len(five_plus))

    avg_match = sum(r["best_match"] for r in results) / total if total else 0
    three_plus = sum(1 for r in results if r["best_match"] >= 3)
    four_plus_count = sum(1 for r in results if r["best_match"] >= 4)

    # Random baseline for lift calculation
    random.seed(99)
    rand_matches = []
    for d in test_draws:
        best = 0
        for _ in range(n_preds):
            c = set(random.sample(range(1, 50), 6))
            m = len(c & set(d["nums"]))
            if m > best:
                best = m
        rand_matches.append(best)
    rand_avg = sum(rand_matches) / total if total else 0
    lift = (avg_match - rand_avg) / rand_avg * 100 if rand_avg > 0 else 0

    return {
        "model_type": model_type,
        "pool_size": pool_size,
        "n_preds": n_preds,
        "total_tested": total,
        "min_history": min_history,
        "match_distribution": dict(match_dist),
        "avg_match": round(avg_match, 4),
        "rand_avg": round(rand_avg, 4),
        "lift_pct": round(lift, 2),
        "three_plus_count": three_plus,
        "three_plus_rate": round(three_plus / total * 100, 2) if total else 0,
        "four_plus_count": four_plus_count,
        "four_plus_rate": round(four_plus_count / total * 100, 2) if total else 0,
        "five_plus_count": len(five_plus),
        "five_plus_rate": round(len(five_plus) / total * 100, 3) if total else 0,
        "five_plus_events": five_plus,
        "four_plus_events": four_plus[:20],
        "rand_3plus_rate": round(sum(1 for r in rand_matches if r >= 3) / total * 100, 2),
    }
