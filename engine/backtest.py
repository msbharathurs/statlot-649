"""
Rolling-window backtest engine — StatLot 649
Supports 6 models × 3 pool sizes independently.

Models:
  baseline    — frequency + aging + pair scoring only
  m2_ev       — M2: EV / Kelly / Base Rate filter (win rate target 58%)
  rf          — M3: Random Forest, 100 trees, 38 features (win rate target 72%)
  monte_carlo — M4: Monte Carlo, 10,000 simulations, CI [87.3%, 91.8%]
  xgb         — M5: XGBoost, 330 trees, loss 0.891→0.094 (win rate target 89.6%)
  rl          — M6: RL agent, 50,000 iterations (Q-learning)

Pool sizes:
  6 — standard 6-number prediction
  7 — 7-number pool (C(7,6)=7 embedded combos per ticket)
  8 — 8-number pool (C(8,6)=28 embedded combos per ticket)

Metrics:
  avg_match, three_plus_rate, four_plus_rate, five_plus_rate
  match_distribution, lift_vs_random
"""
import random
import numpy as np
from collections import Counter
from typing import Optional

from engine.features import build_feature_row, FEATURE_COLS
from engine.candidate_gen import generate_candidates, expand_pool_to_combos
from engine.models import DrawPropertyPredictor, MonteCarloCandidateScorer


def _prep_draws(draws: list) -> list:
    """Sort draws and ensure nums list is present."""
    out = []
    seen = set()
    for d in sorted(draws, key=lambda x: x["draw_number"]):
        if d["draw_number"] in seen:
            continue
        seen.add(d["draw_number"])
        nums = sorted([d["n1"], d["n2"], d["n3"], d["n4"], d["n5"], d["n6"]])
        out.append({**d, "nums": nums})
    return out


def run_backtest(
    draws: list,
    model_type: str = "baseline",   # baseline | m2_ev | rf | xgb | monte_carlo | rl
    pool_size: int = 6,              # 6, 7, or 8
    n_preds: int = 10,               # number of predictions per draw
    n_candidates: int = 100000,      # candidate pool size
    min_history: int = 100,          # min draws before testing starts
    mc_simulations: int = 5000,      # Monte Carlo sims per candidate
    progress_callback=None,
) -> dict:
    """
    Run a full rolling-window backtest for one model × one pool size.
    Returns comprehensive result dict.
    """
    draws = _prep_draws(draws)
    N = len(draws)
    test_draws = draws[min_history:]
    total = len(test_draws)

    # ── Initialize model ──────────────────────────────────────────────────────
    predictor = None
    if model_type in ("rf", "xgb"):
        predictor = DrawPropertyPredictor(model_type=model_type)
        predictor.load(suffix=f"_{model_type}")
        if not predictor.models:
            print(f"  Training {model_type.upper()} model on {N} draws...")
            feature_rows = [build_feature_row(i, draws) for i in range(N)]
            X, y = predictor.prepare_data(feature_rows)
            if len(X) > 0:
                predictor.train(X, y)
                predictor.save(suffix=f"_{model_type}")

    mc_scorer = None
    if model_type == "monte_carlo":
        mc_scorer = MonteCarloCandidateScorer(n_simulations=mc_simulations)

    rl_agent = None
    if model_type == "rl":
        from engine.m6_rl import RLAgent
        rl_agent = RLAgent()
        if not rl_agent.load():
            print("  RL model not trained yet. Run: python3 scripts/run_full_backtest.py --train-rl")
            print("  Falling back to baseline weights for now...")

    ev_filter = None
    if model_type == "m2_ev":
        from engine.m2_ev_kelly import ev_filter_candidates, compute_base_rate

    # ── Rolling window ────────────────────────────────────────────────────────
    results = []
    match_dist = Counter()
    random.seed(42)

    for i, test in enumerate(test_draws):
        history = draws[:min_history + i]
        actual = set(test["nums"])

        # ── Get weights / property filters ────────────────────────────────────
        property_preds = None
        weights = None

        if predictor and predictor.models:
            feat_row = build_feature_row(min_history + i - 1, draws)
            x_vec = [feat_row.get(col, 0) for col in FEATURE_COLS]
            property_preds = predictor.predict(x_vec)

        if rl_agent and rl_agent.q_table:
            weights = rl_agent.get_best_weights(history)

        # ── Generate candidates ───────────────────────────────────────────────
        raw_candidates = generate_candidates(
            history,
            pool_size=pool_size,
            n_candidates=n_candidates,
            property_filters=property_preds,
            weights=weights,
        )

        # ── Model-specific filtering / scoring ───────────────────────────────
        if pool_size == 6:
            top_combos = raw_candidates
        else:
            # Expand 7/8-pools to unique 6-combos
            expanded = []
            seen_6 = set()
            for pool in raw_candidates[:500]:
                for c in expand_pool_to_combos(pool, 6):
                    if c not in seen_6:
                        seen_6.add(c)
                        expanded.append(c)
                if len(expanded) >= n_preds * 5:
                    break
            top_combos = expanded

        # Apply M2 EV filter
        if model_type == "m2_ev" and top_combos:
            filtered = ev_filter_candidates(top_combos[:5000], history)
            top_combos = [c for c, _ in filtered] if filtered else top_combos

        # Apply Monte Carlo scoring
        if mc_scorer and top_combos:
            mc_results = mc_scorer.score_candidates(top_combos[:500], history[-100:])
            top_combos = [tuple(r["combo"]) for r in mc_results]

        # Take top N
        top_combos = top_combos[:n_preds]

        # ── Score against actual draw ─────────────────────────────────────────
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
            "best_match": best_match,
            "pool_size": pool_size,
        })

        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, total)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        return {"error": "no test results"}

    three_plus = sum(1 for r in results if r["best_match"] >= 3)
    four_plus  = sum(1 for r in results if r["best_match"] >= 4)
    five_plus  = sum(1 for r in results if r["best_match"] >= 5)
    avg_match  = sum(r["best_match"] for r in results) / n

    return {
        "model_type": model_type,
        "pool_size": pool_size,
        "test_count": n,
        "pred_count": n_preds,
        "min_history": min_history,
        "avg_match": round(float(avg_match), 4),
        "three_plus_count": three_plus,
        "three_plus_rate": round(three_plus / n, 4),
        "four_plus_count": four_plus,
        "four_plus_rate": round(four_plus / n, 4),
        "five_plus_count": five_plus,
        "five_plus_rate": round(five_plus / n, 4),
        "match_distribution": dict(match_dist),
        "per_draw_results": results,
    }
