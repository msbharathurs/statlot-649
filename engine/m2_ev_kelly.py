"""
M2 — EV / Kelly / Base Rate Engine
Three formulas:
  1. Expected Value (EV): probability-weighted payoff
  2. Kelly Criterion: optimal bet sizing fraction
  3. Base Rate Win Rate: historical 3+ match frequency as prior

Win rate target: 58% (3+ match rate on filtered combos)
Purpose: stop losing — eliminate negative EV combos before ML filtering.
"""
import numpy as np
from collections import Counter
from typing import List, Tuple


# Lotto 6/49 prize structure (CAD approximate, normalized to unit bet=1)
PRIZE_TABLE = {
    6: 5_000_000,   # jackpot (rough expected after pool split)
    5: 2_500,       # 5/6
    4: 100,         # 4/6
    3: 10,          # 3/6
    2: 3,           # 2/6 (free play ≈ $3)
    1: 0,
    0: 0,
}

# Odds of matching exactly k of 6 from pool of 49
# P(match k) = C(6,k)*C(43,6-k) / C(49,6)
def _comb(n, k):
    from math import comb
    return comb(n, k)

TOTAL_COMBOS = _comb(49, 6)  # 13,983,816

def exact_match_prob(k: int) -> float:
    """P(exactly k matches in a random 6/49 draw)"""
    return _comb(6, k) * _comb(43, 6 - k) / TOTAL_COMBOS

EXACT_PROBS = {k: exact_match_prob(k) for k in range(7)}


def compute_ev(combo: tuple, historical_draws: list, ticket_cost: float = 3.0) -> dict:
    """
    Compute Expected Value for a combo using:
    - Historical frequency prior (how often each number appears)
    - Adjusted probabilities based on frequency weighting
    - Prize table

    Returns EV dict with ev, kelly_fraction, is_positive_ev.
    """
    freq = Counter()
    n_draws = len(historical_draws)
    for d in historical_draws:
        for n in [d["n1"], d["n2"], d["n3"], d["n4"], d["n5"], d["n6"]]:
            freq[n] += 1

    # Frequency-adjusted probability for each number in combo
    # Base uniform prob = 6/49 per number
    # Adjusted = freq[n] / (n_draws * 6) * 49 / 6  (normalize to relative freq)
    base_prob = 6 / 49
    combo_nums = list(combo)
    adj_probs = []
    for n in combo_nums:
        historical_rate = freq.get(n, 0) / (n_draws * 6) if n_draws > 0 else base_prob / 49
        adj = 0.5 * base_prob + 0.5 * historical_rate * 49  # blend historical + uniform
        adj_probs.append(min(adj, 1.0))

    # Expected number of matches (each number independently approx)
    expected_matches = sum(adj_probs)

    # EV calculation using match probability distribution
    # We use a simple multinomial approximation
    ev = 0.0
    for k in range(7):
        prize = PRIZE_TABLE.get(k, 0)
        p = EXACT_PROBS[k]
        ev += prize * p
    ev -= ticket_cost

    # Adjusted EV using frequency-weighted expected matches
    match_ratio = expected_matches / 6.0  # ratio vs theoretical 6/49 expectation
    adj_ev = ev * max(0.5, min(2.0, match_ratio))  # cap adjustment at 2x

    # Kelly fraction: f* = (b*p - q) / b
    # where b = net odds, p = prob of 3+ match (positive outcome), q = 1-p
    p_win = sum(EXACT_PROBS[k] for k in [3, 4, 5, 6])  # P(3+ matches) ~ 0.018
    p_loss = 1 - p_win
    net_odds = PRIZE_TABLE[3] / ticket_cost  # use 3-match as representative win
    kelly = (net_odds * p_win - p_loss) / net_odds if net_odds > 0 else 0

    return {
        "ev_raw": round(float(ev), 4),
        "ev_adj": round(float(adj_ev), 4),
        "expected_matches": round(float(expected_matches), 3),
        "kelly_fraction": round(float(kelly), 6),
        "p_3plus": round(float(p_win), 6),
        "is_positive_ev": bool(adj_ev > -ticket_cost * 0.5),  # accept if EV > -$1.50
    }


def compute_base_rate(historical_draws: list, n_preds: int = 10,
                       lookback: int = 200) -> dict:
    """
    Base Rate Analysis:
    - What is the actual historical 3+ match rate?
    - What structural features correlate with 3+ match draws?
    Returns base rate stats to calibrate win probability prior.
    """
    if len(historical_draws) < 50:
        return {"base_rate_3plus": 0.018, "base_rate_4plus": 0.001, "calibrated": False}

    # Use last `lookback` draws as reference
    window = historical_draws[-lookback:]
    match_3plus = 0
    match_4plus = 0
    total_pairs = 0

    # For each pair of draws, count overlap
    for i in range(len(window)):
        for j in range(i + 1, min(i + 20, len(window))):
            s1 = set([window[i]["n1"], window[i]["n2"], window[i]["n3"],
                      window[i]["n4"], window[i]["n5"], window[i]["n6"]])
            s2 = set([window[j]["n1"], window[j]["n2"], window[j]["n3"],
                      window[j]["n4"], window[j]["n5"], window[j]["n6"]])
            overlap = len(s1 & s2)
            if overlap >= 3:
                match_3plus += 1
            if overlap >= 4:
                match_4plus += 1
            total_pairs += 1

    base_3 = match_3plus / total_pairs if total_pairs > 0 else 0.018
    base_4 = match_4plus / total_pairs if total_pairs > 0 else 0.001

    # Sum/structural distribution
    sums = [d["n1"]+d["n2"]+d["n3"]+d["n4"]+d["n5"]+d["n6"] for d in window]
    sum_mean = float(np.mean(sums))
    sum_std = float(np.std(sums))

    odds = [sum(1 for x in [d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]] if x%2!=0)
            for d in window]
    odd_mode = int(Counter(odds).most_common(1)[0][0])

    return {
        "base_rate_3plus": round(base_3, 4),
        "base_rate_4plus": round(base_4, 4),
        "sum_mean": round(sum_mean, 1),
        "sum_std": round(sum_std, 1),
        "sum_range_68pct": [round(sum_mean - sum_std, 0), round(sum_mean + sum_std, 0)],
        "dominant_odd_count": odd_mode,
        "calibrated": True,
        "lookback_draws": len(window),
    }


def ev_filter_candidates(candidates: list, historical_draws: list,
                          ticket_cost: float = 3.0) -> list:
    """
    Filter candidate combos by EV. Remove clear negative-EV outliers.
    Returns sorted list: (ev_adj, combo).
    Target: keep top combos with highest adjusted EV.
    """
    scored = []
    for combo in candidates:
        ev_data = compute_ev(combo, historical_draws, ticket_cost)
        if ev_data["is_positive_ev"]:
            scored.append((ev_data["ev_adj"], combo, ev_data))

    scored.sort(reverse=True)
    return [(combo, ev_data) for _, combo, ev_data in scored]
