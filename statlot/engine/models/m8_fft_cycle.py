"""
m8_fft_cycle.py — FFT Cycle Detector for StatLot 649
Replaces dead GMM (avg weight 0.029). Ports FFT logic from 4D M7.

Core idea: each number has a draw-presence time series (1 if appeared, 0 if not).
FFT reveals dominant periodicity. Numbers whose cycle predicts them "due" score high.
Also scores combos by how well their individual cycles are aligned (all due together).
"""
import numpy as np
from collections import defaultdict

def train_m8_fft(history: list[list[int]], n_numbers: int = 49, lookback: int = 200) -> dict:
    """
    Train FFT cycle model on draw history.
    history: list of draws (each draw = list of 6 ints), newest last.
    Returns a dict of per-number cycle metadata.
    """
    recent = history[-lookback:] if len(history) >= lookback else history
    N = len(recent)

    model = {}
    for num in range(1, n_numbers + 1):
        # Binary presence series
        series = np.array([1.0 if num in draw else 0.0 for draw in recent])

        # Detrend — remove mean to focus on oscillations
        series -= series.mean()

        # FFT
        fft_vals = np.fft.rfft(series)
        power = np.abs(fft_vals) ** 2

        # Dominant frequency (skip DC component at index 0)
        freqs = np.fft.rfftfreq(N)
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        dominant_period = (1.0 / dominant_freq) if dominant_freq > 0 else N
        dominant_power = float(power[dominant_idx])

        # Phase — where in its cycle is this number right now?
        phase = np.angle(fft_vals[dominant_idx])

        # Draws since last appearance
        last_seen = None
        for i, draw in enumerate(reversed(recent)):
            if num in draw:
                last_seen = i
                break
        draws_since = last_seen if last_seen is not None else N

        # "Due score" — how far past its expected period is the number?
        # Positive = overdue, negative = recently appeared
        expected_gap = dominant_period
        due_score = float((draws_since - expected_gap) / max(expected_gap, 1.0))

        # Cycle alignment score — are we near the peak of the cycle?
        # Phase of 0 or 2π = peak (most likely to appear)
        # Normalise to [0,1] where 1 = at peak
        current_phase_position = float(np.cos(phase))  # 1 at peak, -1 at trough

        model[num] = {
            "period": float(dominant_period),
            "power": dominant_power,
            "phase": float(phase),
            "draws_since": draws_since,
            "due_score": due_score,
            "cycle_alignment": current_phase_position,
        }

    return model


def score_m8_fft(model: dict, combo: list[int]) -> float:
    """
    Score a combo by its FFT cycle signal.
    Combines:
      - Mean due_score (positive = overdue numbers)
      - Mean cycle_alignment (are these numbers near cycle peaks?)
      - Power-weighted signal (trust numbers with strong periodicity more)
    """
    if not model:
        return 0.0

    due_scores = []
    alignments = []
    powers = []

    for n in combo:
        if n in model:
            m = model[n]
            due_scores.append(m["due_score"])
            alignments.append(m["cycle_alignment"])
            powers.append(m["power"])

    if not due_scores:
        return 0.0

    powers_arr = np.array(powers)
    total_power = powers_arr.sum()
    if total_power > 0:
        weights = powers_arr / total_power
    else:
        weights = np.ones(len(powers)) / len(powers)

    weighted_due = float(np.dot(weights, due_scores))
    weighted_align = float(np.dot(weights, alignments))

    # Combine: overdue numbers that are also at cycle peak = strong signal
    # Scale to roughly [0, 1] range
    raw = 0.6 * weighted_due + 0.4 * weighted_align
    # Sigmoid to bound output
    score = float(1.0 / (1.0 + np.exp(-raw)))
    return score
