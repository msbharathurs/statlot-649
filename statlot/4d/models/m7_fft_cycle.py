"""
M7 — FFT Cycle Detector
For each of the 10,000 possible numbers, creates a binary time series
(appeared = 1, not appeared = 0) across all draws, then runs FFT to detect
dominant periodicity. Numbers whose cycle predicts an appearance "now" score higher.
"""

import duckdb
import numpy as np
import pickle
import os

DB_PATH   = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
MODEL_DIR = os.path.expanduser("~/statlot-649/models")
os.makedirs(MODEL_DIR, exist_ok=True)

ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]


def train_m7(train_up_to_draw: int, min_appearances: int = 5) -> dict:
    """
    Run FFT cycle detection on all numbers with >= min_appearances.
    Returns dict: number -> {dominant_period, phase_score, cycle_score}
    """
    con = duckdb.connect(DB_PATH, read_only=True)

    draws_list = con.execute(f"""
        SELECT draw_id FROM draws
        WHERE draw_id <= {train_up_to_draw}
        ORDER BY draw_id
    """).fetchall()
    draw_ids = [r[0] for r in draws_list]
    n_draws  = len(draw_ids)
    draw_idx = {d: i for i, d in enumerate(draw_ids)}

    # Appearance matrix: number -> set of draw_ids
    rows = con.execute(f"""
        SELECT number, draw_id
        FROM draw_numbers
        WHERE draw_id <= {train_up_to_draw}
    """).fetchall()
    con.close()

    from collections import defaultdict
    appearances = defaultdict(set)
    for number, draw_id in rows:
        appearances[number].add(draw_id)

    model = {}
    n_processed = 0

    for num in ALL_NUMBERS:
        apps = appearances.get(num, set())
        if len(apps) < min_appearances:
            model[num] = {"cycle_score": 0.0, "dominant_period": None}
            continue

        # Build binary time series
        ts = np.zeros(n_draws, dtype=np.float32)
        for d in apps:
            if d in draw_idx:
                ts[draw_idx[d]] = 1.0

        # Remove mean (detrend)
        ts -= ts.mean()

        # FFT
        fft_vals = np.fft.rfft(ts)
        power    = np.abs(fft_vals) ** 2
        freqs    = np.fft.rfftfreq(n_draws)

        # Ignore DC (freq=0) and very high freqs (noise)
        # Focus on periods between 5 and 500 draws
        valid = (freqs > 1.0/500) & (freqs < 1.0/5)
        if not valid.any():
            model[num] = {"cycle_score": 0.0, "dominant_period": None}
            continue

        power_valid  = power.copy()
        power_valid[~valid] = 0

        dom_idx    = np.argmax(power_valid)
        dom_freq   = freqs[dom_idx]
        dom_power  = power_valid[dom_idx]
        dom_period = round(1.0 / dom_freq) if dom_freq > 0 else None

        # Phase: where are we in the dominant cycle relative to last appearance?
        last_seen   = max(apps)
        last_idx    = draw_idx.get(last_seen, 0)
        current_idx = n_draws - 1

        if dom_period and dom_period > 0:
            gap_from_last = current_idx - last_idx
            phase_in_cycle = gap_from_last % dom_period
            # Score highest when we're near the expected re-appearance point
            # i.e., phase_in_cycle ≈ dom_period
            proximity = 1.0 - abs(phase_in_cycle - dom_period) / dom_period
            cycle_score = float(proximity * (dom_power / (power.sum() + 1e-9)))
        else:
            cycle_score = 0.0

        model[num] = {
            "dominant_period": dom_period,
            "dom_power_ratio": float(dom_power / (power.sum() + 1e-9)),
            "cycle_score":     round(float(cycle_score), 6),
            "last_seen_idx":   last_idx,
            "times_seen":      len(apps),
        }
        n_processed += 1

    print(f"  FFT processed {n_processed} numbers with >={min_appearances} appearances")

    # Normalize cycle_scores to [0,1]
    scores = [v["cycle_score"] for v in model.values() if v["cycle_score"] > 0]
    if scores:
        max_s = max(scores)
        for num in model:
            if model[num]["cycle_score"] > 0:
                model[num]["cycle_score"] = model[num]["cycle_score"] / max_s

    return model


def score_m7(model: dict, candidate: str) -> float:
    entry = model.get(candidate)
    return entry["cycle_score"] if entry else 0.0


def predict_m7(model: dict, top_n: int = 100) -> tuple:
    scores = {n: score_m7(model, n) for n in ALL_NUMBERS}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in ranked[:top_n]], scores


def save_m7(model: dict, suffix: str = ""):
    path = os.path.join(MODEL_DIR, f"m7_fft{suffix}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"M7 saved → {path}")


def load_m7(suffix: str = "") -> dict:
    path = os.path.join(MODEL_DIR, f"m7_fft{suffix}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
    con.close()

    print(f"Training M7 FFT on draws 1–{max_draw} ...")
    model = train_m7(max_draw, min_appearances=5)

    top10, scores = predict_m7(model, top_n=10)
    print(f"\nTop-10 by cycle score: {top10}")
    for n in top10:
        e = model[n]
        print(f"  {n}  period={e.get('dominant_period')}  score={e['cycle_score']:.4f}")

    save_m7(model)
    print("M7 done.")
