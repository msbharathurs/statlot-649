"""
retrain_and_predict.py
Retrains M2+M5+M12 on full updated history, generates predictions for next draw.

CONFIG (2026-04-05 — EV optimized):
  Weights: M2=0.60, M5=0.10, M12=0.30  (Config E)
  Primary output: Top 30 picks  (ROI 92.3%, EV -$2.31/draw over 895-draw backtest)
  Top 50 retained in JSON for reference only.
  M11 replaced by M12 coverage model.

EV backtest result that drove Top30 decision:
  Top10: ROI 65.4%  Top20: ROI 67.7%  Top30: ROI 92.3%  Top50: ROI 79.1%
  Top30 is the sweet spot — picks 31-50 are net negative marginal EV.
"""
import duckdb, numpy as np, json, os, sys, importlib.util
from datetime import datetime
from itertools import permutations

DB_PATH = os.path.expanduser("~/statlot-649/draws_4d.duckdb")

# Use importlib to load models from 4d/ dir (avoids "4d" invalid module name issue)
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_BASE = os.path.join(os.path.dirname(__file__), "models")
_m2  = _load(os.path.join(_BASE, "m2_markov.py"),    "m2_markov")
_m5  = _load(os.path.join(_BASE, "m5_structure.py"), "m5_structure")
_m12 = _load(os.path.join(_BASE, "m12_coverage.py"), "m12_coverage")

train_m2  = _m2.train_m2;   score_m2  = _m2.score_m2
train_m5  = _m5.train_m5;   score_m5  = _m5.score_m5
train_m12 = _m12.train_m12; score_m12 = _m12.score_m12

ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]

WEIGHTS = {"m2": 0.60, "m5": 0.10, "m12": 0.30}
TOP_PICKS = 30   # EV-optimized: Top30 ROI 92.3% vs Top50 ROI 79.1%


def ibox_family(n):
    return set(''.join(p) for p in permutations(n))


def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def next_draw_day():
    """
    Return (date_str, day_abbr) for the draw to predict for.
    4D draw days: Wednesday=2, Saturday=5, Sunday=6.
    If today IS a draw day AND current SGT time is before 18:30 → return today.
    Otherwise return the next draw day.
    """
    from datetime import date, timedelta
    import pytz

    sgt = pytz.timezone("Asia/Singapore")
    now_sgt = datetime.now(sgt)
    today = now_sgt.date()
    draw_days = [2, 5, 6]
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    cutoff_hour, cutoff_min = 18, 30
    if today.weekday() in draw_days:
        if (now_sgt.hour, now_sgt.minute) < (cutoff_hour, cutoff_min):
            return today.strftime("%Y-%m-%d"), day_names[today.weekday()][:3]

    for offset in range(1, 8):
        d = today + timedelta(days=offset)
        if d.weekday() in draw_days:
            return d.strftime("%Y-%m-%d"), day_names[d.weekday()][:3]

    return (today + timedelta(days=3)).strftime("%Y-%m-%d"), "Wed"


# ── Train ─────────────────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
con.close()

print(f"Retraining on {max_draw} draws...", flush=True)
m2  = train_m2(max_draw)
m5  = train_m5(max_draw, recency_window=200)
m12 = train_m12(max_draw, recent_window=200)

s_m2  = normalize(np.array([score_m2(m2, n)                   for n in ALL_NUMBERS]))
s_m5  = normalize(np.array([score_m5(m5, n, tier_group="1st") for n in ALL_NUMBERS]))
s_m12 = score_m12(ALL_NUMBERS, m12)

scores = WEIGHTS["m2"]*s_m2 + WEIGHTS["m5"]*s_m5 + WEIGHTS["m12"]*s_m12
ranked = [(ALL_NUMBERS[i], float(scores[i])) for i in np.argsort(scores)[::-1]]

top30 = ranked[:TOP_PICKS]   # PRIMARY — EV optimized
top50 = ranked[:50]          # retained in JSON for reference only

# iBox families (from top 200 by score)
family_scores = {}
for num, sc in ranked[:200]:
    fam = tuple(sorted(num))
    family_scores[fam] = family_scores.get(fam, 0) + sc
top_families = sorted(family_scores.items(), key=lambda x: -x[1])[:5]

next_date, next_dow = next_draw_day()

result = {
    "generated": datetime.now().isoformat(),
    "draw_day": next_dow,
    "draw_date": next_date,
    "trained_on_draws": max_draw,
    "weights": WEIGHTS,
    "ev_config": "Top30 EV-optimized (ROI 92.3% over 895-draw backtest)",
    "top30_recommended": [{"number": n, "score": s} for n, s in top30],
    "top50_reference":   [{"number": n, "score": s} for n, s in top50],
    "top_ibox_families": [
        {
            "family": ''.join(f),
            "perms": sorted(set(''.join(p) for p in permutations(''.join(f))))
        }
        for f, _ in top_families
    ],
}

os.makedirs(os.path.expanduser("~/statlot-649/predictions"), exist_ok=True)
out = os.path.expanduser(
    f"~/statlot-649/predictions/predict_{next_dow.lower()}_{next_date.replace('-','')}.json"
)
with open(out, "w") as f:
    json.dump(result, f, indent=2)

# ── DuckDB write (Top30 only) ─────────────────────────────────────────────────
import subprocess
model_version = subprocess.check_output(
    ["git", "-C", os.path.expanduser("~/statlot-649"), "rev-parse", "--short", "HEAD"]
).decode().strip()

models_used = f"M2_markov:{WEIGHTS['m2']},M5_structure:{WEIGHTS['m5']},M12_coverage:{WEIGHTS['m12']}"

con_write = duckdb.connect(DB_PATH)
for rank, (num, score) in enumerate(top30, 1):
    con_write.execute("""
        INSERT INTO predictions_log
        (draw_date, model_version, position, predicted_number, confidence, models_used, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        next_date,
        model_version,
        f"top{rank}",
        num,
        float(score),
        models_used,
        f"trained_on_{max_draw}_draws_top30_ev_config"
    ])

for rank, (fam, _) in enumerate(top_families[:3], 1):
    perms = sorted(set(''.join(p) for p in permutations(''.join(fam))))
    con_write.execute("""
        INSERT INTO predictions_log
        (draw_date, model_version, position, predicted_number, confidence, models_used, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        next_date,
        model_version,
        f"ibox_family_{rank}",
        ','.join(perms),
        None,
        models_used,
        f"ibox_family_{''.join(fam)}"
    ])
con_write.close()
print(f"[DB] Written to predictions_log ✅ (top30 + 3 ibox families)")

# ── Print output ──────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"NEXT DRAW PREDICTIONS ({next_dow} {next_date})")
print(f"Config E: M2=0.60 M5=0.10 M12=0.30 | Top30 EV-optimized")
print(f"{'='*55}")
print(f"Top 30 Exact bets (recommended — $30/draw):")
for i, (n, s) in enumerate(top30, 1):
    print(f"  {i:2d}. {n}  ({s:.4f})")
print(f"\nTop iBox families:")
for fam, _ in top_families[:3]:
    fam_str = ''.join(fam)
    print(f"  {fam_str} — {len(set(ibox_family(fam_str)))} permutations")
print(f"\nSaved → {out}")
print(f"[Note] Top50 reference retained in JSON. DB write = Top30 only.")
