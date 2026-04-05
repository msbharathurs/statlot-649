"""
retrain_and_predict.py
Retrains M2+M5+M11 on full updated history, generates predictions for next draw.
Writes top10 + iBox families to predictions_log in draws_4d.duckdb.
"""
import duckdb, numpy as np, json, os, sys
from datetime import datetime
from itertools import permutations

DB_PATH = os.path.expanduser("~/statlot-649/draws_4d.duckdb")
sys.path.insert(0, os.path.dirname(__file__))

from models.m2_markov       import train_m2, score_m2
from models.m5_structure    import train_m5, score_m5
from models.m11_ibox_family import train_m11, score_m11

ALL_NUMBERS = [f"{i:04d}" for i in range(10000)]

def ibox_family(n):
    return set(''.join(p) for p in permutations(n))

def next_draw_day():
    from datetime import date, timedelta
    today = date.today()
    # Next draw: Wed=2, Sat=5, Sun=6
    draw_days = [2, 5, 6]
    for offset in range(1, 8):
        d = today + timedelta(days=offset)
        if d.weekday() in draw_days:
            days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            return d.strftime("%Y-%m-%d"), days[d.weekday()][:3]
    return (today + timedelta(days=3)).strftime("%Y-%m-%d"), "Wed"

con = duckdb.connect(DB_PATH, read_only=True)
max_draw = con.execute("SELECT MAX(draw_id) FROM draws").fetchone()[0]
con.close()

print(f"Retraining on {max_draw} draws...", flush=True)
m2  = train_m2(max_draw)
m5  = train_m5(max_draw)
m11 = train_m11(max_draw)

s_m2  = np.array([score_m2(m2, n)                   for n in ALL_NUMBERS])
s_m5  = np.array([score_m5(m5, n, tier_group="1st") for n in ALL_NUMBERS])
s_m11 = score_m11(ALL_NUMBERS, m11)
scores = 0.40*s_m2 + 0.40*s_m5 + 0.20*s_m11

ranked = [(ALL_NUMBERS[i], float(scores[i])) for i in np.argsort(scores)[::-1]]
top50  = ranked[:50]
top10  = ranked[:10]

# iBox families
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
    "weights": {"m2": 0.40, "m5": 0.40, "m11": 0.20},
    "top10":  [{"number": n, "score": s} for n,s in top10],
    "top25":  [{"number": n, "score": s} for n,s in ranked[:25]],
    "top50":  [{"number": n, "score": s} for n,s in top50],
    "top_ibox_families": [
        {"family": ''.join(f), "perms": sorted(set(''.join(p) for p in permutations(''.join(f))))}
        for f,_ in top_families
    ],
}

out = os.path.expanduser(f"~/statlot-649/predictions/predict_{next_dow.lower()}_{next_date.replace('-','')}.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)

# ── DuckDB write ──────────────────────────────────────────────────────────────
import subprocess
model_version = subprocess.check_output(
    ["git", "-C", os.path.expanduser("~/statlot-649"), "rev-parse", "--short", "HEAD"]
).decode().strip()

con_write = duckdb.connect(DB_PATH)

for rank, (num, score) in enumerate(top10, 1):
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
        "M2_markov:0.40,M5_structure:0.40,M11_ibox:0.20",
        f"trained_on_{max_draw}_draws"
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
        "M2_markov:0.40,M5_structure:0.40,M11_ibox:0.20",
        f"ibox_family_{''.join(fam)}"
    ])

con_write.close()
print(f"[DB] Written to predictions_log ✅")
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"NEXT DRAW PREDICTIONS ({next_dow} {next_date})")
print(f"{'='*50}")
print("Top 10:")
for i,(n,s) in enumerate(top10, 1):
    print(f"  {i:2d}. {n}  ({s:.4f})")
print(f"\nTop iBox families:")
for fam,_ in top_families[:3]:
    print(f"  {''.join(fam)} — any of {len(set(ibox_family(''.join(fam))))} perms")
print(f"\nSaved → {out}")
