"""
check_wins.py
Loads the LATEST prediction file and compares against the latest scraped draw result.
Outputs a win summary JSON and writes results to results_log in draws_4d.duckdb.
"""
import json, os, glob
from itertools import permutations
from datetime import datetime

PRED_DIR    = os.path.expanduser("~/statlot-649/predictions")
RESULT_FILE = os.path.expanduser("~/statlot-649/predictions/latest_draw.json")
OUT_FILE    = os.path.expanduser("~/statlot-649/predictions/win_check.json")

def ibox_family(n):
    return set(''.join(p) for p in permutations(n))

def all_numbers_in_draw(draw):
    nums = {}
    for key in ["prize_1st","prize_2nd","prize_3rd"] + \
               [f"starter_{i}" for i in range(1,11)] + \
               [f"consolation_{i}" for i in range(1,11)]:
        v = draw.get(key,"")
        if v and len(str(v)) == 4:
            nums[str(v)] = key
    return nums

def tier_of(num, draw):
    if num == draw.get("prize_1st"): return "1st Prize 🥇"
    if num == draw.get("prize_2nd"): return "2nd Prize 🥈"
    if num == draw.get("prize_3rd"): return "3rd Prize 🥉"
    for i in range(1,11):
        if num == draw.get(f"starter_{i}"): return "Starter"
    for i in range(1,11):
        if num == draw.get(f"consolation_{i}"): return "Consolation"
    return None

# Load latest draw result
if not os.path.exists(RESULT_FILE):
    print("[ERROR] No latest draw result found. Run scrape_latest.py first.")
    exit(1)
with open(RESULT_FILE) as f:
    draw = json.load(f)

# Find the LATEST prediction file (most recently generated)
pred_files = glob.glob(os.path.join(PRED_DIR, "predict_*.json"))
if not pred_files:
    print("[ERROR] No prediction files found in", PRED_DIR)
    exit(1)
pred_files.sort(key=os.path.getmtime, reverse=True)
latest_pred_file = pred_files[0]
print(f"[CHECK] Using prediction file: {latest_pred_file}")
with open(latest_pred_file) as f:
    pred = json.load(f)

top50 = [x["number"] for x in pred.get("top50", [])]
top10 = [x["number"] for x in pred.get("top10", [])]

# Exact wins
wins = []
for num in top50:
    tier = tier_of(num, draw)
    if tier:
        wins.append({"number": num, "tier": tier, "in_top10": num in top10})

# iBox family wins
ibox_wins = []
for fam_info in pred.get("top_ibox_families", []):
    family_perms = set(fam_info["perms"])
    for perm in family_perms:
        tier = tier_of(perm, draw)
        if tier:
            ibox_wins.append({
                "family": fam_info["family"],
                "matched_number": perm,
                "tier": tier
            })

summary = {
    "checked_at": datetime.now().isoformat(),
    "prediction_file": os.path.basename(latest_pred_file),
    "draw_date": draw.get("draw_date"),
    "actual_1st": draw.get("prize_1st"),
    "actual_2nd": draw.get("prize_2nd"),
    "actual_3rd": draw.get("prize_3rd"),
    "exact_wins": wins,
    "ibox_family_wins": ibox_wins,
    "top50_size": len(top50),
    "any_win": len(wins) > 0 or len(ibox_wins) > 0,
}

with open(OUT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

# ── DuckDB write ──────────────────────────────────────────────────────────────
import duckdb as _ddb
_con = _ddb.connect(os.path.expanduser("~/statlot-649/draws_4d.duckdb"))

# Find the latest prediction_log row for this draw date
pred_rows = _con.execute(
    "SELECT id FROM predictions_log WHERE draw_date = ? ORDER BY id DESC LIMIT 1",
    [draw.get("draw_date")]
).fetchall()
pred_id = pred_rows[0][0] if pred_rows else None

if wins or ibox_wins:
    for w in wins:
        _con.execute("""
            INSERT INTO results_log (draw_date, prediction_id, matched, prize_group, notes)
            VALUES (?, ?, ?, ?, ?)
        """, [draw.get("draw_date"), pred_id, True, w["tier"], f"exact:{w['number']}"])
    for w in ibox_wins:
        _con.execute("""
            INSERT INTO results_log (draw_date, prediction_id, matched, prize_group, notes)
            VALUES (?, ?, ?, ?, ?)
        """, [draw.get("draw_date"), pred_id, True, w["tier"], f"ibox:{w['matched_number']}"])
else:
    _con.execute("""
        INSERT INTO results_log (draw_date, prediction_id, matched, notes)
        VALUES (?, ?, ?, ?)
    """, [draw.get("draw_date"), pred_id, False, "no_wins"])

_con.close()
print("[DB] Written to results_log ✅")
# ─────────────────────────────────────────────────────────────────────────────

print(json.dumps(summary, indent=2))
