"""
toto/check_wins.py — Compare last prediction against actual draw, save to DuckDB.
"""
import json, sys, duckdb
from datetime import datetime
sys.path.insert(0, "/home/ubuntu/statlot-649/statlot")
from toto.toto_prizes import check_ordinary, check_system_entry

DB_PATH    = "/home/ubuntu/statlot-649/statlot_toto.duckdb"
DRAW_FILE  = "/home/ubuntu/statlot-649/statlot/toto/latest_draw.json"
STATUS_FILE= "/home/ubuntu/statlot-649/logs/toto_pipeline_status.json"

def write_status(step, status):
    with open(STATUS_FILE, "w") as f:
        json.dump({"step": step, "status": status, "ts": datetime.utcnow().isoformat()}, f)

GROUP_LABELS = {1:"Group 1 (JACKPOT!🎰)",2:"Group 2 (5+add 🔥)",3:"Group 3 (5 nums ✅)",
                4:"Group 4 (4+add)",5:"Group 5 ($50)",6:"Group 6 ($25)",7:"Group 7 ($10)"}

def run():
    write_status("check_wins", "running")

    # Load actual draw
    with open(DRAW_FILE) as f:
        draw = json.load(f)
    winning = draw["numbers"]
    additional = draw["additional"]
    draw_number = draw["draw_number"]
    draw_date = draw["draw_date"]
    print(f"\nChecking draw #{draw_number} ({draw_date})")
    print(f"Winning: {winning} + Additional: {additional}")

    con = duckdb.connect(DB_PATH)

    # Get latest prediction
    row = con.execute("""
        SELECT id, sys6_t1, sys6_t2, sys6_t3, sys7_t1,
               sys8_t1, sys9_t1, sys10_t1, sys11_t1, sys12_t1, bonus_t6
        FROM toto_predictions
        ORDER BY generated_at DESC LIMIT 1
    """).fetchone()

    if not row:
        print("No predictions found in DB!")
        write_status("check_wins", "no_prediction")
        return None

    pred_id = row[0]
    tickets = {
        "sys6_t1": json.loads(row[1]) if row[1] else None,
        "sys6_t2": json.loads(row[2]) if row[2] else None,
        "sys6_t3": json.loads(row[3]) if row[3] else None,
        "sys7_t1": json.loads(row[4]) if row[4] else None,
        "sys8_t1": json.loads(row[5]) if row[5] else None,
        "sys9_t1": json.loads(row[6]) if row[6] else None,
        "sys10_t1":json.loads(row[7]) if row[7] else None,
        "sys11_t1":json.loads(row[8]) if row[8] else None,
        "sys12_t1":json.loads(row[9]) if row[9] else None,
        "bonus_t6": json.loads(row[10]) if row[10] else None,
    }

    results = {}
    best_group = None
    total_mandatory = 0.0
    total_full = 0.0
    any_win = False

    print("\n--- Prize Check ---")
    for name, nums in tickets.items():
        if nums is None:
            results[name] = None
            continue
        if len(nums) == 6:
            r = check_ordinary(nums, winning, additional)
        else:
            r = check_system_entry(nums, winning, additional)
        results[name] = r
        g = r["best_group"]
        prize_fixed = r["total_fixed_prize_sgd"]
        label = GROUP_LABELS.get(g, "No win") if g else "No win"
        print(f"  {name}: {nums} → {label} | Fixed prize: ${prize_fixed:.0f}")
        if g:
            any_win = True
            if best_group is None or g < best_group:
                best_group = g
            # mandatory = sys6 x3 + sys7
            if name in ("sys6_t1","sys6_t2","sys6_t3","sys7_t1"):
                total_mandatory += prize_fixed
            total_full += prize_fixed

    print(f"\nBest group: {GROUP_LABELS.get(best_group,'No win')}")
    print(f"Total prize (mandatory only): ${total_mandatory:.0f}")
    print(f"Total prize (all sys): ${total_full:.0f}")
    print(f"Any win: {any_win}")

    # Upsert actual draw into toto_draws
    con.execute("""
        INSERT OR REPLACE INTO toto_draws
            (draw_number, draw_date, n1,n2,n3,n4,n5,n6, additional)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, [draw_number, draw_date] + winning + [additional])

    # Insert result
    result_id = f"{pred_id}_result"
    def jr(r): return json.dumps(r) if r else None

    con.execute("""
        INSERT OR REPLACE INTO toto_results
            (id, prediction_id, draw_number, draw_date,
             actual_n1,actual_n2,actual_n3,actual_n4,actual_n5,actual_n6,actual_additional,
             sys6_t1_result,sys6_t2_result,sys6_t3_result,sys7_t1_result,
             sys8_t1_result,sys9_t1_result,sys10_t1_result,sys11_t1_result,sys12_t1_result,
             bonus_t6_result,
             best_group, total_prize_mandatory, total_prize_full,
             total_cost_mandatory, any_win)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [result_id, pred_id, draw_number, draw_date] +
        winning + [additional] +
        [jr(results.get(k)) for k in ("sys6_t1","sys6_t2","sys6_t3","sys7_t1",
                                       "sys8_t1","sys9_t1","sys10_t1","sys11_t1","sys12_t1",
                                       "bonus_t6")] +
        [best_group, total_mandatory, total_full, 10.0, any_win])

    # Update prediction record with draw number
    con.execute("""
        UPDATE toto_predictions SET draw_number=?, draw_date=? WHERE id=?
    """, [draw_number, draw_date, pred_id])

    con.close()
    write_status("check_wins", "done")
    return {"draw": draw, "best_group": best_group, "total_mandatory": total_mandatory,
            "total_full": total_full, "any_win": any_win, "results": results}

if __name__ == "__main__":
    run()
