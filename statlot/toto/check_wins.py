"""
toto/check_wins.py — Compare latest toto_predictions_log entries against actual draw.
Reads from toto_predictions_log (canonical table). Writes result to toto_results.

Schema used:
  toto_predictions_log: id, draw_no, predicted_at, model_version, predicted_numbers,
                        system_type, retrain_draw_no, confidence_scores, notes
  toto_results:         id (PK), prediction_id, draw_number, draw_date, checked_at,
                        actual_n1..n6, actual_additional,
                        sys6_t1_result..bonus_t6_result (JSON),
                        best_group, total_prize_mandatory, total_prize_full,
                        total_cost_mandatory, any_win, notes
"""
import json, sys, os, duckdb
from datetime import datetime, timezone

sys.path.insert(0, "/home/ubuntu/statlot-649/statlot")
from toto.toto_prizes import check_ordinary, check_system_entry

DB_PATH     = "/home/ubuntu/statlot-649/statlot_toto.duckdb"
DRAW_FILE   = "/home/ubuntu/statlot-649/statlot/toto/latest_draw.json"
STATUS_FILE = "/home/ubuntu/statlot-649/logs/toto_pipeline_status.json"

TICKET_ORDER = [
    "sys6_t1", "sys6_t2", "sys6_t3",
    "sys7_t1",
    "sys8_t1",
    "sys9_t1",
    "sys10_t1",
    "sys11_t1",
    "sys12_t1",
    "bonus_t6",
]

MANDATORY_TICKETS = {"sys6_t1", "sys6_t2", "sys6_t3", "sys7_t1"}

GROUP_LABELS = {
    1: "Group 1 (JACKPOT!🎰)",
    2: "Group 2 (5+add 🔥)",
    3: "Group 3 (5 nums ✅)",
    4: "Group 4 (4+add)",
    5: "Group 5 ($50)",
    6: "Group 6 ($25)",
    7: "Group 7 ($10)",
}


def write_status(step, status):
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump({
            "step": step,
            "status": status,
            "ts": datetime.now(timezone.utc).isoformat(),
        }, f)


def run():
    write_status("check_wins", "running")

    # ── Load actual draw ──────────────────────────────────────────────────────
    with open(DRAW_FILE) as f:
        draw = json.load(f)
    winning     = draw["numbers"]       # list of 6 ints
    additional  = draw["additional"]    # int
    draw_number = draw["draw_number"]   # int
    draw_date   = draw["draw_date"]     # "YYYY-MM-DD"

    print(f"\nChecking draw #{draw_number} ({draw_date})")
    print(f"Winning: {winning} + Additional: {additional}")

    con = duckdb.connect(DB_PATH)

    # ── Load predictions from toto_predictions_log ───────────────────────────
    # Get the most recent batch: latest predicted_at for draw_no = draw_number
    # (these are predictions FOR this draw — draw_no is the target draw)
    latest_ts = con.execute("""
        SELECT MAX(predicted_at)
        FROM toto_predictions_log
        WHERE draw_no = ?
    """, [draw_number]).fetchone()[0]

    if latest_ts is None:
        # Fallback: no prediction for this exact draw_no — use globally latest batch
        print(f"[WARN] No predictions found for draw #{draw_number} in toto_predictions_log.")
        print("       Falling back to most recent prediction batch overall.")
        latest_ts = con.execute("""
            SELECT MAX(predicted_at) FROM toto_predictions_log
        """).fetchone()[0]
        if latest_ts is None:
            print("ERROR: toto_predictions_log is empty. Cannot check wins.")
            write_status("check_wins", "no_prediction")
            con.close()
            return None

    rows = con.execute("""
        SELECT system_type, predicted_numbers, model_version, draw_no
        FROM toto_predictions_log
        WHERE predicted_at = ?
        ORDER BY id
    """, [latest_ts]).fetchall()

    if not rows:
        print("ERROR: No rows found at latest_ts. Cannot check wins.")
        write_status("check_wins", "no_prediction")
        con.close()
        return None

    model_version  = rows[0][2]
    pred_draw_no   = rows[0][3]  # the draw_no these predictions were made FOR
    prediction_id  = f"log_{pred_draw_no}_model_{model_version}"

    print(f"Using prediction batch: draw_no={pred_draw_no}, model={model_version}, predicted_at={latest_ts}")
    print(f"Prediction ID: {prediction_id}")

    # Build tickets dict: system_type → list of ints
    tickets = {}
    for system_type, predicted_numbers, _mv, _dn in rows:
        tickets[system_type] = predicted_numbers  # already a Python list from DuckDB INTEGER[]

    # ── Prize check ───────────────────────────────────────────────────────────
    results     = {}
    best_group  = None
    total_mandatory = 0.0
    total_full      = 0.0
    any_win         = False

    print("\n--- Prize Check ---")
    for name in TICKET_ORDER:
        nums = tickets.get(name)
        if nums is None:
            results[name] = None
            print(f"  {name}: [NOT IN PREDICTION BATCH]")
            continue

        if len(nums) == 6:
            r = check_ordinary(nums, winning, additional)
        else:
            r = check_system_entry(nums, winning, additional)

        results[name] = r
        g           = r["best_group"]
        prize_fixed = r["total_fixed_prize_sgd"]
        label       = GROUP_LABELS.get(g, "No win") if g else "No win"
        print(f"  {name}: {nums} → {label} | Fixed prize: ${prize_fixed:.0f}")

        if g:
            any_win = True
            if best_group is None or g < best_group:
                best_group = g
            if name in MANDATORY_TICKETS:
                total_mandatory += prize_fixed
            total_full += prize_fixed

    print(f"\nBest group: {GROUP_LABELS.get(best_group, 'No win')}")
    print(f"Total prize (mandatory only, sys6×3 + sys7): ${total_mandatory:.0f}")
    print(f"Total prize (all systems): ${total_full:.0f}")
    print(f"Any win: {any_win}")

    # ── Upsert actual draw into toto_draws ────────────────────────────────────
    con.execute("""
        INSERT OR REPLACE INTO toto_draws
            (draw_number, draw_date, n1, n2, n3, n4, n5, n6, additional)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [draw_number, draw_date] + winning + [additional])

    # ── Write result row to toto_results ─────────────────────────────────────
    result_id = f"{prediction_id}_result_{draw_number}"

    def jr(r):
        return json.dumps(r) if r else None

    con.execute("""
        INSERT OR REPLACE INTO toto_results
            (id, prediction_id, draw_number, draw_date,
             actual_n1, actual_n2, actual_n3, actual_n4, actual_n5, actual_n6,
             actual_additional,
             sys6_t1_result, sys6_t2_result, sys6_t3_result, sys7_t1_result,
             sys8_t1_result, sys9_t1_result, sys10_t1_result,
             sys11_t1_result, sys12_t1_result, bonus_t6_result,
             best_group, total_prize_mandatory, total_prize_full,
             total_cost_mandatory, any_win, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        result_id, prediction_id, draw_number, draw_date,
    ] + winning + [additional] + [
        jr(results.get(k)) for k in (
            "sys6_t1", "sys6_t2", "sys6_t3", "sys7_t1",
            "sys8_t1", "sys9_t1", "sys10_t1",
            "sys11_t1", "sys12_t1", "bonus_t6",
        )
    ] + [
        best_group, total_mandatory, total_full, 10.0, any_win,
        f"model={model_version} trained_on={pred_draw_no-1} draw_no={draw_number}",
    ])

    con.close()
    write_status("check_wins", "done")

    return {
        "draw":             draw,
        "prediction_id":    prediction_id,
        "model_version":    model_version,
        "best_group":       best_group,
        "total_mandatory":  total_mandatory,
        "total_full":       total_full,
        "any_win":          any_win,
        "results":          results,
    }


if __name__ == "__main__":
    run()
