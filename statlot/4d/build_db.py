"""
build_db.py — 4D DuckDB builder (memory-efficient streaming version)
Reads draws_4d_working.json (JSONL), computes all feature columns,
writes draws_4d.duckdb with two tables:
  - draws        : 1 row per draw (raw source)
  - draw_numbers : 1 row per number per draw (~102k rows) with all pattern columns

Memory strategy: process one draw at a time, stream insert — never holds all rows in RAM.
"""

import json
import duckdb
import os
import gc

SOURCE   = os.path.expanduser("~/statlot-649/draws_4d_working.json")
DB_PATH  = os.path.expanduser("~/statlot-649/draws_4d.duckdb")

PRIMES = {2, 3, 5, 7}
TIERS  = ["prize_1st", "prize_2nd", "prize_3rd"] + \
         [f"starter_{i}" for i in range(1, 11)] + \
         [f"consolation_{i}" for i in range(1, 11)]
TIER_LABELS = (
    ["1st", "2nd", "3rd"] +
    [f"starter_{i}" for i in range(1, 11)] +
    [f"consolation_{i}" for i in range(1, 11)]
)

COLS = [
    "row_id","draw_id","draw_date","day_of_week","tier","tier_rank","number",
    "d1","d2","d3","d4","digit_sum","digit_product","sum_band",
    "odd_count","even_count","all_odd","all_even","parity_pattern",
    "low_count","high_count","all_low","all_high","hl_pattern",
    "unique_digit_count","number_class","pair_type",
    "has_pair","has_triple","has_quad","repeated_digit","repeat_positions",
    "has_adjacent_pair","has_adjacent_triple","adj_pair_digit","adj_pair_pos",
    "has_2_same_nonadj","is_palindrome",
    "max_consecutive_run","has_3_consecutive","has_4_consecutive",
    "is_arithmetic_prog","ap_step",
    "first2_val","last2_val","first2_sum","last2_sum","is_mirror_sum",
    "is_double_double","double_double_type",
    "prime_digit_count","all_prime_digits",
    "max_gap","min_gap","mean_gap","ibox_key",
    "times_seen_all_time","times_seen_last_100","times_seen_last_500",
    "draws_since_last_seen","last_seen_draw_id",
    "ibox_times_seen_all_time","ibox_last_seen_draw_id","ibox_draws_since_last"
]


def analyze(num, draw_id, draw_date, dow, tier, tier_rank,
            seen_history, ibox_history, row_id):
    """Compute all feature columns for one 4-digit number. Returns a tuple."""
    if not num or len(num) != 4:
        return None

    d = [int(c) for c in num]
    d1, d2, d3, d4 = d
    from collections import Counter

    digit_sum     = sum(d)
    digit_product = d[0] * d[1] * d[2] * d[3]
    sum_band      = "small" if digit_sum <= 13 else ("medium" if digit_sum <= 22 else "large")

    odd_count  = sum(x % 2 != 0 for x in d)
    even_count = 4 - odd_count
    all_odd    = odd_count == 4
    all_even   = even_count == 4
    parity_pat = "".join("O" if x % 2 != 0 else "E" for x in d)

    low_count  = sum(x <= 4 for x in d)
    high_count = 4 - low_count
    all_low    = low_count == 4
    all_high   = high_count == 4
    hl_pat     = "".join("L" if x <= 4 else "H" for x in d)

    freq      = Counter(d)
    unique_cnt = len(freq)
    counts    = sorted(freq.values(), reverse=True)
    has_pair  = counts[0] >= 2
    has_triple = counts[0] >= 3
    has_quad  = counts[0] == 4

    if unique_cnt == 1:
        pair_type = "AAAA"
    elif unique_cnt == 2:
        if counts == [3, 1]:
            pair_type = "AAAB"
        else:
            if d[0] == d[1] and d[2] == d[3]:
                pair_type = "AABB"
            elif d[0] == d[2] and d[1] == d[3]:
                pair_type = "ABAB"
            elif d[0] == d[3] and d[1] == d[2]:
                pair_type = "ABBA"
            else:
                pair_type = "AABB_other"
    elif unique_cnt == 3:
        pair_digit = [x for x, c in freq.items() if c == 2][0]
        positions  = [i for i, x in enumerate(d) if x == pair_digit]
        pos_map = {
            (0,1): "AABC", (2,3): "ABCC", (0,2): "ABAC",
            (1,3): "ABBC", (0,3): "ABCA", (1,2): "ABCB"
        }
        pair_type = pos_map.get(tuple(positions), "PAIR_OTHER")
    else:
        pair_type = "ABCD"

    if has_pair or has_triple or has_quad:
        repeated_digit   = max(freq, key=freq.get)
        repeat_positions = ",".join(str(i) for i, x in enumerate(d) if x == repeated_digit)
    else:
        repeated_digit   = None
        repeat_positions = None

    adj_pairs = [(i, d[i]) for i in range(3) if d[i] == d[i+1]]
    has_adjacent_pair   = len(adj_pairs) >= 1
    has_adjacent_triple = len(adj_pairs) >= 2
    adj_pair_digit      = adj_pairs[0][1] if adj_pairs else None
    adj_pair_pos        = adj_pairs[0][0] if adj_pairs else None

    has_2_same_nonadj = bool(
        has_pair and not has_adjacent_pair and not has_triple and not has_quad
    )

    is_palindrome = (num == num[::-1])

    def max_run(digits):
        max_r = cur = 1
        for i in range(1, len(digits)):
            cur = cur + 1 if digits[i] == digits[i-1] + 1 else 1
            max_r = max(max_r, cur)
        return max_r

    asc_run  = max_run(d)
    desc_run = max_run(list(reversed(d)))
    max_consecutive_run = max(asc_run, desc_run)
    has_3_consecutive   = max_consecutive_run >= 3
    has_4_consecutive   = max_consecutive_run == 4

    steps = [d[i+1] - d[i] for i in range(3)]
    is_ap   = (steps[0] == steps[1] == steps[2])
    ap_step = steps[0] if is_ap else None

    first2_val  = d1 * 10 + d2
    last2_val   = d3 * 10 + d4
    first2_sum  = d1 + d2
    last2_sum   = d3 + d4
    is_mirror_sum = (first2_sum == last2_sum)

    is_double_double   = pair_type in ("AABB", "ABAB", "ABBA", "AAAA")
    double_double_type = pair_type if is_double_double else None

    prime_digit_count = sum(x in PRIMES for x in d)
    all_prime_digits  = prime_digit_count == 4

    gaps     = [abs(d[i+1] - d[i]) for i in range(3)]
    max_gap  = max(gaps)
    min_gap  = min(gaps)
    mean_gap = round(sum(gaps) / 3, 4)

    ibox_key = "".join(sorted(num))

    if has_quad:
        number_class = "quad"
    elif has_triple:
        number_class = "triple"
    elif pair_type in ("AABB", "ABAB", "ABBA"):
        number_class = "double_pair"
    elif has_pair:
        number_class = "single_pair"
    else:
        number_class = "unique"

    # lookback
    past      = seen_history.get(num, [])
    past_ibox = ibox_history.get(ibox_key, [])
    times_seen_all      = len(past)
    times_seen_100      = sum(1 for x in past if x >= draw_id - 100)
    times_seen_500      = sum(1 for x in past if x >= draw_id - 500)
    draws_since         = (draw_id - past[-1]) if past else None
    last_seen_id        = past[-1] if past else None
    ibox_all            = len(past_ibox)
    ibox_last           = past_ibox[-1] if past_ibox else None
    ibox_since          = (draw_id - past_ibox[-1]) if past_ibox else None

    return (
        row_id, draw_id, draw_date, dow, tier, tier_rank, num,
        d1, d2, d3, d4, digit_sum, digit_product, sum_band,
        odd_count, even_count, all_odd, all_even, parity_pat,
        low_count, high_count, all_low, all_high, hl_pat,
        unique_cnt, number_class, pair_type,
        has_pair, has_triple, has_quad, repeated_digit, repeat_positions,
        has_adjacent_pair, has_adjacent_triple, adj_pair_digit, adj_pair_pos,
        has_2_same_nonadj, is_palindrome,
        max_consecutive_run, has_3_consecutive, has_4_consecutive,
        is_ap, ap_step,
        first2_val, last2_val, first2_sum, last2_sum, is_mirror_sum,
        is_double_double, double_double_type,
        prime_digit_count, all_prime_digits,
        max_gap, min_gap, mean_gap, ibox_key,
        times_seen_all, times_seen_100, times_seen_500,
        draws_since, last_seen_id,
        ibox_all, ibox_last, ibox_since
    )


# ── load & sort draws ─────────────────────────────────────────────────────────
print("Loading draws_4d_working.json ...")
with open(SOURCE) as f:
    draws = [json.loads(line) for line in f if line.strip()]
draws.sort(key=lambda x: x["draw_number"])
print(f"  {len(draws)} draws loaded")

# ── open DuckDB ───────────────────────────────────────────────────────────────
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
con = duckdb.connect(DB_PATH)

# ── Table 1: draws ────────────────────────────────────────────────────────────
print("Creating table: draws ...")
con.execute("""
CREATE TABLE draws (
    draw_id INTEGER PRIMARY KEY, draw_date DATE, day_of_week VARCHAR,
    prize_1st VARCHAR, prize_2nd VARCHAR, prize_3rd VARCHAR,
    starter_1 VARCHAR, starter_2 VARCHAR, starter_3 VARCHAR, starter_4 VARCHAR,
    starter_5 VARCHAR, starter_6 VARCHAR, starter_7 VARCHAR, starter_8 VARCHAR,
    starter_9 VARCHAR, starter_10 VARCHAR,
    consolation_1 VARCHAR, consolation_2 VARCHAR, consolation_3 VARCHAR,
    consolation_4 VARCHAR, consolation_5 VARCHAR, consolation_6 VARCHAR,
    consolation_7 VARCHAR, consolation_8 VARCHAR, consolation_9 VARCHAR,
    consolation_10 VARCHAR
)
""")
draw_rows = [
    (
        d["draw_number"], d["draw_date"], d.get("day_of_week",""),
        d.get("prize_1st"), d.get("prize_2nd"), d.get("prize_3rd"),
        d.get("starter_1"), d.get("starter_2"), d.get("starter_3"), d.get("starter_4"),
        d.get("starter_5"), d.get("starter_6"), d.get("starter_7"), d.get("starter_8"),
        d.get("starter_9"), d.get("starter_10"),
        d.get("consolation_1"), d.get("consolation_2"), d.get("consolation_3"),
        d.get("consolation_4"), d.get("consolation_5"), d.get("consolation_6"),
        d.get("consolation_7"), d.get("consolation_8"), d.get("consolation_9"),
        d.get("consolation_10"),
    )
    for d in draws
]
con.executemany("INSERT INTO draws VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", draw_rows)
del draw_rows; gc.collect()
print(f"  {len(draws)} draws inserted")

# ── Table 2: draw_numbers ─────────────────────────────────────────────────────
print("Creating table: draw_numbers ...")
con.execute("""
CREATE TABLE draw_numbers (
    row_id INTEGER, draw_id INTEGER, draw_date DATE, day_of_week VARCHAR,
    tier VARCHAR, tier_rank INTEGER, number VARCHAR(4),
    d1 TINYINT, d2 TINYINT, d3 TINYINT, d4 TINYINT,
    digit_sum TINYINT, digit_product INTEGER, sum_band VARCHAR,
    odd_count TINYINT, even_count TINYINT, all_odd BOOLEAN, all_even BOOLEAN,
    parity_pattern VARCHAR(4),
    low_count TINYINT, high_count TINYINT, all_low BOOLEAN, all_high BOOLEAN,
    hl_pattern VARCHAR(4),
    unique_digit_count TINYINT, number_class VARCHAR, pair_type VARCHAR,
    has_pair BOOLEAN, has_triple BOOLEAN, has_quad BOOLEAN,
    repeated_digit TINYINT, repeat_positions VARCHAR,
    has_adjacent_pair BOOLEAN, has_adjacent_triple BOOLEAN,
    adj_pair_digit TINYINT, adj_pair_pos TINYINT,
    has_2_same_nonadj BOOLEAN, is_palindrome BOOLEAN,
    max_consecutive_run TINYINT, has_3_consecutive BOOLEAN, has_4_consecutive BOOLEAN,
    is_arithmetic_prog BOOLEAN, ap_step TINYINT,
    first2_val TINYINT, last2_val TINYINT, first2_sum TINYINT, last2_sum TINYINT,
    is_mirror_sum BOOLEAN,
    is_double_double BOOLEAN, double_double_type VARCHAR,
    prime_digit_count TINYINT, all_prime_digits BOOLEAN,
    max_gap TINYINT, min_gap TINYINT, mean_gap FLOAT, ibox_key VARCHAR(4),
    times_seen_all_time INTEGER, times_seen_last_100 INTEGER, times_seen_last_500 INTEGER,
    draws_since_last_seen INTEGER, last_seen_draw_id INTEGER,
    ibox_times_seen_all_time INTEGER, ibox_last_seen_draw_id INTEGER,
    ibox_draws_since_last INTEGER
)
""")

SQL_INSERT = "INSERT INTO draw_numbers VALUES (" + ",".join(["?"] * len(COLS)) + ")"

# ── stream-process: one draw at a time ───────────────────────────────────────
seen_history = {}   # number  -> [draw_ids]
ibox_history = {}   # ibox_key-> [draw_ids]
row_id       = 0
total_rows   = 0
BATCH_SIZE   = 200  # draws per DuckDB commit (keeps memory tiny)
batch_rows   = []

print("Streaming feature computation + insert ...")
for i, draw in enumerate(draws):
    draw_id   = draw["draw_number"]
    draw_date = draw["draw_date"]
    dow       = draw.get("day_of_week", "")

    for tier_field, tier_label in zip(TIERS, TIER_LABELS):
        num = draw.get(tier_field)
        if not num or len(num) != 4:
            continue
        row_id += 1
        row = analyze(num, draw_id, draw_date, dow,
                      tier_label, TIER_LABELS.index(tier_label),
                      seen_history, ibox_history, row_id)
        if row:
            batch_rows.append(row)

    # update histories after processing (not before, so lookback is correct)
    for tier_field in TIERS:
        num = draw.get(tier_field)
        if num and len(num) == 4:
            ibox = "".join(sorted(num))
            seen_history.setdefault(num, []).append(draw_id)
            ibox_history.setdefault(ibox, []).append(draw_id)

    # flush batch to DuckDB
    if len(batch_rows) >= BATCH_SIZE * 23:
        con.executemany(SQL_INSERT, batch_rows)
        total_rows += len(batch_rows)
        batch_rows = []

    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{len(draws)}] draws processed, {total_rows} rows written so far")

# flush remainder
if batch_rows:
    con.executemany(SQL_INSERT, batch_rows)
    total_rows += len(batch_rows)

print(f"  Total rows inserted: {total_rows}")
del seen_history, ibox_history, batch_rows; gc.collect()

# ── indexes ───────────────────────────────────────────────────────────────────
print("Building indexes ...")
for idx, col in [
    ("idx_draw", "draw_id"), ("idx_number", "number"),
    ("idx_ibox", "ibox_key"), ("idx_tier", "tier"),
    ("idx_class", "number_class"), ("idx_date", "draw_date"),
]:
    con.execute(f"CREATE INDEX {idx} ON draw_numbers({col})")

# ── sanity checks ─────────────────────────────────────────────────────────────
print("\n=== Sanity Check ===")
r = con.execute("SELECT COUNT(*) FROM draws").fetchone()
print(f"draws table      : {r[0]} rows")
r = con.execute("SELECT COUNT(*) FROM draw_numbers").fetchone()
print(f"draw_numbers table: {r[0]} rows")

print("\n-- 1st prize number_class distribution --")
print(con.execute("""
    SELECT number_class,
           COUNT(*) as cnt,
           ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) as pct
    FROM draw_numbers WHERE tier='1st'
    GROUP BY number_class ORDER BY cnt DESC
""").fetchdf().to_string(index=False))

print("\n-- Top 10 most drawn 1st prize numbers --")
print(con.execute("""
    SELECT number, COUNT(*) as times_drawn
    FROM draw_numbers WHERE tier='1st'
    GROUP BY number ORDER BY times_drawn DESC LIMIT 10
""").fetchdf().to_string(index=False))

print("\n-- Pattern prevalence across all tiers --")
print(con.execute("""
    SELECT
        ROUND(100.0*SUM(CASE WHEN has_adjacent_pair   THEN 1 ELSE 0 END)/COUNT(*),2) as pct_adj_pair,
        ROUND(100.0*SUM(CASE WHEN has_2_same_nonadj   THEN 1 ELSE 0 END)/COUNT(*),2) as pct_nonadj_pair,
        ROUND(100.0*SUM(CASE WHEN is_palindrome        THEN 1 ELSE 0 END)/COUNT(*),2) as pct_palindrome,
        ROUND(100.0*SUM(CASE WHEN has_3_consecutive    THEN 1 ELSE 0 END)/COUNT(*),2) as pct_3consec,
        ROUND(100.0*SUM(CASE WHEN is_arithmetic_prog   THEN 1 ELSE 0 END)/COUNT(*),2) as pct_ap,
        ROUND(100.0*SUM(CASE WHEN all_even             THEN 1 ELSE 0 END)/COUNT(*),2) as pct_all_even,
        ROUND(100.0*SUM(CASE WHEN all_odd              THEN 1 ELSE 0 END)/COUNT(*),2) as pct_all_odd,
        ROUND(100.0*SUM(CASE WHEN is_double_double     THEN 1 ELSE 0 END)/COUNT(*),2) as pct_double_double
    FROM draw_numbers
""").fetchdf().to_string(index=False))

con.close()
db_size = os.path.getsize(DB_PATH) / (1024*1024)
print(f"\nDone. DuckDB → {DB_PATH}  ({db_size:.1f} MB)")
