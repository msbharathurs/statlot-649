# AGENT_STATE.md
# Statlot Project — Live Ground Truth
# Updated: 2026-04-05 (session end — Tasks 1–4 complete)
# THIS FILE IS THE FIRST THING YOU READ EVERY SESSION. KEEP IT CURRENT.

## WHAT IS THIS PROJECT
Statlot is a lottery prediction system for Singapore TOTO and 4D.
It uses statistical models (M1–M9: Bayesian, Poisson, RF, Monte Carlo, XGBoost,
DQN, Markov, FFT/GMM, LSTM) trained on historical draw data stored in DuckDB.
The system runs on EC2 (t3.nano), scrapes results from Singapore Pools, retrains
models after each draw, and stores predictions in DuckDB.

## CANONICAL PATHS (source of truth — update this if anything moves)

| Resource              | Path                                                        | Status |
|-----------------------|-------------------------------------------------------------|--------|
| TOTO DuckDB           | /home/ubuntu/statlot-649/statlot_toto.duckdb                | ✅ Correct path |
| 4D DuckDB             | /home/ubuntu/statlot-649/draws_4d.duckdb                    | ✅ 4,473 draws, 64 cols |
| TOTO scripts          | ~/statlot-649/statlot/toto/                                 | ✅ In git |
| Engine models         | ~/statlot-649/statlot/engine/models/                        | ✅ M1–M9 in git |
| Historical draws JSON | ~/statlot-649/statlot/sp_historical_draws.json              | ⚠️ SEE BELOW |
| Backtest scripts      | ~/statlot-649/statlot/                                      | ✅ In git |
| SSH utility           | ~/statlot-649/statlot/toto/ssh_utils.py                     | ✅ In git (commit 519960b) |
| Agent state           | ~/statlot-649/AGENT_STATE.md                                | ✅ This file |
| Agent rules           | ~/statlot-649/AGENT_RULES.md                                | ✅ In git (commit c55517c) |
| Python venv           | /home/ubuntu/statlot-649/statlot/venv/bin/python3           | ✅ Confirmed |
| scrape_latest.py      | ~/statlot-649/statlot/toto/scrape_latest.py                 | ✅ FIXED 2026-04-05 (commit 7a5c887) |
| latest_draw.json      | ~/statlot-649/statlot/toto/latest_draw.json                 | ✅ draw #4170 |

## DO NOT USE THESE PATHS — they are stale duplicates:
- `~/statlot-649/statlot/statlot_toto.duckdb` — 12KB EMPTY
- `~/statlot-649/statlot.duckdb` — 274KB old draft schema
- Any path under `.agents/scripts/` — sandbox only, wiped every session

---

## CURRENT DB STATE (verified 2026-04-05 13:40 UTC)

### TOTO DuckDB (`/home/ubuntu/statlot-649/statlot_toto.duckdb`)

| Table                | Rows | Notes |
|----------------------|------|-------|
| toto_draws           | 2,973 | Draws #4–#4170. Training window: #2341–#4170 = 1830 draws. |
| toto_predictions     | 4    | Legacy table — old schema |
| toto_results         | 1    | ✅ Win-check ran for draw #4170. Any win: True (Sys11=Group7, Sys12=Group5). |
| toto_predictions_log | 30   | ✅ 10 dry-run + 10 model=1680776 + 10 model=7a5c887 (all for draw #4171) |
| v_weekly_summary     | 4    | View |

**Win-check result for draw #4170 (verbatim stdout):**
```
Winning: [1, 7, 8, 23, 30, 33] + Additional: 21
  sys6_t1:  [4,15,17,35,43,49]                      → No win | $0
  sys6_t2:  [8,13,22,34,37,49]                       → No win | $0
  sys6_t3:  [4,22,31,34,35,49]                       → No win | $0
  sys7_t1:  [4,15,17,22,35,43,49]                    → No win | $0
  sys8_t1:  [4,15,17,22,34,35,37,43,49]              → No win | $0
  sys9_t1:  [4,12,13,15,17,22,28,34,35,37,43,49]     → No win | $0
  sys10_t1: [4,8,10,12,13,15,17,22,28,30,34,35,37,43,46,49] → No win | $0
  sys11_t1: [4,8,10,12,13,15,17,22,24,28,30,31,32,33,34,35,37,40,43,46,49] → Group 7 ($10) | Fixed: $8160
  sys12_t1: [1,4,6,8,10,12,13,14,15,17,18,22,24,28,30,31,32,33,34,35,37,38,40,43,46,48,49] → Group 5 ($50) | Fixed: $83490
  bonus_t6: [4,21,30,35,37,49]                       → No win | $0
Best group: Group 5 ($50) | Total prize (all sys): $91,650
```
NOTE: win-check reads from old `toto_predictions` table. The `toto_results` row uses
pred_id from the legacy table, not from `toto_predictions_log`. This is a known mismatch
to fix in a future session.

**Retrain + predict for draw #4171 (model=7a5c887, 2026-04-05):**
```
Training draws: 1830 (#2341–#4170) | Git: 7a5c887 | Dry run: False
Sys6 T1: [4, 15, 17, 35, 43, 49]
Sys6 T2: [8, 13, 22, 34, 37, 49]
Sys6 T3: [4, 22, 31, 34, 35, 49]
Sys7 T1: [4, 15, 17, 22, 35, 43, 49]
Sys8 T1: [4, 15, 17, 22, 34, 35, 37, 43, 49]
Sys9 T1: [4, 12, 13, 15, 17, 22, 28, 34, 35, 37, 43, 49]
Sys10 T1: [4, 8, 10, 12, 13, 15, 17, 22, 28, 30, 34, 35, 37, 43, 46, 49]
Sys11 T1: [4, 8, 10, 12, 13, 15, 17, 22, 24, 28, 30, 31, 32, 33, 34, 35, 37, 40, 43, 46, 49]
Sys12 T1: [1, 4, 6, 8, 10, 12, 13, 14, 15, 17, 18, 22, 24, 28, 30, 31, 32, 33, 34, 35, 37, 38, 40, 43, 46, 48, 49]
Bonus T6: [4, 21, 30, 35, 37, 49]
Additional picks: [21, 48, 31, 17, 6]
Cost mandatory (3×Sys6 + 1×Sys7): $10
```

**NOTE on toto_draws range:** MIN draw = #4 (corrupted early dates still present in DB).
Training window is **draws >= #2341 only** — enforced in retrain_and_predict.py.
1830 draws used for training (#2341–#4170).

### 4D DuckDB (`/home/ubuntu/statlot-649/draws_4d.duckdb`)

| Table        | Rows    | Notes |
|--------------|---------|-------|
| draws        | 4,473   | Full 4D history ✅ |
| draw_numbers | 102,823 | 64 enriched feature columns ✅ |

---

## BROKEN THINGS (do not paper over these)

### P0 — Must fix before production automation is reliable

1. **win-check reads from legacy `toto_predictions` table, not `toto_predictions_log`**
   - check_wins.py looks up predictions from `toto_predictions` (old schema)
   - `toto_predictions_log` (new schema, 30 rows) is NOT checked by win-check
   - The `toto_results` row written for draw #4170 used the old pred_id
   - Fix: update check_wins.py to read from `toto_predictions_log` instead
   - **Status: UNFIXED — next session priority #1**

2. **Corrupted early draw dates** — draws #4 through ~#2340 have wrong/garbage dates
   - Mitigation: training already excludes draws < #2341
   - **Status: LOW PRIORITY**

### P1 — Known issues, lower urgency
3. **Backtest numbers are invalid** — all cited lift/accuracy numbers (13.6% Sys7 etc.)
   were computed on pre-2341 data (corrupted/incomplete). Not meaningful.
4. **4D pipeline SSH key problem** — 4D automation not verified end-to-end.

---

## COMPLETED THIS SESSION (2026-04-05)

| # | Task | Evidence |
|---|------|----------|
| 1 | AGENT_RULES.md committed to repo root | commit c55517c |
| 2 | scrape_latest.py FIXED — static HTML URL | commit 7a5c887. Fetched draw #4170 correctly. Exit code 0. |
| 3 | Win-check ran for draw #4170 | check_wins.py exit code 0. toto_results: 1 row. Any win: True (Sys11 Group7, Sys12 Group5). |
| 4 | Full retrain + predict for draw #4171 | retrain_and_predict.py exit code 0. model=7a5c887. toto_predictions_log: 30 rows (+10 new). EC2 scaled t3.nano→t3.medium→t3.nano. |
| 5 | GitHub PAT refreshed | New token set in Base44 secrets. All 5 pending commits pushed (e7e1662..7a5c887). |
| 6 | joblib/scikit-learn/xgboost/torch installed in venv | pip install exit code 0 |

## GIT COMMIT HISTORY (this session)

| Hash    | Description |
|---------|-------------|
| 7a5c887 | fix: scrape_latest.py uses static HTML archive URL; update AGENT_STATE.md |
| c55517c | feat: add AGENT_RULES.md to repo root — session operating rules |

---

## NEXT STEPS (in priority order)

1. **Fix check_wins.py** — update to read from `toto_predictions_log` instead of legacy `toto_predictions`
   - Verify toto_results gets a correct row after draw #4171
   - Test end-to-end: run check_wins.py after draw #4171 result is published (Mon 07 Apr 2026)

2. **Verify automation pipeline end-to-end** — manually trigger the Monday pipeline on a real draw day,
   paste full output, verify DB row counts change correctly

3. **Fix corrupted early draw dates** (low priority — not blocking predictions)
