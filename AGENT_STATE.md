# AGENT_STATE.md
# Statlot Project — Live Ground Truth
# Updated: 2026-04-04
# THIS FILE IS THE FIRST THING YOU READ EVERY SESSION. KEEP IT CURRENT.

## WHAT IS THIS PROJECT
Statlot is a lottery prediction system for Singapore TOTO and 4D.
It uses statistical models (M1-M9: Bayesian, Poisson, RF, Monte Carlo, XGBoost,
DQN, Markov, FFT/GMM, LSTM) trained on historical draw data stored in DuckDB.
The system runs on EC2 (t3.nano), scrapes results from Singapore Pools, retrains
models after each draw, and stores predictions in DuckDB.

## CANONICAL PATHS (source of truth — update this if anything moves)

| Resource              | Path                                                        | Status |
|-----------------------|-------------------------------------------------------------|--------|
| TOTO DuckDB           | /home/ubuntu/statlot-649/statlot_toto.duckdb                | ✅ Correct path |
| 4D DuckDB             | /home/ubuntu/statlot-649/draws_4d.duckdb                    | ✅ 4,473 draws, 64 cols |
| TOTO scripts          | ~/statlot-649/statlot/toto/                                 | ✅ In git |
| Engine models         | ~/statlot-649/statlot/engine/models/                        | ✅ M1-M9 in git |
| Historical draws JSON | ~/statlot-649/statlot/sp_historical_draws.json              | ✅ 4,126 draws (was 2,298) |
| Backtest scripts      | ~/statlot-649/statlot/                                      | ✅ In git |
| Agent state           | ~/statlot-649/AGENT_STATE.md                                | ✅ This file |
| Agent rules           | ~/statlot-649/AGENT_RULES.md                                | ✅ In git |
| Python venv           | /home/ubuntu/statlot-649/statlot/venv/bin/python3           | ✅ Confirmed |

## DO NOT USE THESE PATHS — they are stale duplicates:
- `~/statlot-649/statlot/statlot_toto.duckdb` — 12KB EMPTY
- `~/statlot-649/statlot.duckdb` — 274KB old draft schema
- Any path under `.agents/scripts/` — sandbox only, wiped every session

---

## CURRENT DB STATE (verified 2026-04-03 session)

### TOTO DuckDB (`/home/ubuntu/statlot-649/statlot_toto.duckdb`)

| Table                | Rows  | Notes |
|----------------------|-------|-------|
| toto_draws           | 2,973 | Clean data only. 1,143 pre-2341 + 1,830 post-2341 (incl #4169, #4170) |
| toto_predictions     | 2     | Draws #4168 and #4170 only |
| toto_results         | 0     | Win-check has NEVER run |
| toto_predictions_log | 0     | Table created this session. Needs retrain to populate. |

### 4D DuckDB (`/home/ubuntu/statlot-649/draws_4d.duckdb`)

| Table        | Rows    | Notes |
|--------------|---------|-------|
| draws        | 4,473   | Full 4D history ✅ |
| draw_numbers | 102,823 | 64 enriched feature columns ✅ |

---

## DATA QUALITY RULES (permanent — do not change without Bharath approval)

- **Draws #1–#2340:** NULL date or NULL numbers → silently excluded from toto_draws. No retry.
- **Draws #2341+:** NULL date or NULL numbers → must retry. If retry fails, log draw_no and
  report to Bharath for manual entry. Never silently drop post-2341 data.
- scrape_history.py enforces these rules at both scrape time and rebuild time.

## DRAW HISTORY STATUS

sp_historical_draws.json: 4,126 draws (was 2,298)
toto_draws table: 2,973 clean rows
- Pre-2341 rows with valid data: 1,143
- Pre-2341 rows excluded (corrupted/null): ~1,155 silently dropped per quality rule
- Post-2341 rows: 1,830 (draws #2341-#4170, all clean, 0 nulls)
- Post-2341 draws with missing data: 0 — none need manual entry
- JSON backup at: sp_historical_draws.json.bak

---

## BROKEN THINGS (do not paper over these)

### P0 — Must fix before any retrain is meaningful
~~1. 1,828 missing TOTO draws~~ ✅ FIXED — all #2341-#4168 scraped, 0 failures
~~2. Corrupted dates in sp_historical_draws.json~~ ✅ FIXED — pre-2341 bad rows excluded by rule
~~3. toto_predictions_log table does not exist~~ ✅ FIXED — table created, 0 rows (populated on next retrain)

### P1 — Fix next
4. **Win-check has never run** — toto_results has 0 rows. Don't know if any prediction ever hit.
5. **scrape_latest.py is broken** — hits JS-rendered URL (regex never matches), saves to wrong path.
6. **SSH key at /tmp/statlot.pem is ephemeral** — wiped between automation runs. Pipeline always fails at SSH step.
7. **retrain_and_predict.py does not write to toto_predictions_log** — Task 3 not yet done.

### P2 — Known but lower priority
8. **Backtest numbers are invalid** — all cited numbers (13.6% Sys7 lift etc.) on incomplete data. Re-run after retrain on full 2,973 draws.
9. **4D pipeline has same SSH key problem** — not verified working end-to-end.

---

## TASK LIST (current)

### IN PROGRESS
- Nothing (session still open)

### NEXT UP (in priority order)
1. **Task 3: Update retrain_and_predict.py** to write to toto_predictions_log after every prediction run
2. **Fix scrape_latest.py** — use static archive URL, save to correct path (toto/latest_draw.json)
3. **Fix SSH key handling** — write key from secret on each automation run, not relying on /tmp/
4. **Run win-check on all stored predictions** — populate toto_results table
5. **Re-run backtests on complete data** — now that 2,973 draws are available

### COMPLETED THIS SESSION
- toto_predictions_log table created in DuckDB ✅ (git: a646e38)
- toto_db.py updated — init_schema() now creates toto_predictions_log ✅ (git: a646e38)
- scrape_history.py built and tested ✅ (git: b949018)
  - 1,828 draws #2341-#4168 scraped — 0 failures
  - Data quality rules permanently encoded: pre-2341 NULLs silently excluded, post-2341 NULLs reported
  - JSON backup confirmed at sp_historical_draws.json.bak
  - toto_draws rebuilt: 2,973 clean rows

### COMPLETED PREVIOUS SESSIONS
- All models M1-M9 committed to git ✅
- 4D DuckDB built with 64 enriched feature columns ✅
- TOTO pipeline scripts committed to correct path (statlot/toto/) ✅
- toto_db.py confirmed pointing to correct canonical DB path ✅
- AGENT_STATE.md and AGENT_RULES.md committed to repo root ✅ (git: 09b58b1)

---

## MODEL INVENTORY (what exists in git)

| Model            | File                              | Trained? | Data it trained on       |
|------------------|-----------------------------------|----------|--------------------------|
| M1 Bayesian      | engine/models/m1_bayes.py         | Yes      | 649 draws #1-2340 only ⚠️ |
| M2 Poisson       | engine/models/m2_ev_kelly.py      | Yes      | 649 draws #1-2340 only ⚠️ |
| M3 Random Forest | engine/models/m3_rf.py            | Yes      | 649 draws #1-2340 only ⚠️ |
| M4 Monte Carlo   | engine/models/m4_monte_carlo.py   | Yes      | 649 draws #1-2340 only ⚠️ |
| M5 XGBoost       | engine/models/m5_xgb.py           | Yes      | 649 draws #1-2340 only ⚠️ |
| M6 DQN           | engine/models/m6_dqn.py           | Yes      | 649 draws #1-2340 only ⚠️ |
| M7 Markov        | engine/models/m7_markov.py        | Yes      | 649 draws #1-2340 only ⚠️ |
| M8 FFT/GMM       | engine/models/m8_gmm.py           | Yes      | 649 draws #1-2340 only ⚠️ |
| M9 LSTM          | engine/models/m9_lstm.py          | Yes      | 649 draws #1-2340 only ⚠️ |

⚠️ ALL models need retraining on full 2,973-draw dataset. No retrain has run on complete data.
⚠️ TOTO-specific retraining has never successfully completed end-to-end.

---

## LAST SESSION SUMMARY
**Date:** 2026-04-03
**What happened:**
- AGENT_STATE.md and AGENT_RULES.md committed to git from Bharath's PDFs
- SESSION START RITUAL completed — DB verified, git verified
- Task 1: toto_predictions_log table created in DuckDB, toto_db.py updated, pushed to GitHub (a646e38)
- Task 2: scrape_history.py built — scraped all 1,828 missing draws (#2341-#4168), 0 failures
  - Data quality rule applied: pre-2341 NULLs silently excluded, post-2341 NULLs reported
  - toto_draws rebuilt with 2,973 clean rows
  - Pushed to GitHub (b949018)
- Task 3: NOT YET DONE — retrain_and_predict.py not yet updated to write to toto_predictions_log
**Git state:** Updated — b949018 is latest
**Next session must start with:** Task 3 — update retrain_and_predict.py to write to toto_predictions_log
