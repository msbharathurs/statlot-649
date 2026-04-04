# AGENT_STATE.md
# Statlot Project — Live Ground Truth
# Updated: 2026-04-04 (session end)
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
| Agent rules           | ~/statlot-649/AGENT_RULES.md                                | ✅ In git |
| Python venv           | /home/ubuntu/statlot-649/statlot/venv/bin/python3           | ✅ Confirmed |

## DO NOT USE THESE PATHS — they are stale duplicates:
- `~/statlot-649/statlot/statlot_toto.duckdb` — 12KB EMPTY
- `~/statlot-649/statlot.duckdb` — 274KB old draft schema
- Any path under `.agents/scripts/` — sandbox only, wiped every session

---

## CURRENT DB STATE (verified 2026-04-04 18:00 SGT)

### TOTO DuckDB (`/home/ubuntu/statlot-649/statlot_toto.duckdb`)

| Table                | Rows | Notes |
|----------------------|------|-------|
| toto_draws           | 2,973 | Draws #4–#4170. Range: min=#4, max=#4170. Gap filled via scraper last session. |
| toto_predictions     | 3    | Draws #4168, #4170, #4171 |
| toto_results         | 0    | Win-check has NEVER run — still broken |
| toto_predictions_log | 20   | ✅ Table exists. 10 dry-run rows + 10 real rows for draw #4171 |
| v_weekly_summary     | 3    | View, not a real table |

**toto_predictions_log — actual rows as of 2026-04-04 (verbatim from DB):**
```
(1,  4171, 'sys6_t1',  [4,15,17,35,43,49],                            '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(2,  4171, 'sys6_t2',  [8,13,22,34,37,49],                            '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(3,  4171, 'sys6_t3',  [4,22,31,34,35,49],                            '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(4,  4171, 'sys7_t1',  [4,15,17,22,35,43,49],                         '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(5,  4171, 'sys8_t1',  [4,15,17,22,34,35,37,43,49],                   '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(6,  4171, 'sys9_t1',  [4,12,13,15,17,22,28,34,35,37,43,49],          '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(7,  4171, 'sys10_t1', [4,8,10,12,13,15,17,22,28,30,34,35,37,43,46,49], '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(8,  4171, 'sys11_t1', [4,8,10,12,13,15,17,22,24,28,30,31,32,33,34,35,37,40,43,46,49], '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(9,  4171, 'sys12_t1', [1,4,6,8,10,12,13,14,15,17,18,22,24,28,30,31,32,33,34,35,37,38,40,43,46,48,49], '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(10, 4171, 'bonus_t6', [4,21,30,35,37,49],                            '003c985', 4170, '...DRY RUN', 2026-04-04 04:26:29)
(11, 4171, 'sys6_t1',  [4,15,17,35,43,49],                            '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(12, 4171, 'sys6_t2',  [8,13,22,34,37,49],                            '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(13, 4171, 'sys6_t3',  [4,22,31,34,35,49],                            '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(14, 4171, 'sys7_t1',  [4,15,17,22,35,43,49],                         '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(15, 4171, 'sys8_t1',  [4,15,17,22,34,35,37,43,49],                   '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(16, 4171, 'sys9_t1',  [4,12,13,15,17,22,28,34,35,37,43,49],          '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(17, 4171, 'sys10_t1', [4,8,10,12,13,15,17,22,28,30,34,35,37,43,46,49], '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(18, 4171, 'sys11_t1', [4,8,10,12,13,15,17,22,24,28,30,31,32,33,34,35,37,40,43,46,49], '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(19, 4171, 'sys12_t1', [1,4,6,8,10,12,13,14,15,17,18,22,24,28,30,31,32,33,34,35,37,38,40,43,46,48,49], '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
(20, 4171, 'bonus_t6', [4,21,30,35,37,49],                            '1680776', 4170, 'trained on 1830 draws (>= draw #2341)', 2026-04-04 06:50:20)
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
1. **scrape_latest.py is NOT reliably working** — hits JS-rendered ASPX page via BeautifulSoup.
   - File exists: `~/statlot-649/statlot/toto/scrape_latest.py`
   - TOTO_URL = `https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx`
   - latest_draw.json shows draw #4169 (fetched 2026-04-02) — STALE by 2+ draws
   - The regex/BS4 parsing is fragile; worked once, will fail when page structure changes
   - **Fix required:** Use static archive URL instead:
     `https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_top_draws_en.html`
   - Save output to: `~/statlot-649/statlot/toto/latest_draw.json`
   - **Status: NOT FIXED — next session priority #1**

2. **Win-check has never run** — `toto_results` has 0 rows.
   - `check_wins.py` exists but has never been tested end-to-end
   - Cannot verify prediction quality until this is fixed

3. **Corrupted early draw dates** — draws #4 through ~#2340 have wrong/garbage dates
   - Impact: toto_draws.draw_date unreliable for pre-2341 draws
   - Mitigation: training already excludes draws < #2341, so prediction quality unaffected
   - But: draw range queries by date will return wrong results

### P1 — Known issues, lower urgency
4. **Backtest numbers are invalid** — all cited lift/accuracy numbers (13.6% Sys7 etc.)
   were computed on pre-2341 data (corrupted/incomplete). Not meaningful.
5. **4D pipeline SSH key problem** — 4D automation not verified end-to-end.

---

## COMPLETED THIS SESSION (2026-04-04)

| # | Task | Evidence |
|---|------|----------|
| 1 | DB gitignore confirmed at repo root | .gitignore verified, commit 09b58b1 |
| 2 | toto_draws filled with 2,973 rows | DB verified: COUNT=2973, MIN=#4, MAX=#4170 |
| 3 | toto_predictions_log table created + tested | 20 rows in DB (10 dry-run + 10 real). Full pipeline exit code 0 on t3.medium. |
| 4 | SSH key fix: ssh_utils.py written and deployed | commit 519960b (ssh_utils.py), commit 9681b20 (toto_result_poller.ts updated). restore_key.py in sandbox scripts. Key restored from EC2_SSH_KEY env var at automation start — no longer relies on /tmp persisting. |
| 5 | boto3 confirmed working for EC2 control | AWS CLI mangles + in secrets; boto3 works. ec2_control/run.sh updated to use boto3. |

**Full pipeline stdout (t3.medium run, 2026-04-04 06:50 UTC):**
```
Git commit: 1680776
Training window: draws >= 2341 only (pre-2341 excluded by rule)
Training draws loaded from toto_draws: 1830 draws (#2341–#4170)
Generating candidates (trained on 1830 draws)...
Schema initialised at /home/ubuntu/statlot-649/statlot_toto.duckdb

============================================================
  PREDICTION SAVED — pred_id: pred_4171_20260404_065020
  For draw: #4171 | Trained on 1830 draws (>= #2341)
  Git version: 1680776 | Dry run: False
============================================================
  Sys6 T1: [4, 15, 17, 35, 43, 49]
  Sys6 T2: [8, 13, 22, 34, 37, 49]
  Sys6 T3: [4, 22, 31, 34, 35, 49]
  Sys7 T1: [4, 15, 17, 22, 35, 43, 49]
  Sys8 T1: [4, 15, 17, 22, 34, 35, 37, 43, 49]
  Sys9 T1: [4, 12, 13, 15, 17, 22, 28, 34, 35, 37, 43, 49]
  Sys10 T1:[4, 8, 10, 12, 13, 15, 17, 22, 28, 30, 34, 35, 37, 43, 46, 49]
  Sys11 T1:[4, 8, 10, 12, 13, 15, 17, 22, 24, 28, 30, 31, 32, 33, 34, 35, 37, 40, 43, 46, 49]
  Sys12 T1:[1, 4, 6, 8, 10, 12, 13, 14, 15, 17, 18, 22, 24, 28, 30, 31, 32, 33, 34, 35, 37, 38, 40, 43, 46, 48, 49]
  Bonus T6:[4, 21, 30, 35, 37, 49]
  Additional picks: [21, 48, 31, 17, 6]
  Cost mandatory (3×Sys6 + 1×Sys7): $10
  Cost with Sys8: $38  |  Cost with Sys12: $934

  toto_predictions_log rows written: 10
    → sys6_t1     draw#4171  numbers=[4, 15, 17, 35, 43, 49]  model=1680776
    → sys6_t2     draw#4171  numbers=[8, 13, 22, 34, 37, 49]  model=1680776
    → sys6_t3     draw#4171  numbers=[4, 22, 31, 34, 35, 49]  model=1680776
    → sys7_t1     draw#4171  numbers=[4, 15, 17, 22, 35, 43, 49]  model=1680776
    → sys8_t1     draw#4171  numbers=[4, 15, 17, 22, 34, 35, 37, 43, 49]  model=1680776
    → sys9_t1     draw#4171  numbers=[4, 12, 13, 15, 17, 22, 28, 34, 35, 37, 43, 49]  model=1680776
    → sys10_t1    draw#4171  numbers=[4, 8, 10, 12, 13, 15, 17, 22, 28, 30, 34, 35, 37, 43, 46, 49]  model=1680776
    → sys11_t1    draw#4171  numbers=[4, 8, 10, 12, 13, 15, 17, 22, 24, 28, 30, 31, 32, 33, 34, 35, 37, 40, 43, 46, 49]  model=1680776
    → sys12_t1    draw#4171  numbers=[1, 4, 6, 8, 10, 12, 13, 14, 15, 17, 18, 22, 24, 28, 30, 31, 32, 33, 34, 35, 37, 38, 40, 43, 46, 48, 49]  model=1680776
    → bonus_t6    draw#4171  numbers=[4, 21, 30, 35, 37, 49]  model=1680776
EXIT CODE: 0
```

**Note on t3.nano OOM:** retrain_and_predict.py requires t3.medium (or larger).
Running it on t3.nano exits with code 137 (SIGKILL / OOM). Always scale up first.

---

## TASK LIST (current)

### IN PROGRESS
- Nothing in progress (session ended 2026-04-04)

### NEXT UP (in priority order)
1. **Fix scrape_latest.py** — switch from JS-rendered ASPX URL to static archive URL.
   Static URL: `https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_top_draws_en.html`
   Output path: `~/statlot-649/statlot/toto/latest_draw.json`
   Must parse draw_number, draw_date, winning_numbers (list of 6), additional.
   Test: run manually, verify latest_draw.json matches Singapore Pools website.

2. **Run win-check end-to-end** — populate toto_results table with outcomes for existing predictions.
   Script: `~/statlot-649/statlot/toto/check_wins.py`
   Requires: latest_draw.json to be correct first (depends on task 1).

3. **Fix corrupted draw dates** — draws #4–#2340 have wrong dates in toto_draws table.
   Low urgency (not used in training) but makes queries unreliable.

4. **Re-run backtest on clean data** — only valid after full draw history is confirmed clean.
   Do NOT re-quote old lift numbers — they are based on incomplete/corrupted data.

### COMPLETED (all sessions, cumulative)
- All models M1–M9 committed to git ✅
- 4D DuckDB built with 64 enriched feature columns ✅
- TOTO pipeline scripts committed to correct path (statlot/toto/) ✅
- toto_db.py confirmed pointing to correct canonical DB path ✅
- AGENT_STATE.md and AGENT_RULES.md added to repo root (commit 09b58b1) ✅
- toto_draws table filled: 2,973 rows, draws #4–#4170 (commit: scraper session 2026-04-03) ✅
- toto_predictions_log table created and tested: 20 rows, draw #4171 (2026-04-04) ✅
- SSH key fix deployed: ssh_utils.py (commit 519960b), toto_result_poller.ts (commit 9681b20) ✅
- restore_key.py in sandbox .agents/scripts/ ✅
- EC2 control updated to use boto3 (not AWS CLI which mangles + in secrets) ✅
- retrain_and_predict.py full end-to-end run verified on t3.medium, exit code 0 ✅

---

## GIT COMMIT HISTORY (recent, relevant)

| Commit  | Description |
|---------|-------------|
| 9681b20 | fix: toto_result_poller — restore SSH key from env var, never rely on /tmp persisting |
| 519960b | feat: ssh_utils.py — restore SSH key from Secrets Manager (syntax verified) |
| ad2c5e5 | feat: add ssh_utils.py — restore SSH key from Secrets Manager on every run (first attempt, superseded) |
| 09b58b1 | Add AGENT_STATE.md and AGENT_RULES.md — session ground truth files |
| 1680776 | Most recent retrain run (model version used in draw #4171 predictions) |
| 003c985 | Dry-run retrain for draw #4171 |

---

## MODEL INVENTORY

| Model            | File                              | Notes |
|------------------|-----------------------------------|-------|
| M1 Bayesian      | engine/models/m1_bayes.py         | In git ✅ |
| M2 EV/Kelly      | engine/models/m2_ev_kelly.py      | In git ✅ |
| M3 Random Forest | engine/models/m3_rf.py            | In git ✅ |
| M4 Monte Carlo   | engine/models/m4_monte_carlo.py   | In git ✅ |
| M5 XGBoost       | engine/models/m5_xgb.py           | In git ✅ |
| M6 DQN           | engine/models/m6_dqn.py           | In git ✅ |
| M7 Markov        | engine/models/m7_markov.py        | In git ✅ |
| M8 FFT/GMM       | engine/models/m8_gmm.py           | In git ✅ |
| M9 LSTM          | engine/models/m9_lstm.py          | In git ✅ |
| Additional       | engine/models/additional.py       | In git ✅ |

⚠️ All backtest lift numbers previously quoted are INVALID — trained on incomplete/corrupted data.
Do not cite them without first re-running on clean full history.

---

## EC2 RULES
- Instance at rest: **t3.nano** (3.1.133.166 — Elastic IP, never changes)
- Scale to **t3.medium** for retrain (retrain_and_predict.py OOMs on t3.nano — exit 137)
- Scale back to t3.nano immediately after — do not leave on medium
- Never scale beyond t3.medium without explicit instruction from Bharath
- Always use **boto3** for EC2 control — AWS CLI mangles the + character in secrets
- SSH key: always restore from EC2_SSH_KEY env var via restore_key.py before any SSH call
