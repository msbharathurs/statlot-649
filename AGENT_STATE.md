# AGENT_STATE.md
# Statlot Project — Live Ground Truth
# Updated: 2026-04-05 (session end)
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
| scrape_latest.py      | ~/statlot-649/statlot/toto/scrape_latest.py                 | ✅ FIXED 2026-04-05 |
| latest_draw.json      | ~/statlot-649/statlot/toto/latest_draw.json                 | ✅ draw #4170 |

## DO NOT USE THESE PATHS — they are stale duplicates:
- `~/statlot-649/statlot/statlot_toto.duckdb` — 12KB EMPTY
- `~/statlot-649/statlot.duckdb` — 274KB old draft schema
- Any path under `.agents/scripts/` — sandbox only, wiped every session

---

## CURRENT DB STATE (verified 2026-04-05)

### TOTO DuckDB (`/home/ubuntu/statlot-649/statlot_toto.duckdb`)

| Table                | Rows | Notes |
|----------------------|------|-------|
| toto_draws           | 2,973 | Draws #4–#4170. Range: min=#4, max=#4170. |
| toto_predictions     | 3    | Draws #4168, #4170, #4171 |
| toto_results         | 0    | Win-check has NEVER run — still broken |
| toto_predictions_log | 20   | ✅ Table exists. 10 dry-run rows + 10 real rows for draw #4171 |
| v_weekly_summary     | 3    | View, not a real table |

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

1. **Win-check has never run** — `toto_results` has 0 rows.
   - `check_wins.py` exists but has never been tested end-to-end
   - Cannot verify prediction quality until this is fixed
   - **Status: UNFIXED**

2. **GitHub PAT is expired** — GITHUB_TOKEN in Base44 secrets returns 401.
   - EC2 cannot push to GitHub (pull still works via HTTPS)
   - Sandbox cannot push to GitHub either
   - EC2 is currently **4 commits ahead of origin** (including c55517c AGENT_RULES.md)
   - **Fix:** Bharath must generate a new GitHub PAT (classic, repo scope) and update
     the GITHUB_TOKEN secret in Base44 secrets vault
   - **Status: BLOCKED on Bharath**

3. **Corrupted early draw dates** — draws #4 through ~#2340 have wrong/garbage dates
   - Impact: toto_draws.draw_date unreliable for pre-2341 draws
   - Mitigation: training already excludes draws < #2341
   - **Status: LOW PRIORITY, not blocking anything**

### P1 — Known issues, lower urgency
4. **Backtest numbers are invalid** — all cited lift/accuracy numbers (13.6% Sys7 etc.)
   were computed on pre-2341 data (corrupted/incomplete). Not meaningful.
5. **4D pipeline SSH key problem** — 4D automation not verified end-to-end.

---

## COMPLETED THIS SESSION (2026-04-05)

| # | Task | Evidence |
|---|------|----------|
| 1 | AGENT_RULES.md committed to repo root | EC2 commit c55517c. Push blocked (GITHUB_TOKEN expired). |
| 2 | scrape_latest.py FIXED | Now uses static archive URL. Fetched draw #4170 correctly. latest_draw.json verified. Exit code 0. Full output pasted in session. |
| 3 | bs4 + requests installed in venv | pip install confirmed exit code 0 |

**scrape_latest.py fix summary:**
- OLD URL: `https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx` (JS-rendered, fragile)
- NEW URL: `https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_top_draws_en.html` (static HTML, reliable)
- Output path: `~/statlot-649/statlot/toto/latest_draw.json` ✅ (was writing to wrong path before)
- Actual output from 2026-04-05 run:
  ```
  Draw number : 4170
  Draw date   : 2026-04-02
  Numbers     : [1, 7, 8, 23, 30, 33]
  Additional  : 21
  Fetched at  : 2026-04-04T13:08:26
  ```

---

## PREVIOUS SESSION COMPLETIONS (2026-04-04)

| # | Task | Evidence |
|---|------|----------|
| 1 | DB gitignore confirmed at repo root | .gitignore verified, commit 09b58b1 |
| 2 | toto_draws filled with 2,973 rows | DB verified: COUNT=2973, MIN=#4, MAX=#4170 |
| 3 | toto_predictions_log table created + tested | 20 rows in DB (10 dry-run + 10 real). Full pipeline exit code 0 on t3.medium. |
| 4 | SSH key fix: ssh_utils.py written and deployed | commit 519960b. Key restored from EC2_SSH_KEY env var at automation start. |
| 5 | boto3 confirmed working for EC2 control | boto3 works; AWS CLI mangles + in secrets. |

---

## NEXT STEPS (in priority order)

1. **Bharath: regenerate GitHub PAT** (classic, repo scope) → update GITHUB_TOKEN in Base44 secrets
   → then agent can `git push` from sandbox and sync EC2 (4 commits pending)

2. **Fix win-check** — test check_wins.py against draw #4170 using toto_predictions_log rows
   → verify toto_results gets rows written
   → confirm exit code 0

3. **End-to-end automation test** — manually trigger the Monday pipeline, paste full output,
   verify DB changes (new prediction row, win-check row)
