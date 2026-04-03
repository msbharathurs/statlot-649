# AGENT_STATE.md
# Statlot Project — Live Ground Truth
# Updated: 2026-04-04
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
| Agent state           | ~/statlot-649/AGENT_STATE.md                                | ✅ This file |
| Agent rules           | ~/statlot-649/AGENT_RULES.md                                | ✅ In git |
| Python venv           | /home/ubuntu/statlot-649/statlot/venv/bin/python3           | ✅ Confirmed |

## DO NOT USE THESE PATHS — they are stale duplicates:
- `~/statlot-649/statlot/statlot_toto.duckdb` — 12KB EMPTY
- `~/statlot-649/statlot.duckdb` — 274KB old draft schema
- Any path under `.agents/scripts/` — sandbox only, wiped every session

---

## CURRENT DB STATE (update row counts every session)

### TOTO DuckDB (`/home/ubuntu/statlot-649/statlot_toto.duckdb`)

| Table                | Rows        | Notes |
|----------------------|-------------|-------|
| toto_draws           | 1           | Only draw #4169. CRITICAL GAP — see below |
| toto_predictions     | 2           | Draws #4168 and #4170 only |
| toto_results         | 0           | Win-check has NEVER run |
| toto_predictions_log | NOT CREATED | Needs to be built — see task list |

### 4D DuckDB (`/home/ubuntu/statlot-649/draws_4d.duckdb`)

| Table        | Rows    | Notes |
|--------------|---------|-------|
| draws        | 4,473   | Full 4D history ✅ |
| draw_numbers | 102,823 | 64 enriched feature columns ✅ |

---

## CRITICAL DATA GAP — TOTO DRAW HISTORY

`sp_historical_draws.json` has 2,298 draws BUT:
- Draws #2341 to #4168 are MISSING — 1,828 draws (~9 years, 2008–2026)
- Draws #4 to #2340 have CORRUPTED dates (draw #4 shows "23 Mar 2026" instead of 1986)
- Draw #539 date is "01 Jan 0001" — completely broken
- Early draws have inconsistent schema (extra `format` and `source` fields)
- Draw #4169 and #4170 are the only recent draws present

**Impact:** Every model trained so far has only seen pre-2008 data + 2 draws. Every backtest
number ever quoted is based on this incomplete dataset. No retrain is worth running until this gap is filled.

**Fix required:** Scrape draws #2341–#4168 from Singapore Pools and append to JSON, then fix
the date corruption, then rebuild toto_draws table from scratch.

The SP static archive URL works:
- `https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx` → 200
- `toto_result_top_draws_en.html` → 200

---

## BROKEN THINGS (do not paper over these)

### P0 — Must fix before any retrain is meaningful
1. **1,828 missing TOTO draws** — toto_draws table has 1 row. Historical JSON has 9-year gap.
2. **Corrupted dates in sp_historical_draws.json** — early draw dates are wrong/garbage.
3. **toto_predictions_log table does not exist** — all predictions after retrain need a permanent home.

### P1 — Fix next
4. **Win-check has never run** — toto_results has 0 rows. We don't know if any prediction ever hit.
5. **scrape_latest.py is broken** — hits JS-rendered URL (regex never matches), saves to wrong path.
6. **SSH key at /tmp/statlot.pem is ephemeral** — wiped between automation runs. Pipeline always fails at SSH step.

### P2 — Known but lower priority
7. **Backtest numbers are invalid** — all cited numbers (13.6% Sys7 lift etc.) were on incomplete data.
8. **4D pipeline has same SSH key problem** — not verified working end-to-end.

---

## TASK LIST (current)

### IN PROGRESS
- Nothing currently in progress (session ended 2026-04-03)

### NEXT UP (in priority order)
1. **Build TOTO draw history scraper** — fetch draws #2341–#4168 from SP, fix date corruption, append to sp_historical_draws.json, rebuild toto_draws table
2. **Create toto_predictions_log table** — schema: draw_no, predicted_numbers (array), model_version, retrain_date, confidence_scores, created_at
3. **Fix scrape_latest.py** — use working static URL, save to correct path (toto/latest_draw.json)
4. **Fix SSH key handling** — write key from secret on each automation run, not relying on /tmp/
5. **Run win-check on all stored predictions** — populate toto_results table
6. **Re-run backtests on complete data** — only after draws #2341–#4168 are filled

### COMPLETED
- All models M1–M9 committed to git ✅
- 4D DuckDB built with 64 enriched feature columns ✅
- TOTO pipeline scripts committed to correct path (statlot/toto/) ✅
- toto_db.py confirmed pointing to correct canonical DB path ✅

---

## MODEL INVENTORY (what exists in git)

| Model            | File                              | Trained? | Data it trained on       |
|------------------|-----------------------------------|----------|--------------------------|
| M1 Bayesian      | engine/models/m1_bayes.py         | Yes      | 649 full history         |
| M2 Poisson       | engine/models/m2_ev_kelly.py      | Yes      | 649 full history         |
| M3 Random Forest | engine/models/m3_rf.py            | Yes      | 649 draws #1–2340 only   |
| M4 Monte Carlo   | engine/models/m4_monte_carlo.py   | Yes      | 649 full history         |
| M5 XGBoost       | engine/models/m5_xgb.py           | Yes      | 649 draws #1–2340 only   |
| M6 DQN           | engine/models/m6_dqn.py           | Yes      | 649 draws #1–2340 only   |
| M7 Markov        | engine/models/m7_markov.py        | Yes      | 649 full history         |
| M8 FFT/GMM       | engine/models/m8_gmm.py           | Yes      | 649 draws #1–2340 only   |
| M9 LSTM          | engine/models/m9_lstm.py          | Yes      | 649 draws #1–2340 only   |

⚠️ TOTO-specific retraining has never successfully completed end-to-end.
All TOTO predictions so far are 649 model output applied to TOTO with no TOTO-specific validation.

---

## LAST SESSION SUMMARY
**Date:** 2026-04-03
**What happened:** Forensic audit of entire system. Discovered:
- toto_db.py path is actually correct (points to right file)
- sp_historical_draws.json has 1,828-draw gap and date corruption
- scrape_latest.py hits JS-rendered page (never works)
- SSH key is ephemeral (all automations have been failing silently)
- toto_predictions_log table was never created
- All backtest numbers are based on incomplete training data

**What was committed:** AGENT_STATE.md and AGENT_RULES.md added to repo root.
**Git state:** Clean, up to date with main.
**Next session must start with:** Building the draw history scraper to fill the 1,828-draw gap.
