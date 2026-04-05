# AGENT_STATE.md
# Statlot Project — Live Ground Truth
# Updated: 2026-04-05 session 3 (4D pipeline fixed end-to-end)
# THIS FILE IS THE FIRST THING YOU READ EVERY SESSION. KEEP IT CURRENT.

## WHAT IS THIS PROJECT
Statlot is a lottery prediction system for Singapore TOTO and 4D.
Models M1–M9 (Bayesian, Poisson, RF, Monte Carlo, XGBoost, DQN, Markov, FFT/GMM, LSTM)
trained on historical draw data stored in DuckDB. Runs on EC2 (t3.nano), scrapes from
Singapore Pools after each draw, retrains, stores predictions in DuckDB.

## CANONICAL PATHS

| Resource                   | Path                                                               | Status |
|----------------------------|--------------------------------------------------------------------|--------|
| TOTO DuckDB                | /home/ubuntu/statlot-649/statlot_toto.duckdb                       | ✅ |
| 4D DuckDB                  | /home/ubuntu/statlot-649/draws_4d.duckdb                           | ✅ 4,475 draws |
| TOTO scripts               | ~/statlot-649/statlot/toto/                                        | ✅ In git |
| 4D scripts                 | ~/statlot-649/statlot/4d/                                          | ✅ In git |
| Engine models              | ~/statlot-649/statlot/engine/models/                               | ✅ M1–M9 in git |
| Agent state                | ~/statlot-649/AGENT_STATE.md                                       | ✅ This file |
| Agent rules                | ~/statlot-649/AGENT_RULES.md                                       | ✅ commit c55517c |
| Python venv                | /home/ubuntu/statlot-649/statlot/venv/bin/python3                  | ✅ All deps installed |
| TOTO scrape_latest.py      | ~/statlot-649/statlot/toto/scrape_latest.py                        | ✅ FIXED commit 7a5c887 |
| TOTO check_wins.py         | ~/statlot-649/statlot/toto/check_wins.py                           | ✅ FIXED commit 9cb7743 |
| TOTO retrain_and_predict.py| ~/statlot-649/statlot/toto/retrain_and_predict.py                  | ✅ Working commit 7a5c887 |
| toto_result_poller.ts      | functions/toto_result_poller.ts (Base44 backend)                   | ✅ FIXED commit 3f4d516 |
| TOTO latest_draw.json      | ~/statlot-649/statlot/toto/latest_draw.json                        | ✅ draw #4170 |
| 4D scrape_latest.py        | ~/statlot-649/statlot/4d/scrape_latest.py                          | ⚠️ Uses static HTML URL — NOT YET VERIFIED end-to-end |
| 4D check_wins.py           | ~/statlot-649/statlot/4d/check_wins.py                             | ✅ FIXED commit e42b2a4 — writes to results_log |
| 4D retrain_and_predict.py  | ~/statlot-649/statlot/4d/retrain_and_predict.py                    | ✅ FIXED commit 495ef62 — writes to predictions_log |
| predict_4d_post_draw.ts    | functions/predict_4d_post_draw.ts (Base44 backend)                 | ✅ NEW — full pipeline function |
| 4D latest_draw.json        | ~/statlot-649/predictions/latest_draw.json                         | ✅ draw 2026-04-01 |

## DO NOT USE THESE PATHS — stale duplicates:
- `~/statlot-649/statlot/statlot_toto.duckdb` — 12KB EMPTY
- `~/statlot-649/statlot.duckdb` — 274KB old draft schema
- `~/statlot-649/statlot/4d/draws_4d.duckdb` — DELETED (was empty, removed commit fd5453e)
- Any path under `.agents/scripts/` — sandbox only, wiped every session

---

## CURRENT DB STATE (verified 2026-04-05 session 3)

### TOTO DuckDB (`/home/ubuntu/statlot-649/statlot_toto.duckdb`)

| Table                | Rows | Notes |
|----------------------|------|-------|
| toto_draws           | 2,973 | Draws #4–#4170. Training window: #2341–#4170 = 1830 draws |
| toto_predictions     | 4    | Legacy table — NOT read by any active script |
| toto_results         | 2    | Row 2 = new format ✅ |
| toto_predictions_log | 30   | Latest 10: draw_no=4171, model=7a5c887 |
| v_weekly_summary     | 4    | View |

### 4D DuckDB (`/home/ubuntu/statlot-649/draws_4d.duckdb`)

| Table            | Rows    | Notes |
|------------------|---------|-------|
| draws            | 4,475   | Full 4D history ✅ |
| draw_numbers     | 102,823 | 64 enriched feature columns ✅ |
| predictions_log  | 13      | First real run: draw_date=2026-04-08, model=495ef62 ✅ |
| results_log      | 1       | draw_date=2026-04-01, matched=False, no_wins ✅ |

---

## AUTOMATION SCHEDULE

| Automation | Cron (UTC) | SGT | Days | What it calls |
|---|---|---|---|---|
| TOTO Result Poller | 45 10 * * 1,4 | 6:45 PM | Mon+Thu | toto_result_poller backend function |
| TOTO Safety Watchdog | 0 13 * * 1,4 | 9:00 PM | Mon+Thu | Checks EC2 type, kills if still t3.medium |
| 4D Post-Draw Pipeline | 30 11 * * 3,6,0 | 7:30 PM | Wed+Sat+Sun | predict_4d_post_draw backend function |
| 4D Safety Watchdog | 30 12 * * 3,6,0 | 8:30 PM | Wed+Sat+Sun | 4D watchdog |

### TOTO pipeline (toto_result_poller.ts, commit 3f4d516):
1. Restore SSH key → scale t3.nano → t3.medium → git pull
2. toto.scrape_latest → toto.check_wins → toto.retrain_and_predict
3. Scale back t3.nano → broadcast_message

### 4D pipeline (predict_4d_post_draw.ts, session 3):
1. Restore SSH key → scale t3.nano → t3.medium → git pull
2. 4d.scrape_latest → 4d.check_wins → 4d.retrain_and_predict
3. Scale back t3.nano → broadcast_message

---

## BROKEN THINGS

### P1 — Needs verification before next Wed draw
1. **4D scrape_latest.py — NOT end-to-end verified**
   - Uses static HTML URL (same pattern as TOTO fix). Looks correct but never run through automation.
   - Must verify on Wed 2026-04-08 draw day.

### P2 — Known, lower urgency
2. **Corrupted early draw dates** — draws #4–#2340 have wrong/garbage dates. Not blocking.
3. **Backtest numbers invalid** — all lift/accuracy from pre-2341 data. Not meaningful.
4. **Legacy toto_results row** — one row still uses old pred_id format. Harmless.

### RESOLVED THIS SESSION (session 3)
- ✅ Duplicate empty 4D DuckDB removed (commit fd5453e)
- ✅ 4D predictions_log + results_log tables created with SEQUENCE auto-id
- ✅ 4D retrain_and_predict.py writes 13 rows to predictions_log (commit 495ef62, tested ✅)
- ✅ 4D check_wins.py writes to results_log (commit e42b2a4, tested ✅)
- ✅ predict_4d_post_draw.ts created and deployed as Base44 backend function

---

## COMPLETED (cumulative)

| Date | Task | Evidence |
|------|------|----------|
| 2026-04-05 s1 | scrape_latest.py FIXED | commit 7a5c887 |
| 2026-04-05 s1 | Win-check ran for #4170 | toto_results: 1→2 rows |
| 2026-04-05 s1 | Retrain + predict #4171 | toto_predictions_log: 20→30 rows |
| 2026-04-05 s1 | AGENT_RULES.md in git | commit c55517c |
| 2026-04-05 s2 | check_wins.py reads toto_predictions_log | commit 9cb7743 ✅ |
| 2026-04-05 s2 | toto_result_poller.ts full pipeline fixed | commit 3f4d516, deployed ✅ |
| 2026-04-05 s3 | 4D duplicate DB removed + gitignore | commit fd5453e ✅ |
| 2026-04-05 s3 | 4D predictions_log + results_log created | DB verified ✅ |
| 2026-04-05 s3 | 4D retrain_and_predict.py → DB write | commit 495ef62, 13 rows ✅ |
| 2026-04-05 s3 | 4D check_wins.py → DB write | commit e42b2a4, 1 row ✅ |
| 2026-04-05 s3 | predict_4d_post_draw.ts created + deployed | Base44 backend ✅ |

## GIT LOG (recent)

| Hash    | Description |
|---------|-------------|
| e42b2a4 | feat: check_wins.py writes results to results_log in draws_4d.duckdb |
| 495ef62 | feat: retrain_and_predict.py writes top10+ibox to predictions_log in draws_4d.duckdb |
| 60bc053 | merge: remove empty 4d duplicate duckdb + gitignore |
| 3f4d516 | fix: toto_result_poller now runs full pipeline |
| 9cb7743 | fix: check_wins.py reads from toto_predictions_log |

---

## NEXT STEPS

1. **Wednesday 08 Apr 2026 — verify 4D automation end-to-end**
   - predict_4d_post_draw fires at 7:30 PM SGT
   - After it runs: check predictions_log for new rows, results_log for win check,
     EC2 back on t3.nano, broadcast_message received
   - If scrape_latest.py fails: fix URL/regex same pattern as TOTO

2. **Monday 07 Apr 2026 — verify TOTO automation end-to-end**
   - toto_result_poller fires at 6:45 PM SGT
   - Same verification checklist as above for TOTO tables

3. **Corrupted draw dates** — low priority, not blocking
