# AGENT_STATE.md
# Statlot Project — Live Ground Truth
# Updated: 2026-04-05 session 2 (check_wins fix + automation fix complete)
# THIS FILE IS THE FIRST THING YOU READ EVERY SESSION. KEEP IT CURRENT.

## WHAT IS THIS PROJECT
Statlot is a lottery prediction system for Singapore TOTO and 4D.
Models M1–M9 (Bayesian, Poisson, RF, Monte Carlo, XGBoost, DQN, Markov, FFT/GMM, LSTM)
trained on historical draw data stored in DuckDB. Runs on EC2 (t3.nano), scrapes from
Singapore Pools after each draw, retrains, stores predictions in DuckDB.

## CANONICAL PATHS

| Resource              | Path                                                        | Status |
|-----------------------|-------------------------------------------------------------|--------|
| TOTO DuckDB           | /home/ubuntu/statlot-649/statlot_toto.duckdb                | ✅ |
| 4D DuckDB             | /home/ubuntu/statlot-649/draws_4d.duckdb                    | ✅ 4,473 draws |
| TOTO scripts          | ~/statlot-649/statlot/toto/                                 | ✅ In git |
| Engine models         | ~/statlot-649/statlot/engine/models/                        | ✅ M1–M9 in git |
| Agent state           | ~/statlot-649/AGENT_STATE.md                                | ✅ This file |
| Agent rules           | ~/statlot-649/AGENT_RULES.md                                | ✅ commit c55517c |
| Python venv           | /home/ubuntu/statlot-649/statlot/venv/bin/python3           | ✅ All deps installed |
| scrape_latest.py      | ~/statlot-649/statlot/toto/scrape_latest.py                 | ✅ FIXED commit 7a5c887 |
| check_wins.py         | ~/statlot-649/statlot/toto/check_wins.py                    | ✅ FIXED commit 9cb7743 |
| retrain_and_predict.py| ~/statlot-649/statlot/toto/retrain_and_predict.py           | ✅ Working commit 7a5c887 |
| toto_result_poller.ts | functions/toto_result_poller.ts (Base44 backend)            | ✅ FIXED commit 3f4d516 |
| latest_draw.json      | ~/statlot-649/statlot/toto/latest_draw.json                 | ✅ draw #4170 |

## DO NOT USE THESE PATHS — stale duplicates:
- `~/statlot-649/statlot/statlot_toto.duckdb` — 12KB EMPTY
- `~/statlot-649/statlot.duckdb` — 274KB old draft schema
- Any path under `.agents/scripts/` — sandbox only, wiped every session

---

## CURRENT DB STATE (verified 2026-04-05 ~14:00 UTC)

### TOTO DuckDB (`/home/ubuntu/statlot-649/statlot_toto.duckdb`)

| Table                | Rows | Notes |
|----------------------|------|-------|
| toto_draws           | 2,973 | Draws #4–#4170. Training window: #2341–#4170 = 1830 draws |
| toto_predictions     | 4    | Legacy table — NOT read by any active script |
| toto_results         | 2    | Row 1=legacy pred_id, Row 2=log_4171_model_7a5c887_result_4170 (new format ✅) |
| toto_predictions_log | 30   | Latest 10: draw_no=4171, model=7a5c887, predicted_at=2026-04-04 13:38 |
| v_weekly_summary     | 4    | View |

### 4D DuckDB (`/home/ubuntu/statlot-649/draws_4d.duckdb`)

| Table        | Rows    | Notes |
|--------------|---------|-------|
| draws        | 4,473   | Full 4D history ✅ |
| draw_numbers | 102,823 | 64 enriched feature columns ✅ |

---

## AUTOMATION SCHEDULE

| Automation | Cron (UTC) | SGT | Days | What it calls |
|---|---|---|---|---|
| TOTO Result Poller | 45 10 * * 1,4 | 6:45 PM | Mon+Thu | toto_result_poller backend function |
| TOTO Safety Watchdog | 0 13 * * 1,4 | 9:00 PM | Mon+Thu | Checks EC2 type, kills if still t3.medium |
| 4D Post-Draw Pipeline | 30 11 * * 3,6,0 | 7:30 PM | Wed+Sat+Sun | 4D pipeline |
| 4D Safety Watchdog | 30 12 * * 3,6,0 | 8:30 PM | Wed+Sat+Sun | 4D watchdog |

### TOTO Result Poller — full pipeline (commit 3f4d516):
1. Restore SSH key from EC2_SSH_KEY env var
2. Scale EC2: t3.nano → t3.medium
3. git pull origin main (picks up latest script fixes from GitHub)
4. python3 -m toto.scrape_latest → updates latest_draw.json
5. python3 -m toto.check_wins → reads toto_predictions_log, writes toto_results
6. python3 -m toto.retrain_and_predict → trains on draws #2341+, writes toto_predictions_log
7. Scale EC2: t3.medium → t3.nano
8. broadcast_message to Telegram + web

---

## BROKEN THINGS

### P1 — Known, lower urgency
1. **Corrupted early draw dates** — draws #4–#2340 have wrong/garbage dates
   - Mitigation: training excludes draws < #2341. Not blocking.
2. **Backtest numbers invalid** — all lift/accuracy numbers from pre-2341 data. Not meaningful.
3. **4D pipeline** — not fully verified end-to-end (SSH key issue, script path issues).
4. **Legacy toto_results row** — one row still uses old pred_id format. Harmless.

### RESOLVED THIS SESSION
- ✅ check_wins.py now reads from toto_predictions_log (commit 9cb7743)
- ✅ toto_result_poller.ts now runs all 4 steps in order (commit 3f4d516)

---

## COMPLETED (cumulative)

| Date | Task | Evidence |
|------|------|----------|
| 2026-04-05 s1 | scrape_latest.py FIXED | commit 7a5c887 |
| 2026-04-05 s1 | Win-check ran for #4170 | toto_results: 1→2 rows |
| 2026-04-05 s1 | Retrain + predict #4171 | toto_predictions_log: 20→30 rows, model=7a5c887 |
| 2026-04-05 s1 | AGENT_RULES.md in git | commit c55517c |
| 2026-04-05 s2 | check_wins.py reads toto_predictions_log | commit 9cb7743, tested on EC2 ✅ |
| 2026-04-05 s2 | toto_result_poller.ts full pipeline fixed | commit 3f4d516, deployed |

## GIT LOG (recent)

| Hash    | Description |
|---------|-------------|
| 3f4d516 | fix: toto_result_poller now runs full pipeline: git pull → scrape → check_wins → retrain |
| 9cb7743 | fix: check_wins.py reads from toto_predictions_log (not legacy toto_predictions) |
| 473bdb7 | chore: update AGENT_STATE.md — session 2026-04-05, tasks 1-4 complete |
| 7a5c887 | fix: scrape_latest.py uses static HTML archive URL; update AGENT_STATE.md |
| c55517c | feat: add AGENT_RULES.md to repo root — session operating rules |

---

## NEXT STEPS

1. **Monday 07 Apr 2026 — verify automation end-to-end**
   - TOTO Result Poller fires at 6:45 PM SGT
   - After it runs: check DB for new toto_draws row, new toto_predictions_log rows (+10),
     new toto_results row, EC2 back on t3.nano
   - Paste full output. If any step failed, fix it.

2. **4D pipeline** — verify scrape_latest, check_wins, predict scripts work on EC2 (separate session)

3. **Corrupted draw dates** — low priority, not blocking
