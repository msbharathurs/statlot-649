# StatLot 649 — Full ML Prediction Engine

## Project Structure
```
statlot/
├── api/
│   └── main.py          # FastAPI app — all endpoints
├── db/
│   ├── database.py      # SQLAlchemy engine + session
│   └── models.py        # ORM: draws, features, backtest_runs, predictions
├── engine/
│   ├── features.py      # 39-column feature engineering (data lake)
│   ├── models.py        # RF (M3), XGBoost (M5), Monte Carlo (M4)
│   ├── candidate_gen.py # 6/7/8-number pool generation
│   └── backtest.py      # Rolling-window backtest engine
├── models/              # Saved .pkl model files
├── scripts/
│   ├── export_draws.py      # Seed PostgreSQL from CSV
│   └── run_full_backtest.py # Run all 7 model variants, print table
├── requirements.txt
├── deploy.sh            # One-command EC2 deployment
└── .env.example
```

## Quick Start (EC2)

```bash
# 1. Clone / upload project to EC2
scp -r statlot/ ubuntu@your-ec2-ip:~/statlot/

# 2. SSH in and deploy
ssh ubuntu@your-ec2-ip
cd statlot
bash deploy.sh

# 3. Seed database
python scripts/export_draws.py --csv your_draws.csv

# 4. Build feature lake
curl -X POST http://localhost:8000/features/build

# 5. Run full backtest comparison
python scripts/run_full_backtest.py

# 6. Generate predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pool_size": 7, "n_combos": 10, "use_monte_carlo": true}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /draws/bulk | Ingest draws |
| GET | /draws/count | Total draws in DB |
| POST | /features/build | Build 39-col feature lake |
| GET | /features/{draw_number} | Get features for one draw |
| GET | /features/export/csv | Export full feature CSV |
| POST | /backtest/run | Start backtest (async) |
| GET | /backtest/status/{job_id} | Poll backtest progress |
| GET | /backtest/results | List all backtest runs |
| POST | /predict | Generate predictions |
| GET | /predict/history | List past predictions |

## Models

| Model | Type | What it predicts |
|-------|------|-----------------|
| baseline | Freq+Aging+Pair | Raw combo scoring |
| rf | Random Forest (M3) | Draw property → filter candidates |
| xgb | XGBoost (M5) | Draw property → tighter filter |
| monte_carlo | Monte Carlo (M4) | Rank combos by simulated match rate |

## Pool Sizes

- **6-number**: Standard ticket. Tests exact 6-combo match.
- **7-number**: Pool of 7. C(7,6)=7 embedded combos per pool.
- **8-number**: Pool of 8. C(8,6)=28 embedded combos per pool.

Higher pool = higher chance of 3+ match, but costs more tickets.

## Feature Lake (39 columns)

- `repeat_1..10`: numbers matching the last N draws
- `sum`, `sum_delta`, `sum_ma3/5/10`: sum and moving averages
- `odd_count`, `low_count`, structural stats
- `max_gap`, `min_gap`, `avg_gap`: spacing between numbers
- `decade_1..5`, `empty_decades`: decade distribution
- `avg_freq_last10/20/50`: rolling frequency
- `hot_count`, `cold_count`: vs top/bottom 10 frequency
- `avg_pair_freq`, `max_pair_freq`: pair co-occurrence strength
- `draws_since_any_repeat`, `numbers_from_last2/3`
