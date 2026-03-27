#!/bin/bash
# ============================================================
# run_pipeline.sh — Full 4D Prediction Pipeline
# Phase 1: M1–M7 train + backtest  (t3.medium)
# Phase 2: M8–M10 ML models        (c5.xlarge — run separately)
# Phase 3: Final predictions        (any instance)
# ============================================================

VENV=~/statlot-649/statlot/venv/bin/python
BASE=~/statlot-649/statlot/4d
LOG_DIR=~/statlot-649/logs
mkdir -p $LOG_DIR

echo "========================================"
echo "4D Prediction Pipeline — Phase 1"
echo "Started: $(date)"
echo "========================================"

# Step 1: Verify DuckDB exists
if [ ! -f ~/statlot-649/draws_4d.duckdb ]; then
    echo "[ERROR] DuckDB not found. Run build_db.py first."
    exit 1
fi
echo "[OK] DuckDB found."

# Step 2: Run backtest (M1–M7, walk-forward)
echo ""
echo "--- Step 2: Backtesting M1–M7 ---"
$VENV -u $BASE/backtest.py 2>&1 | tee $LOG_DIR/backtest_phase1.log
if [ $? -ne 0 ]; then echo "[FAILED] Backtest failed."; exit 1; fi
echo "[OK] Backtest complete."

# Step 3: Generate final predictions (M1–M7 ensemble)
echo ""
echo "--- Step 3: Generating predictions ---"
$VENV -u $BASE/predict.py 2>&1 | tee $LOG_DIR/predict_phase1.log
if [ $? -ne 0 ]; then echo "[FAILED] Prediction failed."; exit 1; fi
echo "[OK] Predictions generated."

echo ""
echo "========================================"
echo "Phase 1 Complete: $(date)"
echo "Results in ~/statlot-649/backtest_results/"
echo "Predictions in ~/statlot-649/predictions/"
echo "========================================"
echo ""
echo "NEXT: Upgrade to c5.xlarge, then run phase2_ml.sh"
