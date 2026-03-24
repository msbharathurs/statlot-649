#!/bin/bash
# StatLot 649 — EC2 Setup & Backtest Runner
set -e
echo "=== StatLot 649 EC2 Setup ==="
cd ~

# Update system
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git

# Clone/pull repo
if [ -d ~/statlot-649 ]; then
    cd ~/statlot-649 && git pull origin main
else
    git clone https://github.com/msbharathurs/statlot-649.git ~/statlot-649
    cd ~/statlot-649
fi

cd ~/statlot-649

# Create venv
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install deps (torch CPU only to save space)
pip install --quiet numpy scikit-learn imbalanced-learn xgboost optuna joblib pandas boto3
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet shap

echo "✅ Dependencies installed"

# Download draws from S3 (using EC2 IAM role - no keys needed)
aws s3 cp s3://statlot-649/data/draws.json statlot/data/draws.json --region ap-southeast-1 2>/dev/null || \
    aws s3 cp s3://statlot-649/data/draws.json ~/statlot-649/data/draws.json --region ap-southeast-1

# Generate CSV
python3 scripts/gen_draws_csv.py

echo "✅ Draws ready"
echo "=== Starting backtest ==="

# Run walk-forward backtest
export S3_BUCKET=statlot-649
export S3_PREFIX=statlot-649
nohup python3 backtest_v2.py 2>&1 | tee ~/backtest_run.log &
echo "Backtest running in background. PID=$!"
echo "Tail logs: tail -f ~/backtest_run.log"
