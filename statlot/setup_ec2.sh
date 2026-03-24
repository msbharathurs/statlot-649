#!/bin/bash
# ============================================================
# StatLot 649 — EC2 Setup Script
# Run this ONCE on a fresh c5a.2xlarge (Ubuntu 22.04/24.04)
# Usage: bash setup_ec2.sh <S3_BUCKET_NAME>
# Example: bash setup_ec2.sh statlot-649
# ============================================================
set -e

S3_BUCKET=${1:-"statlot-649"}
REPO="https://github.com/msbharathurs/statlot-649.git"
PROJECT_DIR="$HOME/statlot-649"
VENV_DIR="$HOME/.statlot_env"
LOG_DIR="$PROJECT_DIR/logs"
DATA_DIR="$PROJECT_DIR/data"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         StatLot 649 — EC2 Setup (c5a.2xlarge)           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  S3 bucket  : $S3_BUCKET"
echo "  Project dir: $PROJECT_DIR"
echo "  venv       : $VENV_DIR"
echo ""

# ─── 1. System packages ───────────────────────────────────────
echo "━━━ [1/7] Installing system packages ━━━"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip git wget curl unzip \
    build-essential libopenblas-dev

# ─── 2. Clone / pull repo ─────────────────────────────────────
echo ""
echo "━━━ [2/7] Cloning repository ━━━"
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "  Repo exists — pulling latest..."
    cd "$PROJECT_DIR" && git pull
else
    git clone "$REPO" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# ─── 3. Create venv ───────────────────────────────────────────
echo ""
echo "━━━ [3/7] Creating Python venv ━━━"
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ─── 4. Install ML libraries ──────────────────────────────────
echo ""
echo "━━━ [4/7] Installing ML libraries (takes ~5 min) ━━━"
pip install -q \
    numpy>=1.24 \
    pandas>=2.0 \
    scikit-learn>=1.3 \
    imbalanced-learn>=0.11 \
    xgboost>=2.0 \
    optuna>=3.4 \
    joblib>=1.3 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    boto3>=1.34

echo "  ✅ Libraries installed"

# ─── 5. Create data + log dirs ────────────────────────────────
echo ""
echo "━━━ [5/7] Creating directories ━━━"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$PROJECT_DIR/results" "$PROJECT_DIR/saved_models"
echo "  ✅ data/ logs/ results/ saved_models/ ready"

# ─── 6. Download draws.csv from S3 ───────────────────────────
echo ""
echo "━━━ [6/7] Downloading draws.csv from S3 ━━━"
if aws s3 cp "s3://${S3_BUCKET}/statlot-649/data/draws.csv" "$DATA_DIR/draws.csv" 2>/dev/null; then
    NLINES=$(wc -l < "$DATA_DIR/draws.csv")
    echo "  ✅ draws.csv downloaded ($NLINES lines)"
else
    echo "  ⚠️  draws.csv not found in S3 yet."
    echo "  → Upload it manually: aws s3 cp draws.csv s3://${S3_BUCKET}/statlot-649/data/draws.csv"
    echo "  → Or copy directly:   scp draws.csv ubuntu@<EC2_IP>:~/statlot-649/data/"
fi

# ─── 7. Write run script ──────────────────────────────────────
echo ""
echo "━━━ [7/7] Writing run_backtest.sh ━━━"
cat > "$PROJECT_DIR/run_backtest.sh" << RUNEOF
#!/bin/bash
# Run the full 3-iteration walk-forward backtest
source $VENV_DIR/bin/activate
cd $PROJECT_DIR
export S3_BUCKET=$S3_BUCKET
export S3_PREFIX=statlot-649
LOGFILE=$LOG_DIR/backtest_\$(date +%Y%m%d_%H%M%S).log
echo "Starting backtest — log: \$LOGFILE"
python backtest_v2.py data/draws.csv 2>&1 | tee "\$LOGFILE"
echo ""
echo "Done. Uploading log to S3..."
aws s3 cp "\$LOGFILE" s3://${S3_BUCKET}/statlot-649/logs/
echo "✅ Complete"
RUNEOF
chmod +x "$PROJECT_DIR/run_backtest.sh"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE ✅                     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  To run the full backtest:                               ║"
echo "║    cd ~/statlot-649 && bash run_backtest.sh              ║"
echo "║                                                          ║"
echo "║  Estimated runtime: ~55 minutes (c5a.2xlarge)           ║"
echo "║  Results auto-saved to: s3://$S3_BUCKET/statlot-649/   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
