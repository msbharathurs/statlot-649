#!/bin/bash
# StatLot 649 — EC2 t3.small deployment script
# Run as: bash deploy.sh
set -e

echo "=== StatLot 649 EC2 Deploy ==="

# 1. System deps
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv postgresql postgresql-contrib git

# 2. PostgreSQL setup
sudo systemctl start postgresql
sudo systemctl enable postgresql
sudo -u postgres psql -c "CREATE USER statlot WITH PASSWORD 'statlot649';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE statlot OWNER statlot;" 2>/dev/null || true
echo "✅ PostgreSQL ready"

# 3. Python venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Python dependencies installed"

# 4. Environment file
if [ ! -f .env ]; then
  cp .env.example .env
  sed -i 's/yourpassword/statlot649/' .env
  echo "✅ Created .env (edit API_HOST/PORT if needed)"
fi

# 5. Init DB tables
python3 -c "
import sys; sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv()
from db.database import init_db; init_db()
print('✅ DB tables created')
"

# 6. Systemd service
sudo tee /etc/systemd/system/statlot.service > /dev/null <<EOF
[Unit]
Description=StatLot 649 Engine
After=network.target postgresql.service

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable statlot
sudo systemctl restart statlot
echo "✅ Systemd service started"

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "API running at: http://$(curl -s ifconfig.me):8000"
echo "Health check:   curl http://localhost:8000/health"
echo ""
echo "Next steps:"
echo "  1. python scripts/export_draws.py --csv your_draws.csv"
echo "  2. curl -X POST http://localhost:8000/features/build"
echo "  3. python scripts/run_full_backtest.py"
