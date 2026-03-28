#!/bin/bash
# toto/post_draw_pipeline.sh — step-tracked TOTO pipeline
# Called by agent after EC2 is upgraded

set -e
LOG=~/statlot-649/logs/toto_post_draw_pipeline.log
STATUS=~/statlot-649/logs/toto_pipeline_status.json
mkdir -p ~/statlot-649/logs

write_status() {
    echo "{\"step\": \"$1\", \"status\": \"$2\", \"ts\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > $STATUS
}

exec >> $LOG 2>&1
echo ""
echo "======================================================"
echo "TOTO POST-DRAW PIPELINE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

cd ~/statlot-649/statlot
source venv/bin/activate

write_status "check_wins" "running"
echo "[1] Checking wins against last prediction..."
python3 -m toto.check_wins
write_status "check_wins" "done"

write_status "retrain" "running"
echo "[2] Full retrain + generate prediction..."
python3 -m toto.retrain_and_predict
write_status "retrain" "done"

write_status "all" "complete"
echo "[DONE] TOTO Pipeline complete — $(date '+%Y-%m-%d %H:%M:%S')"
