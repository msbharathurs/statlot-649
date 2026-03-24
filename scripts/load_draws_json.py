"""
Load draws from a JSON file into local SQLite DB.
Used when deploying to EC2: we ship the JSON, then load it.
Usage: python3 scripts/load_draws_json.py draws_clean.json
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db, init_db
from db.models import DrawRecord


def load(path: str):
    init_db()
    with open(path) as f:
        draws = json.load(f)

    db = next(get_db())
    inserted = skipped = 0

    for d in draws:
        existing = db.query(DrawRecord).filter_by(draw_number=d["draw_number"]).first()
        if existing:
            skipped += 1
            continue
        nums = sorted([d["n1"], d["n2"], d["n3"], d["n4"], d["n5"], d["n6"]])
        rec = DrawRecord(
            draw_number=d["draw_number"],
            draw_date=d.get("draw_date"),
            n1=nums[0], n2=nums[1], n3=nums[2],
            n4=nums[3], n5=nums[4], n6=nums[5],
            additional=d.get("additional"),
        )
        db.add(rec)
        inserted += 1

    db.commit()
    print(f"✅ Inserted: {inserted}, Skipped (existing): {skipped}, Total: {inserted+skipped}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "draws_clean.json"
    load(path)
