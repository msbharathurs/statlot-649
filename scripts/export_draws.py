"""
One-time script: Export all draws from Base44 DB and seed PostgreSQL.
Run once on EC2 after deployment.
Usage: python scripts/export_draws.py --csv your_draws.csv
"""
import sys
import csv
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from db.database import init_db, SessionLocal
from db.models import DrawRecord


def parse_csv(path: str) -> list:
    draws = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            row = [r.strip() for r in row]
            if not row or not row[0].isdigit():
                continue
            dn = int(row[0])
            try:
                nums = sorted([int(row[1]),int(row[2]),int(row[3]),
                               int(row[4]),int(row[5]),int(row[6])])
            except (IndexError, ValueError):
                continue
            additional = int(row[7]) if len(row) > 7 and row[7].isdigit() else None
            draw_date  = row[8] if len(row) > 8 and row[8] else None
            draws.append({
                "draw_number": dn, "draw_date": draw_date,
                "n1": nums[0], "n2": nums[1], "n3": nums[2],
                "n4": nums[3], "n5": nums[4], "n6": nums[5],
                "additional": additional,
            })
    return draws


def seed_db(draws: list):
    init_db()
    db = SessionLocal()
    inserted, skipped = 0, 0
    for d in draws:
        if db.query(DrawRecord).filter_by(draw_number=d["draw_number"]).first():
            skipped += 1
            continue
        rec = DrawRecord(**{k: v for k, v in d.items() if k != "nums"})
        db.add(rec)
        inserted += 1
        if inserted % 100 == 0:
            db.commit()
            print(f"  {inserted} inserted...")
    db.commit()
    db.close()
    print(f"Done. Inserted: {inserted}, Skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to draws CSV file")
    args = parser.parse_args()
    draws = parse_csv(args.csv)
    print(f"Parsed {len(draws)} draws from {args.csv}")
    seed_db(draws)
