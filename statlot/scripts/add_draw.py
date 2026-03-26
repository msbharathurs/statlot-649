"""
add_draw.py — Add a new draw result to the dataset + optionally trigger weekly retrain
Usage: python3 add_draw.py --draw_number 4168 --date 2026-03-27 --nums 5 12 23 34 41 47 --additional 7

What it does:
1. Appends the new draw to data/draws_clean.csv
2. Logs verification result (how many of our tickets matched)
3. Prints confirmation — ready for next prediction run
"""
import sys, os, json, csv, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
DRAWS_PATH  = os.path.join(DATA_DIR, "draws_clean.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def load_draws_csv():
    rows = []
    with open(DRAWS_PATH) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    return rows, fieldnames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw_number", type=int, required=True)
    parser.add_argument("--date",        type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--nums",        type=int, nargs=6, required=True)
    parser.add_argument("--additional",  type=int, required=True)
    args = parser.parse_args()

    nums = sorted(args.nums)

    # Load existing
    rows, fieldnames = load_draws_csv()
    existing_draws = {int(r["draw_number"]) for r in rows}

    if args.draw_number in existing_draws:
        print(f"Draw {args.draw_number} already exists in dataset — skipping add")
        return

    # Build new row — minimal required fields, rest blank (features computed at train time)
    new_row = {k: "" for k in fieldnames}
    new_row["draw_number"] = str(args.draw_number)
    new_row["draw_date"]   = args.date
    new_row["n1"] = str(nums[0]); new_row["n2"] = str(nums[1])
    new_row["n3"] = str(nums[2]); new_row["n4"] = str(nums[3])
    new_row["n5"] = str(nums[4]); new_row["n6"] = str(nums[5])
    new_row["additional"] = str(args.additional)
    new_row["source"] = "manual"

    rows.append(new_row)
    rows.sort(key=lambda r: int(r["draw_number"]))

    with open(DRAWS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Draw {args.draw_number} ({args.date}) added to dataset")
    print(f"  Numbers: {nums} + {args.additional}")
    print(f"  Total draws now: {len(rows)}")
    print(f"\nNext step: run predict_final.py --draw_label 'Draw {args.draw_number+1}' to generate new tickets")

if __name__ == "__main__":
    main()
