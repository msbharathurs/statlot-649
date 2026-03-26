"""
verify_draw.py — Record actual draw result + score our predictions
Usage: python3 verify_draw.py --draw_number 4168 --nums 5 12 23 34 41 47 --additional 7

What it does:
1. Loads final_prediction.json
2. Scores each of our 10 tickets against the actual draw
3. Updates the Draw entity in Base44 (or prints result for manual entry)
4. Prints match summary
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def score_ticket(ticket, actual_nums, actual_add):
    t = set(ticket)
    a = set(actual_nums)
    main_match = len(t & a)
    bonus_match = actual_add in t
    return main_match, bonus_match

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw_number", type=int, required=True)
    parser.add_argument("--nums", type=int, nargs=6, required=True)
    parser.add_argument("--additional", type=int, required=True)
    args = parser.parse_args()

    actual_nums = sorted(args.nums)
    actual_add  = args.additional

    # Load prediction
    pred_path = os.path.join(RESULTS_DIR, "final_prediction.json")
    if not os.path.exists(pred_path):
        print("ERROR: final_prediction.json not found — run predict_final.py first")
        return

    with open(pred_path) as f:
        pred = json.load(f)

    print(f"\n{'='*60}")
    print(f"  DRAW {args.draw_number} RESULT VERIFICATION")
    print(f"{'='*60}")
    print(f"  Actual draw: {actual_nums}  +{actual_add}")
    print(f"  Prediction was for: {pred['draw_label']}")
    print()

    best_match = 0
    best_ticket = None
    results = []

    for i, ticket in enumerate(pred["tickets"], 1):
        main_m, bonus_m = score_ticket(ticket, actual_nums, actual_add)
        label = f"T{i:02d}: {ticket}"
        if main_m >= 3:
            tag = f"  ← {main_m} MATCH" + (" + BONUS" if bonus_m else "")
        elif main_m == 2 and bonus_m:
            tag = "  ← 2 + BONUS"
        else:
            tag = ""
        print(f"  {label}  →  {main_m}match {'★' if bonus_m else ' '}{tag}")
        if main_m > best_match or (main_m == best_match and bonus_m):
            best_match = main_m
            best_ticket = ticket
        results.append({"ticket": ticket, "main_match": main_m, "bonus_match": bonus_m})

    three_plus = sum(1 for r in results if r["main_match"] >= 3 or (r["main_match"] >= 2 and r["bonus_match"]))
    print(f"\n  Best ticket: {best_ticket}  ({best_match} match)")
    print(f"  Tickets with 3+ or 2+bonus: {three_plus}/10")
    print(f"{'='*60}")

    # Save result
    out = {
        "draw_number": args.draw_number,
        "actual_nums": actual_nums,
        "actual_additional": actual_add,
        "tickets_scored": results,
        "best_match": best_match,
        "three_plus_count": three_plus,
        "prediction_label": pred["draw_label"],
    }
    out_path = os.path.join(RESULTS_DIR, f"draw_{args.draw_number}_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}")

if __name__ == "__main__":
    main()
