"""
scrape_sp_historical.py — Scrape TOTO draws 1–2340 from Singapore Pools
Uses: /en/product/sr/Pages/toto_results.aspx?sppl=base64("DrawNumber=XXXX")

Format eras (ball pool size):
  Draw   1 – ~299 : 5/49  (5 winning numbers, additional may exist)
  Draw ~300 – ~799 : 6/42
  Draw ~800 – 2340 : 6/45
  Draw 2341+        : 6/49  (handled by Lottolyzer scraper)

Outputs: sp_historical_draws.json

Run: python3 scrape_sp_historical.py [start_draw] [end_draw]
     python3 scrape_sp_historical.py 1 2340
"""
import re, time, json, base64, sys, requests
from datetime import datetime

BASE_URL = "https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx?sppl={sppl}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
OUTPUT = "sp_historical_draws.json"
DELAY = 0.4   # seconds between requests
BATCH_SAVE = 100  # save progress every N draws


def get_format(draw_no):
    if draw_no < 300:   return "5/49"
    if draw_no < 800:   return "6/42"
    if draw_no <= 2340: return "6/45"
    return "6/49"


def parse_result(html, draw_no):
    """Parse HTML from SP single-result page. Returns dict or None."""
    # Strip scripts/styles
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)

    # Verify we got the right draw
    if f"Draw No. {draw_no}" not in html:
        return None

    # Extract draw date
    date_m = re.search(r'(\d{2})\s+(\w+)\s+(\d{4})', html)
    draw_date = None
    if date_m:
        try:
            draw_date = datetime.strptime(date_m.group(0), "%d %B %Y").strftime("%Y-%m-%d")
        except ValueError:
            draw_date = date_m.group(0)

    # Extract winning numbers block
    clean = re.sub(r'<[^>]+>', ' ', html)
    clean = ' '.join(clean.split())

    # Pattern: "Winning Numbers 3 17 19 26 35 38 Additional Number 42"
    nums_m = re.search(r'Winning Numbers\s+([\d\s]+?)\s+Additional Number\s+(\d+)', clean)
    if not nums_m:
        # Try without additional (old 5/49 format)
        nums_m2 = re.search(r'Winning Numbers\s+([\d\s]+?)(?:\s+Winning Shares|\s+Group\s+1|$)', clean)
        if not nums_m2:
            return None
        nums_str = nums_m2.group(1).strip()
        additional = None
    else:
        nums_str = nums_m.group(1).strip()
        additional = int(nums_m.group(2))

    nums = sorted([int(n) for n in nums_str.split() if n.isdigit()])
    if len(nums) < 4:  # sanity check
        return None

    draw_fmt = get_format(draw_no)

    result = {
        "draw_number": draw_no,
        "draw_date": draw_date,
        "additional": additional,
        "format": draw_fmt,
        "source": "sp_historical"
    }

    # Fill n1–n6 (or n1–n5 for 5/49)
    for i, n in enumerate(nums[:6], 1):
        result[f"n{i}"] = n
    # Pad missing (5/49 → n6=None)
    for i in range(len(nums) + 1, 7):
        result[f"n{i}"] = None

    return result


def scrape_range(start=1, end=2340):
    session = requests.Session()
    session.headers.update(HEADERS)

    # Load existing progress if any
    try:
        with open(OUTPUT) as f:
            existing = json.load(f)
        seen = {d["draw_number"]: d for d in existing}
        print(f"Resuming from {len(seen)} existing draws")
    except FileNotFoundError:
        seen = {}

    errors = []
    skipped = 0

    total = end - start + 1
    for draw_no in range(start, end + 1):
        if draw_no in seen:
            skipped += 1
            continue

        sppl = base64.b64encode(f"DrawNumber={draw_no}".encode()).decode()
        url = BASE_URL.format(sppl=sppl)

        result = None
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=15)
                resp.raise_for_status()
                result = parse_result(resp.text, draw_no)
                break
            except Exception as e:
                print(f"  Draw {draw_no} attempt {attempt+1} failed: {e}")
                time.sleep(2)

        if result:
            seen[draw_no] = result
        else:
            errors.append(draw_no)
            print(f"  PARSE FAIL draw {draw_no}")

        done = draw_no - start + 1 - skipped
        if done % 50 == 0 or draw_no == end:
            pct = (draw_no - start + 1) / total * 100
            print(f"  [{pct:5.1f}%] draw {draw_no}/{end} | scraped: {len(seen)} | errors: {len(errors)}")
            # Save progress
            sorted_draws = sorted(seen.values(), key=lambda d: d["draw_number"])
            with open(OUTPUT, "w") as f:
                json.dump(sorted_draws, f, indent=2)

        time.sleep(DELAY)

    # Final save
    sorted_draws = sorted(seen.values(), key=lambda d: d["draw_number"])
    with open(OUTPUT, "w") as f:
        json.dump(sorted_draws, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Total draws saved: {len(sorted_draws)}")
    print(f"Errors (parse fail): {len(errors)} — {errors[:20]}")
    if sorted_draws:
        print(f"Range: {sorted_draws[0]['draw_number']} → {sorted_draws[-1]['draw_number']}")
    return sorted_draws


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end   = int(sys.argv[2]) if len(sys.argv) > 2 else 2340
    print(f"=== SP Historical TOTO Scraper (draws {start}–{end}) ===\n")
    draws = scrape_range(start, end)
