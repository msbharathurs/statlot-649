"""
scrape_lottolyzer.py — Scrape TOTO draws 2341–4167 from Lottolyzer
Clean structured table, 37 pages × ~50 draws.
Outputs: lottolyzer_draws.json

Run: python3 scrape_lottolyzer.py
"""
import re, time, json, requests
from datetime import datetime

BASE_URL = "https://en.lottolyzer.com/history/singapore/toto/page/{page}/per-page/50/summary-view"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
OUTPUT = "lottolyzer_draws.json"
TOTAL_PAGES = 37
MIN_DRAW = 2341


def parse_page(html):
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    draws = []
    for row in rows:
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
        clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        clean = [c for c in clean if c]
        # Must have draw_no (integer), date, numbers
        if len(clean) < 4 or not clean[0].isdigit():
            continue
        draw_no = int(clean[0])
        if draw_no < MIN_DRAW:
            continue
        try:
            draw_date = datetime.strptime(clean[1], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            draw_date = clean[1]
        # Numbers: "4,25,28,33,43,48"
        nums_raw = clean[2].split(",")
        nums = sorted([int(n.strip()) for n in nums_raw if n.strip().isdigit()])
        # Additional number
        try:
            additional = int(clean[3])
        except (ValueError, IndexError):
            additional = None
        if len(nums) != 6:
            print(f"  SKIP draw {draw_no} — unexpected nums: {nums_raw}")
            continue
        draws.append({
            "draw_number": draw_no,
            "draw_date": draw_date,
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "additional": additional,
            "format": "6/49",
            "source": "lottolyzer"
        })
    return draws


def scrape_all():
    all_draws = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for page in range(1, TOTAL_PAGES + 1):
        url = BASE_URL.format(page=page)
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=15)
                resp.raise_for_status()
                break
            except Exception as e:
                print(f"  Page {page} attempt {attempt+1} failed: {e}")
                time.sleep(2)
        else:
            print(f"  GIVING UP on page {page}")
            continue

        draws = parse_page(resp.text)
        all_draws.extend(draws)
        print(f"  Page {page:2d}/{TOTAL_PAGES}: {len(draws)} draws | total so far: {len(all_draws)}")
        time.sleep(0.5)  # polite delay

    # Sort by draw_number ascending
    all_draws.sort(key=lambda d: d["draw_number"])

    # Dedup
    seen = {}
    for d in all_draws:
        seen[d["draw_number"]] = d
    deduped = sorted(seen.values(), key=lambda d: d["draw_number"])

    print(f"\nTotal unique draws: {len(deduped)}")
    if deduped:
        print(f"Range: {deduped[0]['draw_number']} → {deduped[-1]['draw_number']}")
        missing = [n for n in range(MIN_DRAW, deduped[-1]['draw_number']+1) if n not in seen]
        print(f"Missing draw numbers: {missing[:20]}{'...' if len(missing)>20 else ''} ({len(missing)} total)")

    with open(OUTPUT, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"Saved to {OUTPUT}")
    return deduped


if __name__ == "__main__":
    print("=== Lottolyzer TOTO Scraper (draws 2341–4167) ===\n")
    draws = scrape_all()
    print("\nDone!")
