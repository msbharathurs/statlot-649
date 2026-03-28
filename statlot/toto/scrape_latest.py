"""
toto/scrape_latest.py — Poll Singapore Pools for latest TOTO result.
Polls every 2 min from 6:45pm until new draw found, then exits with result JSON.
"""
import requests, json, re, time, sys
from datetime import datetime, date
from bs4 import BeautifulSoup

TOTO_URL = "https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx"
RESULTS_FILE = "/home/ubuntu/statlot-649/statlot/toto/latest_draw.json"
STATUS_FILE  = "/home/ubuntu/statlot-649/logs/toto_pipeline_status.json"

def write_status(step, status):
    with open(STATUS_FILE, "w") as f:
        json.dump({"step": step, "status": status, "ts": datetime.utcnow().isoformat()}, f)

def fetch_latest_draw():
    """Scrape latest TOTO result page. Returns dict or None."""
    try:
        r = requests.get(TOTO_URL, timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StatLot/1.0)"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Draw number
        draw_no = None
        for tag in soup.find_all(["span", "div", "td"]):
            m = re.search(r'Draw\s*No[:\.]?\s*(\d{4,5})', tag.get_text(), re.I)
            if m:
                draw_no = int(m.group(1))
                break

        # Winning numbers
        nums = []
        for tag in soup.find_all(class_=re.compile(r'ball|winning|number', re.I)):
            t = tag.get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 49:
                nums.append(int(t))

        additional = None
        for tag in soup.find_all(class_=re.compile(r'additional|extra|bonus', re.I)):
            t = tag.get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 49:
                additional = int(t)
                break

        # Draw date
        draw_date = None
        for tag in soup.find_all(["span","div","td","p"]):
            txt = tag.get_text(strip=True)
            m = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', txt)
            if m:
                try:
                    draw_date = datetime.strptime(m.group(1), "%d %B %Y").date().isoformat()
                    break
                except: pass

        if draw_no and len(nums) >= 6:
            return {
                "draw_number": draw_no,
                "draw_date": draw_date,
                "numbers": sorted(nums[:6]),
                "additional": additional,
                "scraped_at": datetime.utcnow().isoformat()
            }
    except Exception as e:
        print(f"  Scrape error: {e}")
    return None

def get_last_known_draw():
    """Get highest draw number we already have."""
    try:
        import duckdb
        con = duckdb.connect("/home/ubuntu/statlot-649/statlot_toto.duckdb")
        row = con.execute("SELECT MAX(draw_number) FROM toto_draws").fetchone()
        return row[0] if row and row[0] else 0
    except:
        # Fallback: read from json history
        try:
            with open("/home/ubuntu/statlot-649/statlot/sp_historical_draws.json") as f:
                draws = json.load(f)
            return max(d["draw_number"] for d in draws)
        except:
            return 0

def poll_until_new(max_polls=30, interval=120):
    """Poll until a new draw appears. Returns draw dict."""
    last_known = get_last_known_draw()
    print(f"Last known draw: {last_known}")
    write_status("poll", "running")

    for i in range(max_polls):
        print(f"  [{i+1}/{max_polls}] Polling at {datetime.now().strftime('%H:%M:%S')}...")
        result = fetch_latest_draw()
        if result:
            print(f"  Found draw {result['draw_number']}: {result['numbers']} + {result['additional']}")
            if result["draw_number"] > last_known:
                print(f"  NEW draw detected!")
                with open(RESULTS_FILE, "w") as f:
                    json.dump(result, f, indent=2)
                write_status("poll", "done")
                return result
            else:
                print(f"  Draw {result['draw_number']} already known. Waiting...")
        else:
            print("  Could not parse result yet.")
        if i < max_polls - 1:
            time.sleep(interval)

    write_status("poll", "timeout")
    print("Polling timed out — no new draw found.")
    return None

if __name__ == "__main__":
    result = poll_until_new()
    if result:
        print(json.dumps(result, indent=2))
        sys.exit(0)
    else:
        sys.exit(1)
