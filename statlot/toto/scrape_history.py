#!/usr/bin/env python3
"""
scrape_history.py — Fill the TOTO draw history gap.

Scrapes draws #2341 to #4168 from Singapore Pools ASPX individual draw pages.
URL pattern: https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx?sppl=<base64(DrawNumber=NNNN)>

Rules:
- Reads existing sp_historical_draws.json to find what we already have
- Identifies missing draw numbers
- Scrapes missing draws one by one (with polite delays)
- Backs up sp_historical_draws.json before writing
- Flags corrupted dates — does NOT silently fix them
- Rebuilds toto_draws DuckDB table from scratch when done

Usage:
    python3 scrape_history.py [--start 2341] [--end 4168] [--dry-run]

Evidence command (paste output after run):
    python3 -c "
    import duckdb
    con = duckdb.connect('/home/ubuntu/statlot-649/statlot_toto.duckdb')
    n = con.execute('SELECT COUNT(*) FROM toto_draws').fetchone()[0]
    first = con.execute('SELECT draw_no, draw_date FROM toto_draws ORDER BY draw_no LIMIT 3').fetchall()
    last = con.execute('SELECT draw_no, draw_date FROM toto_draws ORDER BY draw_no DESC LIMIT 3').fetchall()
    print(f'Total: {n}')
    print(f'First 3: {first}')
    print(f'Last 3: {last}')
    con.close()
    "
"""

import argparse
import base64
import json
import os
import re
import shutil
import time
from datetime import datetime, date
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import duckdb

# ── Canonical paths ────────────────────────────────────────────────────────────
DB_PATH = Path('/home/ubuntu/statlot-649/statlot_toto.duckdb')
JSON_PATH = Path('/home/ubuntu/statlot-649/statlot/sp_historical_draws.json')
BASE_URL = 'https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
}
SLEEP_BETWEEN = 1.0   # seconds between requests — polite to SP
SLEEP_ON_ERROR = 5.0  # seconds to wait on HTTP error before retry

# ── Date parsing ───────────────────────────────────────────────────────────────
DATE_FORMATS = [
    '%d %b %Y',       # "02 Apr 2026"
    '%a, %d %b %Y',   # "Thu, 02 Apr 2026"
    '%d/%m/%Y',
    '%Y-%m-%d',
    '%d-%m-%Y',
]

def parse_date(text: str) -> tuple[date | None, bool]:
    """Returns (date_obj, is_corrupted). Corrupted = year < 1985 or year > 2030."""
    text = text.strip()
    for fmt in DATE_FORMATS:
        try:
            d = datetime.strptime(text, fmt).date()
            corrupted = d.year < 1985 or d.year > 2030
            return d, corrupted
        except ValueError:
            continue
    return None, True


def scrape_draw(draw_no: int) -> dict | None:
    """Scrape a single draw from the SP ASPX page. Returns dict or None on failure."""
    payload = f'DrawNumber={draw_no}'
    b64 = base64.b64encode(payload.encode()).decode()
    url = f'{BASE_URL}?sppl={b64}'

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            if r.status_code != 200:
                print(f'  [WARN] Draw {draw_no}: HTTP {r.status_code}, attempt {attempt+1}')
                time.sleep(SLEEP_ON_ERROR)
                continue

            soup = BeautifulSoup(r.text, 'html.parser')

            # Draw number — verify we got the right draw
            page_draw_nums = re.findall(r'Draw No\.?\s*(\d+)', r.text)
            if not page_draw_nums or int(page_draw_nums[0]) != draw_no:
                print(f'  [WARN] Draw {draw_no}: page returned draw {page_draw_nums} — skipping')
                return None

            # Winning numbers: class win1..win6
            numbers = []
            for i in range(1, 7):
                cells = soup.find_all(class_=f'win{i}')
                for cell in cells:
                    txt = cell.get_text(strip=True)
                    if txt.isdigit() and 1 <= int(txt) <= 49:
                        numbers.append(int(txt))
                        break

            if len(numbers) != 6:
                # Fallback: all td with class win*
                win_cells = soup.find_all('td', class_=re.compile(r'^win\d$'))
                numbers = [int(c.get_text(strip=True)) for c in win_cells
                           if c.get_text(strip=True).isdigit() and 1 <= int(c.get_text(strip=True)) <= 49]

            if len(numbers) < 6:
                print(f'  [WARN] Draw {draw_no}: only found {len(numbers)} numbers — skipping')
                return None

            numbers = numbers[:6]

            # Additional number
            additional = None
            add_cells = soup.find_all('td', class_='addNum')
            if not add_cells:
                add_cells = soup.find_all('td', class_='additional')
            if not add_cells:
                # Look for "Additional Number" label in page text
                add_match = re.search(r'Additional.*?(\d{1,2})', r.text[r.text.find('Additional'):r.text.find('Additional')+200] if 'Additional' in r.text else '')
                if add_match:
                    val = int(add_match.group(1))
                    if 1 <= val <= 49:
                        additional = val
            else:
                txt = add_cells[0].get_text(strip=True)
                if txt.isdigit():
                    additional = int(txt)

            # Draw date — find in header table
            date_obj = None
            date_corrupted = False
            date_cells = soup.find_all('th', class_='drawDate')
            if date_cells:
                date_obj, date_corrupted = parse_date(date_cells[0].get_text(strip=True))
            else:
                # Fallback: find date pattern in text
                date_match = re.search(r'(\w+,\s+\d{1,2}\s+\w+\s+\d{4})', r.text)
                if date_match:
                    date_obj, date_corrupted = parse_date(date_match.group(1))

            result = {
                'draw_number': draw_no,
                'draw_date': date_obj.isoformat() if date_obj else None,
                'n1': numbers[0], 'n2': numbers[1], 'n3': numbers[2],
                'n4': numbers[3], 'n5': numbers[4], 'n6': numbers[5],
                'additional': additional,
                'source': 'sp_aspx',
                'date_corrupted': date_corrupted,
            }

            if date_corrupted:
                print(f'  [DATE_CORRUPT] Draw {draw_no}: raw date text possibly wrong — flagged')

            return result

        except Exception as e:
            print(f'  [ERROR] Draw {draw_no} attempt {attempt+1}: {e}')
            time.sleep(SLEEP_ON_ERROR)

    return None


def load_existing_json() -> dict:
    """Load existing JSON. Returns dict keyed by draw_number."""
    if not JSON_PATH.exists():
        return {}
    with open(JSON_PATH) as f:
        data = json.load(f)
    return {d['draw_number']: d for d in data}


def backup_json():
    bak = str(JSON_PATH) + '.bak'
    shutil.copy2(JSON_PATH, bak)
    print(f'Backed up JSON to {bak}')


def rebuild_toto_draws_table(all_draws: list[dict]):
    """Rebuild toto_draws from scratch using the complete draw list."""
    print(f'\nRebuilding toto_draws table with {len(all_draws)} draws...')
    con = duckdb.connect(str(DB_PATH))

    # Preserve existing toto_predictions and toto_results — only rebuild toto_draws
    con.execute('DROP TABLE IF EXISTS toto_draws')
    con.execute('''
        CREATE TABLE toto_draws (
            draw_no         INTEGER PRIMARY KEY,
            draw_date       DATE,
            n1 INTEGER, n2 INTEGER, n3 INTEGER,
            n4 INTEGER, n5 INTEGER, n6 INTEGER,
            additional      INTEGER,
            game_format     VARCHAR,
            source          VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    inserted = 0
    skipped = 0
    for d in sorted(all_draws, key=lambda x: x['draw_number']):
        draw_no = d['draw_number']
        draw_date = d.get('draw_date')
        n1 = d.get('n1'); n2 = d.get('n2'); n3 = d.get('n3')
        n4 = d.get('n4'); n5 = d.get('n5'); n6 = d.get('n6')
        additional = d.get('additional')
        source = d.get('source', 'unknown')
        game_format = d.get('format', None)

        # Skip if missing critical fields
        if None in (n1, n2, n3, n4, n5, n6):
            skipped += 1
            continue

        # Parse date
        date_val = None
        if draw_date:
            date_obj, _ = parse_date(str(draw_date))
            if date_obj and date_obj.year >= 1985:
                date_val = date_obj.isoformat()

        try:
            con.execute('''
                INSERT OR IGNORE INTO toto_draws
                (draw_no, draw_date, n1, n2, n3, n4, n5, n6, additional, game_format, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [draw_no, date_val, n1, n2, n3, n4, n5, n6, additional, game_format, source])
            inserted += 1
        except Exception as e:
            print(f'  [DB ERROR] draw {draw_no}: {e}')
            skipped += 1

    con.close()
    print(f'toto_draws rebuilt: {inserted} inserted, {skipped} skipped')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=2341, help='First draw to fetch')
    parser.add_argument('--end', type=int, default=4168, help='Last draw to fetch')
    parser.add_argument('--dry-run', action='store_true', help='Just report gaps, no scraping')
    parser.add_argument('--rebuild-only', action='store_true', help='Rebuild DB from existing JSON only')
    args = parser.parse_args()

    print('='*60)
    print('TOTO HISTORY SCRAPER')
    print(f'Target range: #{args.start} to #{args.end}')
    print('='*60)

    # Load existing data
    existing = load_existing_json()
    existing_nums = set(existing.keys())
    target_nums = set(range(args.start, args.end + 1))
    missing = sorted(target_nums - existing_nums)

    print(f'Existing draws in JSON: {len(existing_nums)}')
    print(f'Target range size: {len(target_nums)}')
    print(f'Missing draws: {len(missing)}')
    if missing:
        print(f'First missing: #{missing[0]}, Last missing: #{missing[-1]}')

    # Report corrupted dates in existing data
    corrupted = [(k, v.get('draw_date')) for k, v in existing.items()
                 if v.get('draw_date') and ('0001' in str(v.get('draw_date')) or '2026' in str(v.get('draw_date')) and k < 100)]
    if corrupted:
        print(f'\n[WARN] Corrupted dates found in existing data: {len(corrupted)} draws flagged')
        for k, d in corrupted[:5]:
            print(f'  Draw #{k}: date={d}')

    if args.dry_run:
        print('\nDRY RUN — no scraping performed')
        return

    if args.rebuild_only:
        print('\nRebuild-only mode — rebuilding toto_draws from existing JSON...')
        rebuild_toto_draws_table(list(existing.values()))
        return

    if not missing:
        print('\nNo missing draws in target range. Nothing to fetch.')
        print('Rebuilding toto_draws table from existing JSON...')
        rebuild_toto_draws_table(list(existing.values()))
        return

    # Backup JSON before modifying
    if JSON_PATH.exists():
        backup_json()

    # Scrape missing draws
    fetched = {}
    failed = []
    print(f'\nFetching {len(missing)} missing draws...')

    for i, draw_no in enumerate(missing):
        if i % 50 == 0:
            print(f'Progress: {i}/{len(missing)} — draw #{draw_no}')

        result = scrape_draw(draw_no)
        if result:
            fetched[draw_no] = result
        else:
            failed.append(draw_no)

        time.sleep(SLEEP_BETWEEN)

    print(f'\nFetch complete: {len(fetched)} fetched, {len(failed)} failed')
    if failed:
        print(f'Failed draw numbers: {failed[:20]}{"..." if len(failed)>20 else ""}')

    # Merge into existing data and save JSON
    all_draws = {**existing, **fetched}
    with open(JSON_PATH, 'w') as f:
        json.dump(list(all_draws.values()), f, indent=2, default=str)
    print(f'JSON saved: {len(all_draws)} total draws')

    # Rebuild toto_draws table
    rebuild_toto_draws_table(list(all_draws.values()))

    # Summary
    still_missing = sorted(target_nums - set(all_draws.keys()))
    print('\n' + '='*60)
    print('SUMMARY')
    print(f'Draws in target range fetched: {len(target_nums) - len(still_missing)}/{len(target_nums)}')
    if still_missing:
        print(f'Still missing: {len(still_missing)} draws')
        print(f'Remaining draw numbers: #{still_missing[0]}–#{still_missing[-1]}')
        print('Run again to retry failed draws.')
    else:
        print('ALL draws in target range fetched successfully.')
    print('='*60)


if __name__ == '__main__':
    main()
