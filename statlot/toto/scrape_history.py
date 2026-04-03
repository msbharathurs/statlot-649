#!/usr/bin/env python3
"""
scrape_history.py — Fill the TOTO draw history gap.

Scrapes draws from Singapore Pools ASPX individual draw pages.
URL pattern:
  https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx?sppl=<base64(DrawNumber=NNNN)>

DATA QUALITY RULES (permanent — do not change without Bharath approval):
  - Draws #1–#2340:  NULL date or NULL numbers → silently excluded from toto_draws. No retry.
  - Draws #2341+:    NULL date or NULL numbers → must retry. If retry fails, log draw_no and
                     report to Bharath for manual entry. Never silently drop post-2341 data.

Usage:
    python3 scrape_history.py [--start 2341] [--end 4168] [--dry-run] [--rebuild-only]
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
DB_PATH   = Path('/home/ubuntu/statlot-649/statlot_toto.duckdb')
JSON_PATH = Path('/home/ubuntu/statlot-649/statlot/sp_historical_draws.json')
BASE_URL  = 'https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx'
HEADERS   = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
}

SLEEP_BETWEEN = 1.0   # seconds between requests
SLEEP_ON_ERROR = 5.0  # seconds to wait on HTTP error before retry
MAX_RETRIES    = 3    # attempts per draw

# ── Data quality thresholds ────────────────────────────────────────────────────
PRE_2341_CUTOFF  = 2340  # draws <= this: clean data only, broken silently excluded
POST_2341_CUTOFF = 2341  # draws >= this: no NULL tolerated, must retry or report


# ── Date parsing ───────────────────────────────────────────────────────────────
DATE_FORMATS = [
    '%d %b %Y',
    '%a, %d %b %Y',
    '%d/%m/%Y',
    '%Y-%m-%d',
    '%d-%m-%Y',
]

def parse_date(text: str) -> tuple:
    """Returns (date_obj | None, is_corrupted: bool)."""
    text = text.strip()
    for fmt in DATE_FORMATS:
        try:
            d = datetime.strptime(text, fmt).date()
            corrupted = d.year < 1985 or d.year > 2030
            return d, corrupted
        except ValueError:
            continue
    return None, True


# ── Scrape one draw ────────────────────────────────────────────────────────────
def scrape_draw(draw_no: int) -> dict | None:
    """Scrape a single draw. Returns dict with keys matching toto_draws schema, or None."""
    payload = f'DrawNumber={draw_no}'
    b64     = base64.b64encode(payload.encode()).decode()
    url     = f'{BASE_URL}?sppl={b64}'

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            if r.status_code != 200:
                print(f'  [WARN] Draw {draw_no}: HTTP {r.status_code}, attempt {attempt+1}')
                time.sleep(SLEEP_ON_ERROR)
                continue

            soup = BeautifulSoup(r.text, 'html.parser')

            # Verify correct draw was returned
            page_draw_nums = re.findall(r'Draw No\.?\s*(\d+)', r.text)
            if not page_draw_nums or int(page_draw_nums[0]) != draw_no:
                print(f'  [WARN] Draw {draw_no}: page returned {page_draw_nums}')
                time.sleep(SLEEP_ON_ERROR)
                continue

            # Winning numbers — class win1..win6
            numbers = []
            for i in range(1, 7):
                cells = soup.find_all(class_=f'win{i}')
                for cell in cells:
                    txt = cell.get_text(strip=True)
                    if txt.isdigit() and 1 <= int(txt) <= 49:
                        numbers.append(int(txt))
                        break

            if len(numbers) != 6:
                win_cells = soup.find_all('td', class_=re.compile(r'^win\d$'))
                numbers = [
                    int(c.get_text(strip=True)) for c in win_cells
                    if c.get_text(strip=True).isdigit() and 1 <= int(c.get_text(strip=True)) <= 49
                ]

            if len(numbers) < 6:
                print(f'  [WARN] Draw {draw_no}: only {len(numbers)} numbers parsed, attempt {attempt+1}')
                time.sleep(SLEEP_ON_ERROR)
                continue

            numbers = numbers[:6]

            # Additional number
            additional = None
            for cls in ('addNum', 'additional', 'addNumber'):
                add_cells = soup.find_all('td', class_=cls)
                if add_cells:
                    txt = add_cells[0].get_text(strip=True)
                    if txt.isdigit() and 1 <= int(txt) <= 49:
                        additional = int(txt)
                    break

            # Draw date
            date_obj      = None
            date_corrupted = False
            date_cells = soup.find_all('th', class_='drawDate')
            if date_cells:
                date_obj, date_corrupted = parse_date(date_cells[0].get_text(strip=True))
            else:
                m = re.search(r'(\w+,\s+\d{1,2}\s+\w+\s+\d{4})', r.text)
                if m:
                    date_obj, date_corrupted = parse_date(m.group(1))

            return {
                'draw_number': draw_no,
                'draw_date':   date_obj.isoformat() if date_obj else None,
                'n1': numbers[0], 'n2': numbers[1], 'n3': numbers[2],
                'n4': numbers[3], 'n5': numbers[4], 'n6': numbers[5],
                'additional':    additional,
                'source':        'sp_aspx',
                'date_corrupted': date_corrupted,
            }

        except Exception as e:
            print(f'  [ERROR] Draw {draw_no} attempt {attempt+1}: {e}')
            time.sleep(SLEEP_ON_ERROR)

    return None


# ── JSON helpers ───────────────────────────────────────────────────────────────
def load_existing_json() -> dict:
    if not JSON_PATH.exists():
        return {}
    with open(JSON_PATH) as f:
        data = json.load(f)
    return {d['draw_number']: d for d in data}


def backup_json():
    bak = str(JSON_PATH) + '.bak'
    shutil.copy2(JSON_PATH, bak)
    print(f'Backed up JSON → {bak}')


# ── Rebuild toto_draws with quality rules applied ─────────────────────────────
def rebuild_toto_draws_table(all_draws: list[dict]):
    """
    Rebuild toto_draws from the complete draw list.
    Data quality rules enforced here:
      - draw_no <= 2340: skip rows with NULL date or NULL numbers (silently)
      - draw_no >= 2341: skip rows with NULL date or NULL numbers, but PRINT a warning
    """
    print(f'\nRebuilding toto_draws ({len(all_draws)} candidates)...')
    con = duckdb.connect(str(DB_PATH))

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

    inserted       = 0
    skipped_pre    = 0   # pre-2341, bad data — silently excluded
    skipped_post   = []  # post-2341, bad data — must be reported

    for d in sorted(all_draws, key=lambda x: x['draw_number']):
        draw_no = d['draw_number']
        n1 = d.get('n1'); n2 = d.get('n2'); n3 = d.get('n3')
        n4 = d.get('n4'); n5 = d.get('n5'); n6 = d.get('n6')
        additional  = d.get('additional')
        source      = d.get('source', 'unknown')
        game_format = d.get('format', None)

        has_numbers = None not in (n1, n2, n3, n4, n5, n6)

        # Parse date — only accept years 1985–2030
        draw_date_raw = d.get('draw_date')
        date_val = None
        if draw_date_raw:
            date_obj, _ = parse_date(str(draw_date_raw))
            if date_obj and 1985 <= date_obj.year <= 2030:
                date_val = date_obj.isoformat()

        has_valid_date = date_val is not None

        # Apply quality rules
        if not has_numbers or not has_valid_date:
            if draw_no <= PRE_2341_CUTOFF:
                skipped_pre += 1
                continue  # silently
            else:
                skipped_post.append(draw_no)
                print(f'  [POST-2341 DATA MISSING] draw #{draw_no}: '
                      f'date={draw_date_raw}, n1={n1} — excluded, needs manual check')
                continue

        try:
            con.execute('''
                INSERT OR IGNORE INTO toto_draws
                (draw_no, draw_date, n1, n2, n3, n4, n5, n6, additional, game_format, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [draw_no, date_val, n1, n2, n3, n4, n5, n6, additional, game_format, source])
            inserted += 1
        except Exception as e:
            print(f'  [DB ERROR] draw #{draw_no}: {e}')
            if draw_no >= POST_2341_CUTOFF:
                skipped_post.append(draw_no)

    con.close()

    print(f'toto_draws rebuilt:')
    print(f'  Inserted: {inserted}')
    print(f'  Skipped (pre-2341, bad data, silently excluded): {skipped_pre}')
    if skipped_post:
        print(f'  POST-2341 DRAWS EXCLUDED (NEED MANUAL DATA): {skipped_post}')
    else:
        print(f'  Post-2341 draws with missing data: 0 — all clean')

    return skipped_post


# ── Clean existing toto_draws in-place (without full rebuild) ─────────────────
def clean_existing_table():
    """
    Apply quality rules to the current toto_draws table in-place:
    - Delete pre-2341 rows with NULL date or NULL numbers (silently)
    - Find and report post-2341 rows with NULL date or NULL numbers
    """
    con = duckdb.connect(str(DB_PATH))

    # Count bad pre-2341 rows
    bad_pre_count = con.execute(
        "SELECT COUNT(*) FROM toto_draws WHERE draw_no <= 2340 AND (draw_date IS NULL OR n1 IS NULL)"
    ).fetchone()[0]

    # Delete them
    con.execute(
        "DELETE FROM toto_draws WHERE draw_no <= 2340 AND (draw_date IS NULL OR n1 IS NULL)"
    )

    # Find bad post-2341 rows
    bad_post = [r[0] for r in con.execute(
        "SELECT draw_no FROM toto_draws WHERE draw_no >= 2341 AND (draw_date IS NULL OR n1 IS NULL)"
    ).fetchall()]

    total    = con.execute("SELECT COUNT(*) FROM toto_draws").fetchone()[0]
    pre2341  = con.execute("SELECT COUNT(*) FROM toto_draws WHERE draw_no < 2341").fetchone()[0]
    post2341 = con.execute("SELECT COUNT(*) FROM toto_draws WHERE draw_no >= 2341").fetchone()[0]

    con.close()

    print(f"Deleted bad pre-2341 rows: {bad_pre_count}")
    print(f"Post-2341 draws needing retry: {bad_post}")
    print(f"Total draws remaining: {total}")
    print(f"  Pre-2341 rows: {pre2341}")
    print(f"  Post-2341 rows: {post2341}")

    return bad_post


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',        type=int, default=2341)
    parser.add_argument('--end',          type=int, default=4168)
    parser.add_argument('--dry-run',      action='store_true')
    parser.add_argument('--rebuild-only', action='store_true',
                        help='Skip scraping — rebuild DB from existing JSON only')
    parser.add_argument('--clean-only',   action='store_true',
                        help='Apply quality rules to current toto_draws table in-place, then exit')
    args = parser.parse_args()

    print('='*60)
    print('TOTO HISTORY SCRAPER')
    print(f'Target range: #{args.start} to #{args.end}')
    print('Data quality rule: pre-2341 NULLs silently excluded; post-2341 NULLs reported')
    print('='*60)

    if args.clean_only:
        print('\nClean-only mode — applying quality rules to existing toto_draws...')
        bad_post = clean_existing_table()
        if bad_post:
            print(f'\nACTION REQUIRED: {len(bad_post)} post-2341 draws have missing data.')
            print('Provide these draw numbers manually or run without --clean-only to retry scraping.')
        return

    existing = load_existing_json()
    existing_nums = set(existing.keys())
    target_nums   = set(range(args.start, args.end + 1))
    missing       = sorted(target_nums - existing_nums)

    print(f'Existing draws in JSON: {len(existing_nums)}')
    print(f'Target range size:      {len(target_nums)}')
    print(f'Missing draws:          {len(missing)}')

    if args.dry_run:
        if missing:
            print(f'First missing: #{missing[0]}, Last missing: #{missing[-1]}')
        print('\nDRY RUN — no scraping performed')
        return

    if args.rebuild_only or not missing:
        if not missing:
            print('No missing draws — rebuilding toto_draws from JSON only')
        bad_post = rebuild_toto_draws_table(list(existing.values()))
        if bad_post:
            print(f'\nACTION REQUIRED: provide data for post-2341 draws: {bad_post}')
        return

    if JSON_PATH.exists():
        backup_json()

    # Scrape missing draws
    fetched  = {}
    failed   = []
    post_failed = []

    print(f'\nFetching {len(missing)} missing draws...')
    for i, draw_no in enumerate(missing):
        if i % 50 == 0:
            print(f'Progress: {i}/{len(missing)} — draw #{draw_no}')

        result = scrape_draw(draw_no)
        if result:
            fetched[draw_no] = result
        else:
            failed.append(draw_no)
            if draw_no >= POST_2341_CUTOFF:
                post_failed.append(draw_no)

    print(f'\nFetch complete: {len(fetched)} fetched, {len(failed)} failed')

    # Merge and save JSON
    all_draws = {**existing, **fetched}
    with open(JSON_PATH, 'w') as f:
        json.dump(list(all_draws.values()), f, indent=2, default=str)
    print(f'JSON saved: {len(all_draws)} total draws')

    # Rebuild table with quality rules
    bad_post = rebuild_toto_draws_table(list(all_draws.values()))

    # Final report
    still_missing = sorted(target_nums - set(all_draws.keys()))
    all_bad_post  = sorted(set(post_failed + bad_post))

    print('\n' + '='*60)
    print('SUMMARY')
    print(f'Draws in target range fetched: {len(target_nums)-len(still_missing)}/{len(target_nums)}')

    if all_bad_post:
        print(f'\nACTION REQUIRED — post-2341 draws with missing data ({len(all_bad_post)}):')
        print(f'  {all_bad_post}')
        print('Provide these manually or they will remain absent from toto_draws.')
    else:
        print('Post-2341 data quality: CLEAN — all draws have valid date + numbers.')

    if still_missing:
        print(f'\nPre-2341 draws absent (corrupted source data, silently excluded): {len([x for x in still_missing if x <= 2340])}')

    print('='*60)


if __name__ == '__main__':
    main()
