#!/usr/bin/env python3
"""
scrape_latest.py — Fetch the latest TOTO draw result.

URL: Static HTML archive (no JS rendering required)
  https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_top_draws_en.html

Output: ~/statlot-649/statlot/toto/latest_draw.json
Schema:
  {
    "draw_number": 4171,
    "draw_date": "2026-04-07",
    "numbers": [1, 7, 8, 23, 30, 33],
    "additional": 21,
    "fetched_at": "2026-04-07T19:05:00"
  }

Exit codes:
  0 = success, latest_draw.json written
  1 = fetch/parse error — full error printed to stderr, nothing written
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ----- CONSTANTS -----

STATIC_URL = (
    "https://www.singaporepools.com.sg"
    "/DataFileArchive/Lottery/Output/toto_result_top_draws_en.html"
)

OUTPUT_PATH = Path(__file__).parent / "latest_draw.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

DATE_FORMATS = [
    "%a, %d %b %Y",   # "Thu, 02 Apr 2026"
    "%d %b %Y",        # "02 Apr 2026"
]


def parse_date(raw: str) -> str:
    """Parse a date string like 'Thu, 02 Apr 2026' → '2026-04-02'."""
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: '{raw}'")


def fetch_latest() -> dict:
    """Fetch and parse the latest draw from the static HTML archive."""
    try:
        resp = requests.get(STATIC_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP fetch failed: {e}") from e

    soup = BeautifulSoup(resp.text, "html.parser")

    # The first <li> in the page is the latest draw.
    # Each draw block has:
    #   <th class='drawDate'>Thu, 02 Apr 2026</th>
    #   <th class='drawNumber'>Draw No. 4170</th>
    #   <td class='win1'>...</td> … <td class='win6'>…</td>
    #   <td class='additional'>…</td>

    draw_date_tag = soup.find("th", class_="drawDate")
    draw_num_tag  = soup.find("th", class_="drawNumber")

    if not draw_date_tag or not draw_num_tag:
        raise RuntimeError(
            "Could not find drawDate / drawNumber tags. "
            "Page structure may have changed."
        )

    # --- Draw number ---
    draw_num_text = draw_num_tag.get_text(strip=True)
    m = re.search(r"(\d+)", draw_num_text)
    if not m:
        raise RuntimeError(f"Cannot parse draw number from: '{draw_num_text}'")
    draw_number = int(m.group(1))

    # --- Draw date ---
    draw_date_raw = draw_date_tag.get_text(strip=True)
    draw_date = parse_date(draw_date_raw)

    # --- Winning numbers (win1–win6) ---
    numbers = []
    for cls in ["win1", "win2", "win3", "win4", "win5", "win6"]:
        tag = soup.find("td", class_=cls)
        if not tag:
            raise RuntimeError(f"Missing winning number cell: class='{cls}'")
        numbers.append(int(tag.get_text(strip=True)))

    # --- Additional number ---
    additional_tag = soup.find("td", class_="additional")
    if not additional_tag:
        raise RuntimeError("Missing additional number cell.")
    additional = int(additional_tag.get_text(strip=True))

    return {
        "draw_number": draw_number,
        "draw_date":   draw_date,
        "numbers":     sorted(numbers),
        "additional":  additional,
        "fetched_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    print(f"Fetching latest TOTO draw from static archive...")
    print(f"URL: {STATIC_URL}")

    try:
        result = fetch_latest()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSUCCESS")
    print(f"  Draw number : {result['draw_number']}")
    print(f"  Draw date   : {result['draw_date']}")
    print(f"  Numbers     : {result['numbers']}")
    print(f"  Additional  : {result['additional']}")
    print(f"  Fetched at  : {result['fetched_at']}")
    print(f"  Written to  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
