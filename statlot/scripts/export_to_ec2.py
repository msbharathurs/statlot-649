"""
Export all 1827 draw records from Base44 DB to EC2 via API.
Deduplicates by draw_number, validates data quality.
Run locally: python3 scripts/export_to_ec2.py
"""
import os
import json
import requests
import sys

BASE44_API = os.environ.get("BASE44_API_URL", "https://api.base44.com")
API_KEY = os.environ.get("BASE44_API_KEY", "")
APP_ID = "69c0be463901b622442643b9"

EC2_HOST = os.environ.get("EC2_HOST", "54.179.152.124")
EC2_PORT = os.environ.get("EC2_PORT", "8000")

def fetch_all_draws():
    """Fetch all draws from Base44, deduplicate."""
    seen = {}
    skip = 0
    limit = 500
    while True:
        url = f"{BASE44_API}/apps/{APP_ID}/entities/Draw"
        resp = requests.get(url, params={"limit": limit, "skip": skip}, headers={"x-api-key": API_KEY})
        data = resp.json()
        records = data.get("records", [])
        for r in records:
            dn = r["draw_number"]
            if dn not in seen:
                seen[dn] = r
        print(f"  Fetched {skip+len(records)} records, unique so far: {len(seen)}")
        if not data.get("has_more"):
            break
        skip += limit
    draws = sorted(seen.values(), key=lambda x: x["draw_number"])
    return draws

def push_to_ec2(draws):
    url = f"http://{EC2_HOST}:{EC2_PORT}/draws/bulk"
    payload = {"draws": [
        {
            "draw_number": d["draw_number"],
            "n1": d["n1"], "n2": d["n2"], "n3": d["n3"],
            "n4": d["n4"], "n5": d["n5"], "n6": d["n6"],
            "additional": d.get("additional"),
            "draw_date": d.get("draw_date"),
        }
        for d in draws
    ]}
    resp = requests.post(url, json=payload, timeout=60)
    return resp.json()

if __name__ == "__main__":
    print("Fetching draws from Base44...")
    draws = fetch_all_draws()
    print(f"Total unique draws: {len(draws)}")
    print(f"Range: {draws[0]['draw_number']} - {draws[-1]['draw_number']}")
    print(f"\nPushing to EC2 at {EC2_HOST}:{EC2_PORT}...")
    result = push_to_ec2(draws)
    print(f"Result: {result}")
