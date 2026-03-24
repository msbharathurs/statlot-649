#!/usr/bin/env python3
"""Generate draws CSV from statlot DB entities via base44 API"""
import os, csv, sys

# Reads draws.json if it exists (pre-uploaded) or generates from embedded data
import json

draws_file = os.path.join(os.path.dirname(__file__), "draws.json")
if not os.path.exists(draws_file):
    print("ERROR: draws.json not found. Upload it to EC2 first.")
    sys.exit(1)

with open(draws_file) as f:
    draws = json.load(f)

# Sort by draw_number
draws.sort(key=lambda x: x["draw_number"])
# Deduplicate
seen = set()
unique = []
for d in draws:
    if d["draw_number"] not in seen:
        seen.add(d["draw_number"])
        unique.append(d)

print(f"Total unique draws: {len(unique)}")
print(f"Draw range: {unique[0]['draw_number']} - {unique[-1]['draw_number']}")

# Write CSV matching backtest_v2.py format
with open("draws.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Draw","1st Number","2nd Number","3","4","5","6th Number","Additional Number"])
    for d in unique:
        nums = sorted([d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]])
        writer.writerow([d["draw_number"], nums[0],nums[1],nums[2],nums[3],nums[4],nums[5], d.get("additional","")])

print("✅ draws.csv written")
