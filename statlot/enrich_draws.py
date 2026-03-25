"""
enrich_draws.py — Build draws_enriched.csv from draws_clean.csv
Computes all 90 features for each draw using PRIOR draws only (zero leakage).
Run after every new draw is added.

Usage:
    python3 enrich_draws.py
    python3 enrich_draws.py --input draws_clean.csv --output draws_enriched.csv
    python3 enrich_draws.py --from-draw 4168   # only recompute from a specific draw

Output: draws_enriched.csv with columns:
    [raw cols] + [90 engine features] + [repeat_from_prev_1..10] + [repeat_count_cumul_1..10]
"""
import os, sys, csv, time, json
import argparse
import numpy as np
from collections import Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.features_v2 import (
    FEATURE_COLS, _build_history_context, build_features_batch,
    dual_grid_score, should_eliminate,
    rowA, colA, rowB, colB
)

RAW_COLS = ['draw_number','draw_date','n1','n2','n3','n4','n5','n6','additional','format','source']

def load_raw(path):
    draws = []
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nums = sorted([int(row[f'n{i}']) for i in range(1, 7)])
            add_raw = row.get('additional', '').strip()
            draws.append({
                'draw_number': int(row['draw_number']),
                'draw_date':   row['draw_date'],
                'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
                'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
                'additional': int(add_raw) if add_raw and add_raw.isdigit() else None,
                'format': row.get('format', '6/49'),
                'source': row.get('source', 'lottolyzer'),
                'nums': nums,
            })
    draws.sort(key=lambda d: d['draw_number'])
    return draws


def enrich_one(draw_dict, history):
    """Compute all enrichment columns for a single draw using prior history."""
    nums = draw_dict['nums']
    combo = tuple(nums)

    # -- Build features via batch path (single candidate)
    ctx = _build_history_context(history)
    feat_matrix = build_features_batch([combo], history)
    feat = {FEATURE_COLS[i]: float(feat_matrix[0, i]) for i in range(len(FEATURE_COLS))}

    # -- repeat_from_prev_1..10 (how many numbers overlap with draw N-lag)
    repeat_cols = {}
    for lag in range(1, 11):
        if len(history) >= lag:
            prev = set(history[-lag]['nums'])
            repeat_cols[f'repeat_from_prev_{lag}'] = len(set(nums) & prev)
        else:
            repeat_cols[f'repeat_from_prev_{lag}'] = 0

    # -- repeat_count_cumul_1..10 (unique numbers seen in last N draws)
    cumul_cols = {}
    seen = set()
    for lag in range(1, 11):
        if len(history) >= lag:
            seen |= set(history[-lag]['nums'])
        cumul_cols[f'repeat_count_prev_{lag}'] = len(set(nums) & seen)

    # -- Grid A row/col distribution (already in feat via rowA_1..6, colA_1..9)
    # -- Grid B row/col distribution (rowB_1..7, colB_1..7)

    # -- Sum z-score vs last 200 draws
    past_sums = [sum(d['nums']) for d in history[-200:]] if len(history) >= 5 else [147]
    sum_mean = float(np.mean(past_sums))
    sum_std  = float(np.std(past_sums)) if len(past_sums) > 1 else 1.0
    sum_zscore = (sum(nums) - sum_mean) / (sum_std + 1e-6)

    # -- Decade entropy
    decade_counts = [feat.get(f'decade_{d}', 0) for d in range(1, 6)]
    decade_probs  = [c/6.0 for c in decade_counts if c > 0]
    decade_entropy = float(-sum(p * np.log2(p) for p in decade_probs)) if decade_probs else 0.0

    # -- Position entropy (how uniformly spread 1-49)
    buckets = [(nums[i+1] - nums[i]) / 48.0 for i in range(5)]
    last_bucket = (49 - nums[-1]) / 48.0
    all_b = buckets + [last_bucket]
    all_b = [b for b in all_b if b > 0]
    position_entropy = float(-sum(b * np.log2(b) for b in all_b)) if all_b else 0.0

    # -- Triplet frequency (how many 3-combos appeared in last 50 draws)
    triplet_freq_50 = Counter()
    for d in history[-50:]:
        for tri in combinations(sorted(d['nums']), 3):
            triplet_freq_50[tri] += 1
    my_triplets = list(combinations(tuple(nums), 3))
    triplet_score = sum(triplet_freq_50.get(t, 0) for t in my_triplets)

    # -- is_eliminated flag
    is_elim = int(should_eliminate(nums))

    extra = {
        'sum_zscore':        round(sum_zscore, 4),
        'decade_entropy':    round(decade_entropy, 4),
        'position_entropy':  round(position_entropy, 4),
        'triplet_score':     triplet_score,
        'is_eliminated':     is_elim,
    }

    return {**feat, **repeat_cols, **cumul_cols, **extra}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='data/draws.csv')
    parser.add_argument('--output', default='data/draws_enriched.csv')
    parser.add_argument('--from-draw', type=int, default=0,
                        help='Only recompute enrichment from this draw_number onwards (incremental)')
    args = parser.parse_args()

    t0 = time.time()
    print(f'=== enrich_draws.py ===')
    print(f'Input:  {args.input}')
    print(f'Output: {args.output}')

    draws = load_raw(args.input)
    print(f'Loaded {len(draws)} draws ({draws[0]["draw_number"]} → {draws[-1]["draw_number"]})')

    # Determine enrichment columns order
    repeat_keys = [f'repeat_from_prev_{i}' for i in range(1, 11)]
    cumul_keys  = [f'repeat_count_prev_{i}' for i in range(1, 11)]
    extra_keys  = ['sum_zscore', 'decade_entropy', 'position_entropy', 'triplet_score', 'is_eliminated']
    enrich_cols = FEATURE_COLS + repeat_keys + cumul_keys + extra_keys

    # If incremental, load existing enriched CSV
    existing = {}
    if args.from_draw > 0 and os.path.exists(args.output):
        with open(args.output, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dn = int(row['draw_number'])
                if dn < args.from_draw:
                    existing[dn] = row

    all_cols = RAW_COLS + enrich_cols
    results = []
    skipped = 0

    for i, draw in enumerate(draws):
        dn = draw['draw_number']
        if args.from_draw > 0 and dn < args.from_draw and dn in existing:
            results.append(existing[dn])
            skipped += 1
            continue

        history = draws[:i]   # strictly prior draws only — zero leakage
        enriched = enrich_one(draw, history)
        row = {k: draw.get(k, '') for k in RAW_COLS}
        row['additional'] = draw['additional'] if draw['additional'] is not None else ''
        for k in enrich_cols:
            row[k] = enriched.get(k, 0)
        results.append(row)

        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(draws)}] {elapsed:.1f}s elapsed | ~{elapsed/(i+1-skipped)*(len(draws)-i-1):.0f}s remaining')

    # Write output
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    total = time.time() - t0
    print(f'\nDone! {len(results)} rows written to {args.output}')
    print(f'Total columns: {len(all_cols)} | Time: {total:.1f}s')
    print(f'Enriched cols: {len(enrich_cols)} ({len(FEATURE_COLS)} engine + {len(repeat_keys)} repeat + {len(cumul_keys)} cumul + {len(extra_keys)} extra)')
    print(f'\nColumn groups:')
    print(f'  Raw:            {len(RAW_COLS)} cols (draw_number, date, n1-n6, additional, format, source)')
    print(f'  Engine (90):    sum, mean, std, range, odd/even/low/high, decades, gaps, gridA/B, repeat_prev, hot/cold, freq, pairs, markov, bayes...')
    print(f'  Repeat lag 1-10: {len(repeat_keys)} cols (overlap with each of last 10 draws)')
    print(f'  Cumul 1-10:     {len(cumul_keys)} cols (unique number overlap with last N draws pool)')
    print(f'  Extra:          {len(extra_keys)} cols (sum_zscore, decade_entropy, position_entropy, triplet_score, is_eliminated)')


if __name__ == '__main__':
    main()
