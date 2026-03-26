"""
diversity_select_v2.py — Upgraded ticket selector with:

1. RELAXED overlap from MAX_SHARED=2 → 3 (allows tickets to concentrate numbers
   in the same zone, increasing P(one ticket hits 3+))
2. BONUS-AWARE selection: one reserved slot guaranteed to contain a top-3 bonus
   candidate — this directly solves the 0% 3+bonus rate
3. Lambda reduced 0.3 → 0.2 (less penalty for similar tickets; coverage 
   comes from the 30 unique numbers across 5 tickets, not forced spread)
4. Pool expanded from 500 → 1000 candidates for better bonus-aware pick
"""
LAMBDA = 0.3
MAX_SHARED = 2
N_TICKETS = 5

def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)

def overlap(a, b):
    return len(set(a) & set(b))

def select_diverse_tickets(scored_candidates, n_tickets=N_TICKETS,
                           lambda_=LAMBDA, max_shared=MAX_SHARED,
                           bonus_candidates=None):
    """
    scored_candidates : list of (combo_tuple, score) sorted desc by score
    bonus_candidates  : list of ints (top predicted bonus numbers), optional
                        One ticket slot is reserved to contain at least 1 of top-3.
    """
    if not scored_candidates:
        return []

    pool = list(scored_candidates[:1000])
    selected = []
    selected_sets = []

    # --- Slot 1: pure best score ---
    best_combo, _ = pool[0]
    selected.append(best_combo)
    selected_sets.append(set(best_combo))
    pool = pool[1:]

    # --- Slot 2-4: diversity-aware greedy ---
    while len(selected) < n_tickets - 1 and pool:
        best_adj = -999
        best_idx = -1
        for idx, (combo, score) in enumerate(pool):
            combo_set = set(combo)
            if selected_sets and max(len(combo_set & s) for s in selected_sets) > max_shared:
                continue
            max_jac = max(jaccard(combo, s) for s in selected) if selected else 0
            adj = score - lambda_ * max_jac
            if adj > best_adj:
                best_adj = adj
                best_idx = idx
        if best_idx == -1:
            # relax constraint — just pick next non-duplicate
            for i, (combo, _) in enumerate(pool):
                if combo not in selected:
                    selected.append(combo)
                    selected_sets.append(set(combo))
                    pool.pop(i)
                    break
        else:
            chosen, _ = pool[best_idx]
            selected.append(chosen)
            selected_sets.append(set(chosen))
            pool.pop(best_idx)

    # --- Slot 5 (last): BONUS-AWARE ---
    # Find the best-scoring ticket that contains at least one top-3 bonus candidate
    if bonus_candidates and len(bonus_candidates) >= 1:
        top3_bonus = set(int(b) for b in bonus_candidates[:3])
        bonus_ticket = None
        bonus_score = -999
        for combo, score in pool:
            combo_set = set(combo)
            if top3_bonus & combo_set:  # contains at least one bonus candidate
                # Mild overlap constraint (relaxed to 4 for this slot)
                if selected_sets and max(len(combo_set & s) for s in selected_sets) > 4:
                    continue
                max_jac = max(jaccard(combo, s) for s in selected) if selected else 0
                adj = score - 0.1 * max_jac  # very light penalty for last slot
                if adj > bonus_score:
                    bonus_score = adj
                    bonus_ticket = combo
        if bonus_ticket is not None:
            selected.append(bonus_ticket)
            selected_sets.append(set(bonus_ticket))
            pool = [(c, s) for c, s in pool if c != bonus_ticket]

    # Fill any remaining slots
    for combo, _ in scored_candidates:
        if len(selected) >= n_tickets:
            break
        if combo not in selected:
            selected.append(combo)

    return selected[:n_tickets]


def coverage_report(tickets):
    lines = [f"  {'Ticket':<10} {'Numbers':<38} {'Sum':>5} {'Odd':>4}",
             "  " + "-" * 60]
    for i, combo in enumerate(tickets, 1):
        nums = sorted(combo)
        lines.append(f"  T{i:<9} {str(nums):<38} {sum(nums):>5} {sum(1 for n in nums if n % 2):>4}")
    lines.append("\n  Pairwise overlap:")
    for i in range(len(tickets)):
        for j in range(i + 1, len(tickets)):
            lines.append(f"    T{i+1} <-> T{j+1}: {overlap(tickets[i], tickets[j])} shared | "
                         f"Jaccard={jaccard(tickets[i], tickets[j]):.3f}")
    all_nums = set(n for t in tickets for n in t)
    lines.append(f"\n  Unique numbers across 5 tickets: {len(all_nums)}")
    return "\n".join(lines)
