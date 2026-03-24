"""
diversity_select.py — Greedy Diversity-Aware 5-Ticket Selector
Jaccard penalty λ=0.3, max shared numbers per pair = 2
Maximises P(at least one of 5 tickets hits 3+) by covering different grid zones.
"""
LAMBDA=0.3; MAX_SHARED=2; N_TICKETS=5

def jaccard(a,b): sa,sb=set(a),set(b); return len(sa&sb)/len(sa|sb)
def overlap(a,b): return len(set(a)&set(b))

def select_diverse_tickets(scored_candidates, n_tickets=N_TICKETS, lambda_=LAMBDA, max_shared=MAX_SHARED):
    if not scored_candidates: return []
    best_combo,_=scored_candidates[0]; selected=[best_combo]; selected_sets=[set(best_combo)]
    pool=list(scored_candidates[1:500])
    while len(selected)<n_tickets and pool:
        best_adj=-999; best_idx=-1
        for idx,(combo,score) in enumerate(pool):
            combo_set=set(combo)
            if selected_sets and max(len(combo_set&s) for s in selected_sets)>max_shared: continue
            max_jac=max(jaccard(combo,s) for s in selected) if selected else 0
            adj=score-lambda_*max_jac
            if adj>best_adj: best_adj=adj; best_idx=idx
        if best_idx==-1:
            for combo,_ in pool:
                if combo not in selected: selected.append(combo); selected_sets.append(set(combo)); pool=[(c,s) for c,s in pool if c!=combo]; break
        else:
            chosen,_=pool[best_idx]; selected.append(chosen); selected_sets.append(set(chosen)); pool.pop(best_idx)
    for combo,_ in scored_candidates:
        if len(selected)>=n_tickets: break
        if combo not in selected: selected.append(combo)
    return selected[:n_tickets]

def coverage_report(tickets):
    lines=[f"  {'Ticket':<10} {'Numbers':<35} {'Sum':>5} {'Odd':>4}","  "+"-"*58]
    for i,combo in enumerate(tickets,1):
        nums=sorted(combo); lines.append(f"  T{i:<9} {str(nums):<35} {sum(nums):>5} {sum(1 for n in nums if n%2):>4}")
    lines.append("\n  Pairwise overlap:")
    for i in range(len(tickets)):
        for j in range(i+1,len(tickets)):
            lines.append(f"    T{i+1} <-> T{j+1}: {overlap(tickets[i],tickets[j])} shared | Jaccard={jaccard(tickets[i],tickets[j]):.3f}")
    all_nums=set(n for t in tickets for n in t)
    lines.append(f"\n  Unique numbers across 5 tickets: {len(all_nums)}")
    return "\n".join(lines)
