"""
candidate_gen_v2.py — Candidate Generation + Dual-Grid Hard Filter
Generates ~20k candidate combos via biased sampling, kills ~30-40% via dual-grid elimination.
"""
import random, numpy as np
from itertools import combinations
from collections import Counter
from engine.features_v2 import should_eliminate, dual_grid_score

def _compute_number_weights(history, lookback=100):
    recent=history[-lookback:] if len(history)>=lookback else history
    alpha=1.0; freq=Counter(n for d in recent for n in d["nums"]); total=len(recent)
    weights={}
    for n in range(1,50):
        bayes=(freq.get(n,0)+alpha)/(total*6/49+alpha*49)
        decay=0.0
        for k,d in enumerate(reversed(recent[-30:]),1):
            if n in d["nums"]: decay+=np.exp(-0.05*k)
        w=bayes*(1.0+decay)
        if n>=46: w*=0.60
        elif n>=43: w*=0.80
        weights[n]=max(w,0.001)
    total_w=sum(weights.values())
    return {n:weights[n]/total_w for n in weights}

def _sample_combo(weights):
    nums=list(weights.keys()); probs=np.array([weights[n] for n in nums])
    probs/=probs.sum()
    chosen=np.random.choice(nums,size=6,replace=False,p=probs)
    return tuple(sorted(chosen))

def generate_candidates(history, n_candidates=20000, seed=None):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    weights=_compute_number_weights(history)
    seen=set(); candidates=[]; attempts=0; max_attempts=n_candidates*5
    while len(candidates)<n_candidates and attempts<max_attempts:
        combo=_sample_combo(weights); attempts+=1
        if combo in seen: continue
        seen.add(combo)
        if not should_eliminate(list(combo)): candidates.append(combo)
    return candidates

def score_candidates_basic(candidates, history):
    recent=history[-50:] if len(history)>=50 else history
    freq=Counter(n for d in recent for n in d["nums"])
    expected=max(len(recent)*6/49,0.001)
    pair_freq=Counter()
    for d in recent:
        pn=sorted(d["nums"])
        for a,b in combinations(pn,2): pair_freq[(a,b)]+=1
    scored=[]
    for combo in candidates:
        nums=list(combo); dgs=dual_grid_score(nums)
        freq_score=sum(freq.get(n,0) for n in nums)/(6*expected)
        pairs=[(nums[a],nums[b]) for a in range(6) for b in range(a+1,6)]
        pair_score=np.mean([pair_freq.get(p,0) for p in pairs])
        composite=(dgs/100)*0.5+freq_score*0.3+min(pair_score/3,1.0)*0.2
        scored.append((combo,composite))
    scored.sort(key=lambda x:x[1],reverse=True)
    return scored
