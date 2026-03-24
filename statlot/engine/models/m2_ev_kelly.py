"""M2 — Expected Value + Fractional Kelly Criterion"""
import numpy as np
from math import comb
from collections import Counter

PRIZE_TIERS={3:10,4:100,5:2500,6:1_000_000}; TICKET_COST=3.0; KELLY_FRACTION=0.25

def _freq_probs(history, lookback=100):
    recent=history[-lookback:] if len(history)>=lookback else history
    freq=Counter(n for d in recent for n in d["nums"])
    total_hits=sum(freq.values()) or 1
    return {n:freq.get(n,0)/total_hits for n in range(1,50)}

def _poisson_match_prob(combo, freq_probs, k):
    probs=[min(max(freq_probs.get(n,1/49)*6,0.0),1.0) for n in combo]
    n=len(probs)
    if k>n: return 0.0
    dp=[0.0]*(n+1); dp[0]=1.0
    for p in probs:
        for j in range(n,0,-1): dp[j]=dp[j]*(1-p)+dp[j-1]*p
    return dp[k]

class EVKellyScorer:
    def __init__(self, kelly_fraction=KELLY_FRACTION, lookback=100):
        self.kelly_fraction=kelly_fraction; self.lookback=lookback; self._freq_probs={}

    def fit(self, history):
        self._freq_probs=_freq_probs(history,self.lookback); return self

    def score(self, combo):
        if not self._freq_probs: return 0.0
        ev=sum(_poisson_match_prob(combo,self._freq_probs,k)*prize for k,prize in PRIZE_TIERS.items())
        if ev<=0: return 0.0
        kelly=self.kelly_fraction*(ev-TICKET_COST)/max(ev,0.001)
        return float(kelly)

    def score_batch(self, candidates):
        return [self.score(c) for c in candidates]
