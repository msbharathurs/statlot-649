"""M7 — 2nd Order Markov Chain + Laplace smoothing"""
import numpy as np
from collections import Counter, defaultdict

class MarkovScorer:
    def __init__(self, laplace=1.0, lookback=300):
        self.laplace=laplace; self.lookback=lookback
        self._first_order=defaultdict(Counter); self._second_order=defaultdict(Counter)
        self._number_counts=Counter(); self._total=0

    def fit(self, history):
        recent=history[-self.lookback:] if len(history)>=self.lookback else history
        self._first_order.clear(); self._second_order.clear(); self._number_counts.clear()
        for i,d in enumerate(recent):
            cur=set(d["nums"])
            for n in cur: self._number_counts[n]+=1
            self._total=len(recent)
            if i>=1:
                prev=set(recent[i-1]["nums"])
                for p in prev:
                    for c in cur: self._first_order[p][c]+=1
            if i>=2:
                prev=frozenset(recent[i-1]["nums"]); prev2=frozenset(recent[i-2]["nums"])
                for c in cur: self._second_order[(prev,prev2)][c]+=1
        return self

    def _p1(self, prev_num, target_num):
        counts=self._first_order.get(prev_num,{}); total=sum(counts.values())+self.laplace*49
        return (counts.get(target_num,0)+self.laplace)/total

    def _p2(self, prev_set, prev2_set, target_num):
        ctx=(frozenset(prev_set),frozenset(prev2_set)); counts=self._second_order.get(ctx,{})
        if not counts:
            total_count=sum(self._number_counts.values()) or 1
            return (self._number_counts.get(target_num,0)+self.laplace)/(total_count+self.laplace*49)
        total=sum(counts.values())+self.laplace*49
        return (counts.get(target_num,0)+self.laplace)/total

    def score(self, combo, history):
        nums=list(combo)
        if len(history)<2: return 0.5
        prev=set(history[-1]["nums"]); prev2=set(history[-2]["nums"]) if len(history)>=2 else set()
        score=0.0
        for n in nums:
            p1=np.mean([self._p1(p,n) for p in prev]) if prev else 1/49
            p2=self._p2(prev,prev2,n)
            score+=0.4*p1+0.6*p2
        return float(score/6)

    def score_batch(self, candidates, history):
        return [self.score(c,history) for c in candidates]
