"""M4 — Monte Carlo + Importance Sampling (recency-weighted bootstrap)"""
import numpy as np

class MonteCarloScorer:
    def __init__(self, n_simulations=3000, lookback=100, decay_lambda=0.03):
        self.n_simulations=n_simulations; self.lookback=lookback; self.decay_lambda=decay_lambda
        self._draw_sets=[]; self._weights=[]

    def fit(self, history):
        recent=history[-self.lookback:] if len(history)>=self.lookback else history
        self._draw_sets=[frozenset(d["nums"]) for d in recent]; n=len(self._draw_sets)
        raw_weights=np.array([np.exp(-self.decay_lambda*(n-i)) for i in range(n)])
        self._weights=raw_weights/raw_weights.sum(); return self

    def score(self, combo):
        if not self._draw_sets: return 0.0
        combo_set=set(combo); n=len(self._draw_sets)
        indices=np.random.choice(n,size=self.n_simulations,replace=True,p=self._weights)
        return float(np.mean([len(combo_set&self._draw_sets[i])>=3 for i in indices]))

    def score_batch(self, candidates, history=None):
        if not self._draw_sets: return [0.0]*len(candidates)
        n=len(self._draw_sets)
        indices=np.random.choice(n,size=self.n_simulations,replace=True,p=self._weights)
        sampled=[self._draw_sets[i] for i in indices]
        return [sum(1 for d in sampled if len(set(c)&d)>=3)/self.n_simulations for c in candidates]
