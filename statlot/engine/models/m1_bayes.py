"""M1 — Bayesian Frequency + Exponential Decay"""
import numpy as np
from collections import Counter

class BayesianFreqScorer:
    def __init__(self, alpha=1.0, decay_lambda=0.05, lookback=200):
        self.alpha=alpha; self.decay_lambda=decay_lambda; self.lookback=lookback
        self._number_scores={}

    def fit(self, history):
        recent=history[-self.lookback:] if len(history)>=self.lookback else history
        freq=Counter(n for d in recent for n in d["nums"]); total=len(recent); alpha=self.alpha
        scores={}
        for n in range(1,50):
            bayes_p=(freq.get(n,0)+alpha)/(total*6/49+alpha*49)
            decay=0.0
            for k,d in enumerate(reversed(recent[-50:]),1):
                if n in d["nums"]: decay+=np.exp(-self.decay_lambda*k)
            scores[n]=bayes_p*(1.0+decay)
        total_s=sum(scores.values())
        self._number_scores={n:scores[n]/total_s for n in scores}
        return self

    def score(self, combo):
        if not self._number_scores: return 0.0
        return float(np.mean([self._number_scores.get(n,0) for n in combo]))

    def score_batch(self, candidates, history=None):
        return [self.score(c) for c in candidates]
