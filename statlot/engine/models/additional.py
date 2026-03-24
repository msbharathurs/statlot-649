"""Additional (Bonus) Number Predictor — Bayesian freq + Markov on bonus sequence"""
import numpy as np
from collections import Counter, defaultdict

class AdditionalPredictor:
    def __init__(self, alpha=1.0, decay_lambda=0.05, lookback=200, top_n=3):
        self.alpha=alpha; self.decay_lambda=decay_lambda; self.lookback=lookback; self.top_n=top_n
        self._scores={}; self._markov=defaultdict(Counter); self._additional_seq=[]

    def fit(self, history):
        recent=history[-self.lookback:] if len(history)>=self.lookback else history
        additional_seq=[d.get("additional") for d in recent if d.get("additional")]
        freq=Counter(additional_seq); total=len(additional_seq) or 1; alpha=self.alpha
        scores={}
        for n in range(1,50):
            bayes_p=(freq.get(n,0)+alpha)/(total+alpha*49)
            decay=sum(np.exp(-self.decay_lambda*k) for k,a in enumerate(reversed(additional_seq[-50:]),1) if a==n)
            scores[n]=bayes_p*(1.0+decay)
        self._markov.clear()
        for i in range(1,len(additional_seq)): self._markov[additional_seq[i-1]][additional_seq[i]]+=1
        total_s=sum(scores.values()); self._scores={n:scores[n]/total_s for n in scores}
        self._additional_seq=additional_seq; return self

    def predict(self, history, main_combo=None, top_n=None):
        top_n=top_n or self.top_n
        if not self._scores: return list(range(1,top_n+1))
        scores=dict(self._scores)
        last_additional=next((d.get("additional") for d in reversed(history) if d.get("additional")),None)
        if last_additional and last_additional in self._markov:
            trans=self._markov[last_additional]; total=sum(trans.values()) or 1
            for n,cnt in trans.items(): scores[n]=scores.get(n,0)*(1+cnt/total)
        if main_combo:
            for n in main_combo: scores.pop(n,None)
        return sorted(scores,key=lambda n:scores[n],reverse=True)[:top_n]

    def score_hit(self, predicted_additionals, actual_additional):
        return actual_additional in predicted_additionals if actual_additional else False
