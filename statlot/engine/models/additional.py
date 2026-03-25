"""Additional (Bonus) Number Predictor — Bayesian freq + Markov + recency on bonus sequence

Bugs fixed vs v1:
- Markov keys were stored as int but history dicts may have str keys → normalised to int
- predict() now recomputes recency-weighted scores fresh each call (not frozen _scores)
- Expanded top_n default to 5 for better 3+bonus coverage
- Added get_bias_scores() so ensemble can nudge tickets toward likely bonus numbers
"""
import numpy as np
from collections import Counter, defaultdict


class AdditionalPredictor:
    def __init__(self, alpha=1.0, decay_lambda=0.05, lookback=200, top_n=5):
        self.alpha = alpha
        self.decay_lambda = decay_lambda
        self.lookback = lookback
        self.top_n = top_n
        self._base_scores = {}       # bayes freq scores (fit-time, normalised)
        self._markov = defaultdict(Counter)   # int → Counter[int]
        self._additional_seq = []    # list of int

    # ------------------------------------------------------------------ fit --
    def fit(self, history):
        recent = history[-self.lookback:] if len(history) >= self.lookback else history

        # Ensure additional values are int (guard against string keys in dicts)
        additional_seq = []
        for d in recent:
            v = d.get("additional")
            if v is not None:
                try:
                    additional_seq.append(int(v))
                except (ValueError, TypeError):
                    pass

        freq = Counter(additional_seq)
        total = len(additional_seq) or 1
        alpha = self.alpha

        # Bayesian frequency base scores
        base = {}
        for n in range(1, 50):
            base[n] = (freq.get(n, 0) + alpha) / (total + alpha * 49)
        total_b = sum(base.values())
        self._base_scores = {n: base[n] / total_b for n in base}

        # Markov: int keys
        self._markov.clear()
        for i in range(1, len(additional_seq)):
            prev = int(additional_seq[i - 1])
            curr = int(additional_seq[i])
            self._markov[prev][curr] += 1

        self._additional_seq = additional_seq
        return self

    # --------------------------------------------------------------- predict --
    def predict(self, history, main_combo=None, top_n=None):
        top_n = top_n or self.top_n
        if not self._base_scores:
            return list(range(1, top_n + 1))

        # Start from base (fit-time) scores
        scores = dict(self._base_scores)

        # Recency decay boost — recomputed fresh each call using current history tail
        recent_seq = []
        for d in reversed(history[-50:]):
            v = d.get("additional")
            if v is not None:
                try:
                    recent_seq.append(int(v))
                except (ValueError, TypeError):
                    pass

        for k, a in enumerate(recent_seq, 1):
            boost = np.exp(-self.decay_lambda * k)
            scores[a] = scores.get(a, 0) * (1.0 + boost)

        # Markov boost from last known additional
        last_additional = recent_seq[0] if recent_seq else None
        if last_additional is not None and last_additional in self._markov:
            trans = self._markov[last_additional]
            total_t = sum(trans.values()) or 1
            for n, cnt in trans.items():
                scores[n] = scores.get(n, 0) * (1 + cnt / total_t)

        # Exclude main combo numbers
        if main_combo:
            for n in main_combo:
                scores.pop(int(n), None)

        return sorted(scores, key=lambda n: scores[n], reverse=True)[:top_n]

    # ------------------------------------------------- bias scores for ensemble
    def get_bias_scores(self, history, strength=0.15):
        """Return a dict {num: bias_weight} for all 49 numbers.
        Used by ensemble to give a soft nudge toward likely bonus numbers.
        strength=0.15 means up to +15% score boost for top additional candidates.
        """
        if not self._base_scores:
            return {}
        scores = dict(self._base_scores)

        recent_seq = []
        for d in reversed(history[-50:]):
            v = d.get("additional")
            if v is not None:
                try:
                    recent_seq.append(int(v))
                except (ValueError, TypeError):
                    pass

        for k, a in enumerate(recent_seq, 1):
            scores[a] = scores.get(a, 0) * (1.0 + np.exp(-self.decay_lambda * k))

        last_additional = recent_seq[0] if recent_seq else None
        if last_additional and last_additional in self._markov:
            trans = self._markov[last_additional]
            total_t = sum(trans.values()) or 1
            for n, cnt in trans.items():
                scores[n] = scores.get(n, 0) * (1 + cnt / total_t)

        # Normalise to [0, strength]
        max_s = max(scores.values()) if scores else 1.0
        return {n: (v / max_s) * strength for n, v in scores.items()}

    # ---------------------------------------------------------- score_hit -----
    def score_hit(self, predicted_additionals, actual_additional):
        if actual_additional is None:
            return False
        return int(actual_additional) in [int(x) for x in predicted_additionals]

    # ------------------------------------------------ serialisation ----------
    def to_state(self):
        return {
            "alpha": self.alpha,
            "decay_lambda": self.decay_lambda,
            "lookback": self.lookback,
            "top_n": self.top_n,
            "base_scores": {str(k): v for k, v in self._base_scores.items()},
            "markov": {str(k): {str(kk): vv for kk, vv in v.items()}
                       for k, v in self._markov.items()},
            "additional_seq": list(self._additional_seq),
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(
            alpha=state["alpha"],
            decay_lambda=state["decay_lambda"],
            lookback=state["lookback"],
            top_n=state["top_n"],
        )
        obj._base_scores = {int(k): v for k, v in state.get("base_scores", {}).items()}
        obj._additional_seq = [int(x) for x in state.get("additional_seq", [])]
        for k, v in state.get("markov", {}).items():
            obj._markov[int(k)] = Counter({int(kk): vv for kk, vv in v.items()})
        return obj
