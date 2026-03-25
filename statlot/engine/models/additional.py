"""
Additional (Bonus) Number Predictor v2 — Major upgrades:

1. Recency-weighted base scores (not static fit-time) — adapts each draw
2. Shorter effective lookback for Markov (30 draws) to track recent transitions
3. Gap/overdue signal — numbers not seen in a long time get boosted
4. Predict returns a WIDER set (top_n=7) by default
5. get_bias_scores uses overdue signal too
6. to_state/from_state unchanged for compatibility
"""
import numpy as np
from collections import Counter, defaultdict


class AdditionalPredictor:
    def __init__(self, alpha=1.0, decay_lambda=0.04, lookback=200, top_n=7):
        self.alpha = alpha
        self.decay_lambda = decay_lambda
        self.lookback = lookback
        self.top_n = top_n
        self._base_scores = {}
        self._markov = defaultdict(Counter)
        self._additional_seq = []

    def fit(self, history):
        recent = history[-self.lookback:] if len(history) >= self.lookback else history

        additional_seq = []
        for d in recent:
            v = d.get("additional")
            if v is not None:
                try:
                    additional_seq.append(int(v))
                except (ValueError, TypeError):
                    pass

        self._additional_seq = additional_seq

        # --- Recency-weighted frequency (not flat count) ---
        scores = {n: self.alpha for n in range(1, 50)}
        for k, a in enumerate(reversed(additional_seq)):
            weight = np.exp(-self.decay_lambda * k)
            scores[a] = scores.get(a, 0) + weight
        total = sum(scores.values())
        self._base_scores = {n: scores[n] / total for n in scores}

        # --- Markov: use shorter window for transitions ---
        markov_window = min(60, len(additional_seq))
        self._markov.clear()
        seq = additional_seq[-markov_window:]
        for i in range(1, len(seq)):
            prev = int(seq[i - 1])
            curr = int(seq[i])
            self._markov[prev][curr] += 1

        return self

    def _overdue_scores(self, history, strength=0.5):
        """Numbers not seen as additional in the last N draws get an overdue boost."""
        recent_seq = []
        for d in reversed(history[-100:]):
            v = d.get("additional")
            if v is not None:
                try:
                    recent_seq.append(int(v))
                except (ValueError, TypeError):
                    pass

        # Last-seen index for each number (lower = more recent = less overdue)
        last_seen = {}
        for k, a in enumerate(recent_seq):
            if a not in last_seen:
                last_seen[a] = k

        overdue = {}
        for n in range(1, 50):
            gap = last_seen.get(n, 100)  # 100 = never seen in window
            overdue[n] = np.log1p(gap) * strength

        max_od = max(overdue.values()) or 1.0
        return {n: v / max_od for n, v in overdue.items()}

    def predict(self, history, main_combo=None, top_n=None):
        top_n = top_n or self.top_n
        if not self._base_scores:
            return list(range(1, top_n + 1))

        scores = dict(self._base_scores)

        # Recency boost from very recent draws (last 20)
        recent_seq = []
        for d in reversed(history[-20:]):
            v = d.get("additional")
            if v is not None:
                try:
                    recent_seq.append(int(v))
                except (ValueError, TypeError):
                    pass

        for k, a in enumerate(recent_seq, 1):
            boost = np.exp(-0.1 * k)   # sharper decay = focus on last 5-6 draws
            scores[a] = scores.get(a, 0) * (1.0 + boost)

        # Markov boost
        last_additional = recent_seq[0] if recent_seq else None
        if last_additional is not None and last_additional in self._markov:
            trans = self._markov[last_additional]
            total_t = sum(trans.values()) or 1
            for n, cnt in trans.items():
                scores[n] = scores.get(n, 0) * (1 + 0.5 * cnt / total_t)

        # Overdue boost
        overdue = self._overdue_scores(history)
        for n in range(1, 50):
            scores[n] = scores.get(n, 0) * (1 + 0.3 * overdue.get(n, 0))

        # Exclude main combo
        if main_combo:
            for n in main_combo:
                scores.pop(int(n), None)

        return sorted(scores, key=lambda n: scores[n], reverse=True)[:top_n]

    def get_bias_scores(self, history, strength=0.20):
        """Bias scores for ensemble — higher strength + overdue signal."""
        if not self._base_scores:
            return {}
        scores = dict(self._base_scores)

        recent_seq = []
        for d in reversed(history[-20:]):
            v = d.get("additional")
            if v is not None:
                try:
                    recent_seq.append(int(v))
                except (ValueError, TypeError):
                    pass

        for k, a in enumerate(recent_seq, 1):
            scores[a] = scores.get(a, 0) * (1.0 + np.exp(-0.1 * k))

        last_additional = recent_seq[0] if recent_seq else None
        if last_additional and last_additional in self._markov:
            trans = self._markov[last_additional]
            total_t = sum(trans.values()) or 1
            for n, cnt in trans.items():
                scores[n] = scores.get(n, 0) * (1 + 0.5 * cnt / total_t)

        overdue = self._overdue_scores(history)
        for n in range(1, 50):
            scores[n] = scores.get(n, 0) * (1 + 0.4 * overdue.get(n, 0))

        max_s = max(scores.values()) if scores else 1.0
        return {n: (v / max_s) * strength for n, v in scores.items()}

    def score_hit(self, predicted_additionals, actual_additional):
        if actual_additional is None:
            return False
        return int(actual_additional) in [int(x) for x in predicted_additionals]

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
