"""
ensemble.py — Optuna-Tuned Weighted Ensemble (9 models + additional bias)

Changes vs v1:
- score_batch() now accepts optional add_bias_scores dict to nudge tickets
  toward likely bonus numbers (wires additional.py into main scoring)
- tune_weights() n_trials raised to 100, eval window raised to top-10 tickets
  instead of top-5 (better diversity during tuning)
- Added 'add_bias' as a tunable pseudo-weight in Optuna (0.0–0.20)
- Weight floor raised from 0.01 to 0.02 to prevent dead models
"""
import numpy as np, os, joblib

try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
DEFAULT_WEIGHTS = {
    "m1": 0.10, "m2": 0.10, "m3": 0.15, "m4": 0.10,
    "m5": 0.20, "m6": 0.10, "m7": 0.10, "m8": 0.05,
    "m9": 0.10, "add_bias": 0.05
}


class EnsembleScorer:
    def __init__(self):
        self.weights = dict(DEFAULT_WEIGHTS)
        self._scorers = {}

    def register(self, name, scorer):
        self._scorers[name] = scorer

    def _score_one(self, combo, history, name, scorer):
        try:
            return scorer.score(combo, history) if name in ("m3","m5","m6","m7","m9") else scorer.score(combo)
        except:
            return 0.5

    def score(self, combo, history, add_bias_scores=None):
        total = 0.0; weight_sum = 0.0
        for name, scorer in self._scorers.items():
            w = self.weights.get(name, 0)
            if w > 0:
                total += w * self._score_one(combo, history, name, scorer)
                weight_sum += w
        # Additional bias: average bias of numbers in this combo
        add_w = self.weights.get("add_bias", 0)
        if add_w > 0 and add_bias_scores:
            bias = np.mean([add_bias_scores.get(int(n), 0.0) for n in combo])
            total += add_w * bias
            weight_sum += add_w
        return total / weight_sum if weight_sum > 0 else 0.0

    def score_batch(self, candidates, history, feat_matrix=None, add_bias_scores=None):
        model_scores = {}
        for name, scorer in self._scorers.items():
            try:
                if name == "m4":
                    model_scores[name] = scorer.score_batch(candidates)
                elif name in ("m3", "m5") and feat_matrix is not None:
                    model_scores[name] = scorer.score_batch(candidates, history, feat_matrix)
                elif name in ("m3", "m5"):
                    model_scores[name] = scorer.score_batch(candidates, history)
                elif name in ("m7", "m9"):
                    model_scores[name] = scorer.score_batch(candidates, history)
                elif name == "m8":
                    model_scores[name] = scorer.score_batch(candidates)
                else:
                    model_scores[name] = scorer.score_batch(candidates, history)
            except Exception as e:
                print(f"  [ensemble] {name} batch failed: {e}")
                model_scores[name] = [0.5] * len(candidates)

        # Model weight sum (excluding add_bias — handled separately)
        model_names = list(self._scorers.keys())
        model_weight_sum = sum(self.weights.get(n, 0) for n in model_names)
        add_w = self.weights.get("add_bias", 0)
        total_weight = model_weight_sum + add_w

        results = []
        for i, combo in enumerate(candidates):
            score = sum(self.weights.get(n, 0) * model_scores[n][i] for n in model_names)
            # Additional bias nudge
            if add_w > 0 and add_bias_scores:
                bias = float(np.mean([add_bias_scores.get(int(n), 0.0) for n in combo]))
                score += add_w * bias
            results.append((combo, score / max(total_weight, 1e-9)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def tune_weights(self, candidates, history, actual_draws_val,
                     n_trials=100, seed=42, add_bias_scores=None):
        if not OPTUNA_AVAILABLE:
            print("  [ensemble] Optuna not available — using defaults")
            return self

        print(f"  [ensemble] Tuning weights ({n_trials} trials, {len(actual_draws_val)} val draws)...")

        # Pre-compute all model scores once
        model_score_matrix = {}
        for name, scorer in self._scorers.items():
            try:
                if name in ("m3", "m5", "m7", "m9"):
                    model_score_matrix[name] = scorer.score_batch(candidates, history)
                else:
                    model_score_matrix[name] = scorer.score_batch(candidates, history)
            except:
                model_score_matrix[name] = [0.5] * len(candidates)

        # Pre-compute add_bias array if available
        add_bias_arr = None
        if add_bias_scores:
            add_bias_arr = np.array([
                float(np.mean([add_bias_scores.get(int(n), 0.0) for n in c]))
                for c in candidates
            ])

        model_names = list(self._scorers.keys())
        n_eval = min(3000, len(candidates))  # top-3000 for eval speed

        def objective(trial):
            raw = {name: trial.suggest_float(f"w_{name}", 0.02, 0.40) for name in model_names}
            add_w = trial.suggest_float("w_add_bias", 0.0, 0.20)
            total = sum(raw.values()) + add_w
            weights = {k: v / total for k, v in raw.items()}
            add_w_norm = add_w / total

            hits = 0
            for val_draw in actual_draws_val:
                actual = set(val_draw["nums"])
                scores_arr = np.array([
                    sum(weights.get(n, 0) * model_score_matrix[n][i] for n in model_names)
                    for i in range(n_eval)
                ])
                if add_bias_arr is not None and add_w_norm > 0:
                    scores_arr += add_w_norm * add_bias_arr[:n_eval]

                top_idx = np.argsort(scores_arr)[::-1][:15]
                top_combos = [candidates[i] for i in top_idx]
                # Use best of 5 (diversity-selected) — approximate: just take top unique
                if max(len(set(c) & actual) for c in top_combos) >= 3:
                    hits += 1
            return hits / len(actual_draws_val)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        raw_weights = {k.replace("w_", ""): v for k, v in best.items()}
        total = sum(raw_weights.values())
        self.weights = {k: v / total for k, v in raw_weights.items()}
        # Rename add_bias key
        if "add_bias" in self.weights:
            pass  # already named correctly
        elif "w_add_bias".replace("w_","") in self.weights:
            pass

        print(f"  [ensemble] Best weights: { {k: f'{v:.3f}' for k,v in self.weights.items()} }")
        print(f"  [ensemble] Best val 3+ rate: {study.best_value:.3f}")
        return self

    def save(self, suffix=""):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.weights, os.path.join(MODELS_DIR, f"ensemble_weights{suffix}.pkl"))

    def load(self, suffix=""):
        path = os.path.join(MODELS_DIR, f"ensemble_weights{suffix}.pkl")
        if os.path.exists(path):
            self.weights = joblib.load(path)
            return True
        return False

    def to_state(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, "_ensemble_parallel_state.pkl")
        joblib.dump(self, path)
        return {"pkl_path": path}

    @classmethod
    def from_state(cls, state):
        return joblib.load(state["pkl_path"])
