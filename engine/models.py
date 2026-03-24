"""
ML models: Random Forest (M3), XGBoost (M5), Monte Carlo (M4).
Each model predicts STRUCTURAL PROPERTIES of the next draw,
which are then used to filter candidate combinations.
"""
import numpy as np
import joblib
import os
from collections import Counter

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier, XGBRegressor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from engine.features import FEATURE_COLS


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class DrawPropertyPredictor:
    """
    Predicts the structural properties of the NEXT draw:
      - sum_bucket (0-7)
      - odd_count (0-6)
      - low_count (0-6)
      - empty_decades (0-4)
      - consecutive_count (0-5)
    """

    def __init__(self, model_type="rf"):
        self.model_type = model_type
        self.models = {}
        self.targets = ["sum_bucket", "odd_count", "low_count",
                        "empty_decades", "consecutive_count"]

    def _sum_to_bucket(self, s):
        if s < 100: return 0
        elif s < 120: return 1
        elif s < 140: return 2
        elif s < 160: return 3
        elif s < 180: return 4
        elif s < 200: return 5
        elif s < 220: return 6
        else: return 7

    def prepare_data(self, feature_rows: list[dict]):
        """feature_rows already built by features.py, sorted ascending."""
        X, y = [], []
        for i in range(len(feature_rows) - 1):
            row = feature_rows[i]
            nxt = feature_rows[i + 1]
            x_vec = [row.get(col, 0) for col in FEATURE_COLS]
            y_vec = [
                self._sum_to_bucket(nxt["sum"]),
                nxt["odd_count"],
                nxt["low_count"],
                nxt["empty_decades"],
                nxt["consecutive_count"],
            ]
            X.append(x_vec)
            y.append(y_vec)
        return np.array(X), np.array(y)

    def train(self, X, y):
        if not ML_AVAILABLE:
            print("ML libraries not available — skipping training")
            return

        for i, target in enumerate(self.targets):
            y_t = y[:, i]
            if self.model_type == "rf":
                m = RandomForestClassifier(
                    n_estimators=100, max_depth=8,
                    random_state=42, n_jobs=-1
                )
            else:  # xgb
                m = XGBClassifier(
                    n_estimators=330, max_depth=6,
                    learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, random_state=42,
                    eval_metric="mlogloss", verbosity=0
                )
            m.fit(X, y_t)
            self.models[target] = m

    def predict(self, x_row: list) -> dict:
        if not self.models:
            return {}
        x = np.array(x_row).reshape(1, -1)
        preds = {}
        for target, m in self.models.items():
            preds[target] = int(m.predict(x)[0])
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(x)[0]
                preds[f"{target}_proba"] = proba.tolist()
        return preds

    def save(self, suffix=""):
        os.makedirs(MODELS_DIR, exist_ok=True)
        for target, m in self.models.items():
            path = os.path.join(MODELS_DIR, f"{self.model_type}_{target}{suffix}.pkl")
            joblib.dump(m, path)

    def load(self, suffix=""):
        for target in self.targets:
            path = os.path.join(MODELS_DIR, f"{self.model_type}_{target}{suffix}.pkl")
            if os.path.exists(path):
                self.models[target] = joblib.load(path)


class MonteCarloCandidateScorer:
    """
    M4: Run N simulations. For each candidate combo, simulate
    how often it would match 3+, 4+, 5+ against draws from
    a bootstrapped sample of historical draws.
    Returns ranked combos with confidence intervals.
    """

    def __init__(self, n_simulations=10000):
        self.n_sims = n_simulations

    def score_candidates(self, candidates: list, historical_draws: list) -> list:
        """
        candidates: list of 6-tuples
        historical_draws: list of draw dicts with n1-n6
        Returns list of (combo, scores_dict) sorted by expected_match desc
        """
        results = []
        draw_sets = [
            frozenset([d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]])
            for d in historical_draws
        ]
        n_hist = len(draw_sets)

        for combo in candidates:
            combo_set = set(combo)
            match_counts = []

            for _ in range(self.n_sims):
                # Bootstrap sample a draw
                sample_draw = draw_sets[np.random.randint(0, n_hist)]
                match_counts.append(len(combo_set & sample_draw))

            mc = np.array(match_counts)
            results.append({
                "combo": list(combo),
                "expected_match": round(float(np.mean(mc)), 4),
                "std": round(float(np.std(mc)), 4),
                "p3plus": round(float(np.mean(mc >= 3)), 4),
                "p4plus": round(float(np.mean(mc >= 4)), 4),
                "p5plus": round(float(np.mean(mc >= 5)), 4),
                "ci_low": round(float(np.percentile(mc, 5)), 4),
                "ci_high": round(float(np.percentile(mc, 95)), 4),
            })

        results.sort(key=lambda x: x["p3plus"], reverse=True)
        return results
