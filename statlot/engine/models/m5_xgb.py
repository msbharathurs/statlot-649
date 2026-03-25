"""M5 — XGBoost Binary Classifier + Optuna tuning + SHAP"""
import numpy as np, os, joblib
from engine.features_v2 import build_features, FEATURE_COLS
from engine.models.m3_rf import _build_training_data

try:
    from xgboost import XGBClassifier
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    XGB_AVAILABLE=True
except ImportError:
    XGB_AVAILABLE=False

MODELS_DIR=os.path.join(os.path.dirname(__file__),"..","..","saved_models")

class XGBScorer:
    def __init__(self, seed=42, n_optuna_trials=50):
        self.seed=seed; self.n_optuna_trials=n_optuna_trials; self.model=None; self.best_params={}

    def fit(self, draws, train_end_idx):
        if not XGB_AVAILABLE: print("  [M5] xgboost/optuna not available"); return self
        print(f"  [M5] Building training data (draws 1-{train_end_idx})...")
        X,y=_build_training_data(draws,train_end_idx,neg_per_draw=7)
        val_start=max(0,len(X)-50*8); X_train,X_val=X[:val_start],X[val_start:]; y_train,y_val=y[:val_start],y[val_start:]
        pos=y_train.sum(); neg=len(y_train)-pos; scale_pos=neg/max(pos,1)
        print(f"  [M5] train={len(y_train)} val={len(y_val)} | scale_pos_weight={scale_pos:.2f}")
        def objective(trial):
            params={"n_estimators":trial.suggest_int("n_estimators",100,500),
                    "max_depth":trial.suggest_int("max_depth",3,8),
                    "learning_rate":trial.suggest_float("learning_rate",0.01,0.3,log=True),
                    "subsample":trial.suggest_float("subsample",0.6,1.0),
                    "colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),
                    "min_child_weight":trial.suggest_int("min_child_weight",1,10),
                    "scale_pos_weight":scale_pos,"eval_metric":"logloss",
                    "random_state":self.seed,"verbosity":0,"n_jobs":-1}
            m=XGBClassifier(**params)
            m.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
            preds=m.predict_proba(X_val)[:,1]; pred_pos=preds>=0.5
            if pred_pos.sum()==0: return 0.0
            return float((y_val[pred_pos]==1).mean())
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective,n_trials=self.n_optuna_trials,show_progress_bar=False)
        self.best_params=study.best_params
        print(f"  [M5] Best params: {self.best_params}")
        final_params={**self.best_params,"scale_pos_weight":scale_pos,"eval_metric":"logloss",
                      "random_state":self.seed,"verbosity":0,"n_jobs":-1}
        self.model=XGBClassifier(**final_params); self.model.fit(X,y)
        return self

    def score(self, combo, history):
        if self.model is None: return 0.5
        feats=build_features(list(combo),history)
        x=np.array([[feats.get(c,0) for c in FEATURE_COLS]],dtype=np.float32)
        return float(self.model.predict_proba(x)[0][1])

    def score_batch(self, candidates, history, feat_matrix=None):
        if self.model is None: return [0.5]*len(candidates)
        if feat_matrix is not None:
            import numpy as np
            X = feat_matrix if hasattr(feat_matrix, "shape") else np.array(feat_matrix, dtype=np.float32)
        else:
            from engine.features_v2 import build_features, FEATURE_COLS
            from engine.features_v2 import build_features_batch
            X = build_features_batch([tuple(sorted(c)) for c in candidates], history).astype(np.float32)
        return self.model.predict_proba(X.astype(np.float32))[:,1].tolist()

    def save(self, suffix=""):
        os.makedirs(MODELS_DIR,exist_ok=True)
        joblib.dump(self.model,os.path.join(MODELS_DIR,f"m5_xgb{suffix}.pkl"))

    def load(self, suffix=""):
        path=os.path.join(MODELS_DIR,f"m5_xgb{suffix}.pkl")
        if os.path.exists(path): self.model=joblib.load(path); return True
        return False
