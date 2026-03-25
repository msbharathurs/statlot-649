"""M3 — Random Forest Binary Classifier (direct P(3+ match) + SMOTE + Platt)"""
import numpy as np, os, joblib
from collections import Counter
from engine.features_v2 import build_features, FEATURE_COLS, should_eliminate

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import SMOTE
    ML_AVAILABLE=True
except ImportError:
    ML_AVAILABLE=False

MODELS_DIR=os.path.join(os.path.dirname(__file__),"..","..","saved_models")

def _build_training_data(draws, train_end_idx, neg_per_draw=5, seed=42):
    np.random.seed(seed); X,y=[],[]
    for i in range(1,train_end_idx):
        history=draws[:i]; actual=sorted(draws[i]["nums"]); actual_set=set(actual)
        feats=build_features(actual,history)
        X.append([feats.get(c,0) for c in FEATURE_COLS]); y.append(1)
        added=0; attempts=0
        while added<neg_per_draw and attempts<200:
            attempts+=1
            nums=sorted(np.random.choice(range(1,50),6,replace=False).tolist())
            if should_eliminate(nums): continue
            if len(set(nums)&actual_set)>=3: continue
            feats=build_features(nums,history)
            X.append([feats.get(c,0) for c in FEATURE_COLS]); y.append(0); added+=1
    return np.array(X,dtype=np.float32),np.array(y)

class RFScorer:
    def __init__(self, n_estimators=200, max_depth=10, seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.seed=seed; self.model=None

    def fit(self, draws, train_end_idx):
        if not ML_AVAILABLE: print("  [M3] sklearn/imblearn not available"); return self
        print(f"  [M3] Building training data (draws 1-{train_end_idx})...")
        X,y=_build_training_data(draws,train_end_idx)
        print(f"  [M3] Samples: {len(y)} | pos: {y.sum()} | neg: {len(y)-y.sum()}")
        try:
            sm=SMOTE(random_state=self.seed,k_neighbors=3); X_res,y_res=sm.fit_resample(X,y)
            print(f"  [M3] After SMOTE: {len(y_res)} samples")
        except Exception as e:
            print(f"  [M3] SMOTE failed ({e}), using raw data"); X_res,y_res=X,y
        base_rf=RandomForestClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,
            min_samples_leaf=5,n_jobs=-1,random_state=self.seed,oob_score=True)
        self.model=CalibratedClassifierCV(base_rf,method="sigmoid",cv=3)
        self.model.fit(X_res,y_res)
        base_rf.fit(X_res,y_res); print(f"  [M3] OOB: {base_rf.oob_score_:.4f}")
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
        joblib.dump(self.model,os.path.join(MODELS_DIR,f"m3_rf{suffix}.pkl"))

    def load(self, suffix=""):
        path=os.path.join(MODELS_DIR,f"m3_rf{suffix}.pkl")
        if os.path.exists(path): self.model=joblib.load(path); return True
        return False
