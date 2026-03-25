"""M8 — Gaussian Mixture Model Density Estimator (k=8 components)"""
import numpy as np, os, joblib
from engine.features_v2 import dual_grid_score, rowA, rowB

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    GMM_AVAILABLE=True
except ImportError:
    GMM_AVAILABLE=False

MODELS_DIR=os.path.join(os.path.dirname(__file__),"..","..","saved_models")

def _combo_to_gmm_features(nums):
    nums=sorted(nums); rA=[rowA(n) for n in nums]; rB=[rowB(n) for n in nums]
    gaps=[nums[i+1]-nums[i] for i in range(5)]
    decades=[sum(1 for n in nums if lo<=n<=hi) for lo,hi in [(1,10),(11,20),(21,30),(31,40),(41,49)]]
    return [sum(nums),sum(1 for n in nums if n%2),len(set(rA)),len(set(rB)),
            float(np.mean(gaps)),float(np.std(gaps)),dual_grid_score(nums)/100.0,
            sum(1 for d in decades if d>0),sum(1 for n in nums if n<=24),max(rA)-min(rA)]

class GMMScorer:
    def __init__(self, n_components=8, seed=42):
        self.n_components=n_components; self.seed=seed; self.gmm=None; self.scaler=None

    def fit(self, draws, train_end_idx):
        if not GMM_AVAILABLE: print("  [M8] sklearn not available"); return self
        print(f"  [M8] Fitting GMM on {train_end_idx} draws...")
        # Use float64 for numerical stability
        X=np.array([_combo_to_gmm_features(sorted(d["nums"])) for d in draws[:train_end_idx]],dtype=np.float64)
        self.scaler=StandardScaler(); X_scaled=self.scaler.fit_transform(X)
        # reg_covar=1e-3 prevents singular covariance; try n_components=8, fall back to 5 or 3
        for n_comp in [self.n_components, 5, 3]:
            try:
                self.gmm=GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    random_state=self.seed,
                    n_init=3,
                    reg_covar=1e-3,
                    max_iter=200
                )
                self.gmm.fit(X_scaled)
                print(f"  [M8] n_components={n_comp} log-likelihood={self.gmm.score(X_scaled):.4f}")
                break
            except Exception as e:
                print(f"  [M8] n_components={n_comp} failed: {e}, retrying with fewer components...")
                self.gmm=None
        if self.gmm is None:
            print("  [M8] All GMM fits failed — using diagonal covariance fallback")
            self.gmm=GaussianMixture(
                n_components=3,
                covariance_type="diag",
                random_state=self.seed,
                n_init=3,
                reg_covar=1e-2
            )
            self.gmm.fit(X_scaled)
            print(f"  [M8] Fallback diag GMM log-likelihood={self.gmm.score(X_scaled):.4f}")
        return self

    def score(self, combo):
        if self.gmm is None: return 0.5
        x=np.array([_combo_to_gmm_features(list(combo))],dtype=np.float64)
        ll=float(self.gmm.score_samples(self.scaler.transform(x))[0])
        return float(1/(1+np.exp(-ll/5)))

    def score_batch(self, candidates, history=None):
        if self.gmm is None: return [0.5]*len(candidates)
        X=np.array([_combo_to_gmm_features(list(c)) for c in candidates],dtype=np.float64)
        lls=self.gmm.score_samples(self.scaler.transform(X))
        return [float(1/(1+np.exp(-ll/5))) for ll in lls]

    def save(self, suffix=""):
        os.makedirs(MODELS_DIR,exist_ok=True)
        joblib.dump((self.gmm,self.scaler),os.path.join(MODELS_DIR,f"m8_gmm{suffix}.pkl"))

    def load(self, suffix=""):
        path=os.path.join(MODELS_DIR,f"m8_gmm{suffix}.pkl")
        if os.path.exists(path): self.gmm,self.scaler=joblib.load(path); return True
        return False
