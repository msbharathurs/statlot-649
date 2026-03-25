"""
ensemble.py — Optuna-Tuned Weighted Ensemble (9 models)
Weights tuned on last 50 draws of train set. Batch scoring for efficiency.
"""
import numpy as np, os, joblib

try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); OPTUNA_AVAILABLE=True
except ImportError:
    OPTUNA_AVAILABLE=False

MODELS_DIR=os.path.join(os.path.dirname(__file__),"..","saved_models")
DEFAULT_WEIGHTS={"m1":0.10,"m2":0.10,"m3":0.15,"m4":0.10,"m5":0.20,"m6":0.10,"m7":0.10,"m8":0.05,"m9":0.10}

class EnsembleScorer:
    def __init__(self): self.weights=dict(DEFAULT_WEIGHTS); self._scorers={}

    def register(self, name, scorer): self._scorers[name]=scorer

    def _score_one(self, combo, history, name, scorer):
        try:
            return scorer.score(combo,history) if name in ("m3","m5","m6","m7","m9") else scorer.score(combo)
        except: return 0.5

    def score(self, combo, history):
        total=0.0; weight_sum=0.0
        for name,scorer in self._scorers.items():
            w=self.weights.get(name,0)
            if w>0: total+=w*self._score_one(combo,history,name,scorer); weight_sum+=w
        return total/weight_sum if weight_sum>0 else 0.0

    def score_batch(self, candidates, history, feat_matrix=None):
        model_scores={}
        for name,scorer in self._scorers.items():
            try:
                if name=="m4": model_scores[name]=scorer.score_batch(candidates)
                elif name in ("m3","m5") and feat_matrix is not None: model_scores[name]=scorer.score_batch(candidates,history,feat_matrix)
                elif name in ("m3","m5"): model_scores[name]=scorer.score_batch(candidates,history)
                elif name in ("m7","m9"): model_scores[name]=scorer.score_batch(candidates,history)
                elif name=="m8": model_scores[name]=scorer.score_batch(candidates)
                else: model_scores[name]=scorer.score_batch(candidates, history)
            except Exception as e:
                print(f"  [ensemble] {name} batch failed: {e}"); model_scores[name]=[0.5]*len(candidates)
        weight_sum=sum(self.weights.get(n,0) for n in self._scorers)
        results=[(combo,sum(self.weights.get(name,0)*model_scores[name][i] for name in self._scorers)/max(weight_sum,1e-9))
                 for i,combo in enumerate(candidates)]
        results.sort(key=lambda x:x[1],reverse=True)
        return results

    def tune_weights(self, candidates, history, actual_draws_val, n_trials=50, seed=42):
        if not OPTUNA_AVAILABLE: print("  [ensemble] Optuna not available — using defaults"); return self
        print(f"  [ensemble] Tuning weights ({n_trials} trials)...")
        model_score_matrix={}
        for name,scorer in self._scorers.items():
            try:
                if name in ("m3","m5","m7","m9"): model_score_matrix[name]=scorer.score_batch(candidates,history)
                else: model_score_matrix[name]=scorer.score_batch(candidates, history)
            except: model_score_matrix[name]=[0.5]*len(candidates)

        def objective(trial):
            raw={name:trial.suggest_float(f"w_{name}",0.01,0.40) for name in self._scorers}
            total=sum(raw.values()); weights={k:v/total for k,v in raw.items()}
            hits=0
            for val_draw in actual_draws_val:
                actual=set(val_draw["nums"])
                scored=sorted([(candidates[i],sum(weights.get(n,0)*model_score_matrix[n][i] for n in self._scorers))
                               for i in range(min(2000,len(candidates)))],key=lambda x:x[1],reverse=True)
                if max(len(set(c)&actual) for c,_ in scored[:5])>=3: hits+=1
            return hits/len(actual_draws_val)

        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective,n_trials=n_trials,show_progress_bar=False)
        best=study.best_params; total=sum(best.values())
        self.weights={k.replace("w_",""):v/total for k,v in best.items()}
        print(f"  [ensemble] Best weights: { {k:f'{v:.3f}' for k,v in self.weights.items()} }")
        print(f"  [ensemble] Best val 3+ rate: {study.best_value:.3f}")
        return self

    def save(self, suffix=""):
        os.makedirs(MODELS_DIR,exist_ok=True)
        joblib.dump(self.weights,os.path.join(MODELS_DIR,f"ensemble_weights{suffix}.pkl"))

    def load(self, suffix=""):
        path=os.path.join(MODELS_DIR,f"ensemble_weights{suffix}.pkl")
        if os.path.exists(path): self.weights=joblib.load(path); return True
        return False
