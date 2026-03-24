"""M9 — LSTM Sequence Model (20-draw lookback, predicts P(each number appears))"""
import numpy as np, os, joblib
from engine.features_v2 import build_features, FEATURE_COLS

MODELS_DIR=os.path.join(os.path.dirname(__file__),"..","..","saved_models")
LOOKBACK=20

try:
    import torch, torch.nn as nn, torch.optim as optim
    TORCH_AVAILABLE=True
except ImportError:
    TORCH_AVAILABLE=False

class _LSTMNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_dim, hidden1=128, hidden2=64, output_dim=49):
        if TORCH_AVAILABLE:
            super().__init__()
            self.lstm1=nn.LSTM(input_dim,hidden1,batch_first=True)
            self.lstm2=nn.LSTM(hidden1,hidden2,batch_first=True)
            self.fc1=nn.Linear(hidden2,64); self.relu=nn.ReLU()
            self.fc2=nn.Linear(64,output_dim); self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        out,_=self.lstm1(x); out,_=self.lstm2(out); out=out[:,-1,:]
        return self.sigmoid(self.fc2(self.relu(self.fc1(out))))

def _build_sequences(draws, train_end_idx):
    X,y=[],[]
    for i in range(LOOKBACK,train_end_idx):
        seq=[[build_features(sorted(draws[j]["nums"]),draws[:j]).get(c,0) for c in FEATURE_COLS] for j in range(i-LOOKBACK,i)]
        X.append(seq)
        target=np.zeros(49,dtype=np.float32)
        for n in draws[i]["nums"]: target[n-1]=1.0
        y.append(target)
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

class LSTMScorer:
    def __init__(self, epochs=15, lr=1e-3, batch_size=32, seed=42):
        self.epochs=epochs; self.lr=lr; self.batch_size=batch_size; self.seed=seed
        self.net=None; self._trained=False

    def fit(self, draws, train_end_idx):
        if not TORCH_AVAILABLE: print("  [M9] PyTorch not available"); return self
        import torch; torch.manual_seed(self.seed)
        print(f"  [M9] Building sequences (draws {LOOKBACK}-{train_end_idx})...")
        X,y=_build_sequences(draws,train_end_idx)
        print(f"  [M9] Sequences: {len(X)} | shape: {X.shape}")
        self.net=_LSTMNet(input_dim=X.shape[2])
        optimizer=optim.Adam(self.net.parameters(),lr=self.lr)
        criterion=nn.BCELoss()
        loader=torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(X),torch.FloatTensor(y)),
            batch_size=self.batch_size,shuffle=True)
        for epoch in range(self.epochs):
            total_loss=0.0; self.net.train()
            for xb,yb in loader:
                preds=self.net(xb); loss=criterion(preds,yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss+=loss.item()
            if (epoch+1)%5==0: print(f"  [M9] Epoch {epoch+1}/{self.epochs} | loss={total_loss/len(loader):.5f}")
        self._trained=True; return self

    def _predict_probs(self, history):
        if not TORCH_AVAILABLE or self.net is None or not self._trained: return np.ones(49)/49
        import torch
        recent=history[-LOOKBACK:] if len(history)>=LOOKBACK else history
        seq=[[build_features(d["nums"],recent[:j]).get(c,0) for c in FEATURE_COLS] for j,d in enumerate(recent)]
        while len(seq)<LOOKBACK: seq.insert(0,[0.0]*len(FEATURE_COLS))
        self.net.eval()
        with torch.no_grad(): return self.net(torch.FloatTensor([seq])).numpy()[0]

    def score(self, combo, history):
        probs=self._predict_probs(history)
        return float(np.mean([probs[n-1] for n in combo]))

    def score_batch(self, candidates, history):
        probs=self._predict_probs(history)
        return [float(np.mean([probs[n-1] for n in c])) for c in candidates]

    def save(self, suffix=""):
        if not TORCH_AVAILABLE or self.net is None: return
        import torch; os.makedirs(MODELS_DIR,exist_ok=True)
        torch.save(self.net.state_dict(),os.path.join(MODELS_DIR,f"m9_lstm{suffix}.pt"))
        joblib.dump({"trained":self._trained,"input_dim":self.net.lstm1.input_size},
                    os.path.join(MODELS_DIR,f"m9_meta{suffix}.pkl"))

    def load(self, suffix=""):
        if not TORCH_AVAILABLE: return False
        import torch
        meta_path=os.path.join(MODELS_DIR,f"m9_meta{suffix}.pkl"); pt_path=os.path.join(MODELS_DIR,f"m9_lstm{suffix}.pt")
        if os.path.exists(pt_path) and os.path.exists(meta_path):
            meta=joblib.load(meta_path); self.net=_LSTMNet(input_dim=meta["input_dim"])
            self.net.load_state_dict(torch.load(pt_path)); self._trained=meta["trained"]; return True
        return False
