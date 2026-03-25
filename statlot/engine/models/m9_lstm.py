"""M9 — LSTM Sequence Model (20-draw lookback, predicts P(each number appears))"""
import numpy as np, os, joblib

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


def _build_sequences_fast(draws, train_end_idx):
    """Vectorized sequence builder — builds full feature matrix once, then slices."""
    from engine.features_v2 import build_features_batch, FEATURE_COLS
    n = train_end_idx
    print(f"  [M9] Vectorizing {n} draw features for sequence building...")

    # Build features for all draws using a rolling history approximation:
    # group in chunks so history context is reasonably current
    all_candidates = [tuple(sorted(d["nums"])) for d in draws[:n]]
    feat_rows = np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)

    CHUNK = 50  # smaller chunks = more accurate history context
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        batch = all_candidates[start:end]
        history_ctx = draws[:start] if start >= 1 else draws[:1]
        feats = build_features_batch(batch, history_ctx)
        feat_rows[start:end] = feats[:end-start]

    print(f"  [M9] Feature matrix built: {feat_rows.shape}")

    X, y = [], []
    for i in range(LOOKBACK, n):
        X.append(feat_rows[i-LOOKBACK:i])
        target = np.zeros(49, dtype=np.float32)
        for num in draws[i]["nums"]: target[num-1] = 1.0
        y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMScorer:
    def __init__(self, epochs=15, lr=1e-3, batch_size=32, seed=42):
        self.epochs=epochs; self.lr=lr; self.batch_size=batch_size; self.seed=seed
        self.net=None; self._trained=False; self._feat_cols=None

    def fit(self, draws, train_end_idx):
        if not TORCH_AVAILABLE: print("  [M9] PyTorch not available"); return self
        import torch; torch.manual_seed(self.seed)
        print(f"  [M9] Building sequences (draws {LOOKBACK}-{train_end_idx})...")
        X, y = _build_sequences_fast(draws, train_end_idx)
        print(f"  [M9] Sequences: {len(X)} | shape: {X.shape}")
        self.net = _LSTMNet(input_dim=X.shape[2])
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
            batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            total_loss = 0.0; self.net.train()
            for xb, yb in loader:
                preds = self.net(xb); loss = criterion(preds, yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 5 == 0:
                print(f"  [M9] Epoch {epoch+1}/{self.epochs} | loss={total_loss/len(loader):.5f}")
        self._trained = True; return self

    def _predict_probs(self, history):
        if not TORCH_AVAILABLE or self.net is None or not self._trained:
            return np.ones(49) / 49
        from engine.features_v2 import build_features_batch, FEATURE_COLS
        import torch
        recent = history[-LOOKBACK:] if len(history) >= LOOKBACK else history
        batch = [tuple(sorted(d["nums"])) for d in recent]
        feats = build_features_batch(batch, history[:-len(recent)] if len(history) > len(recent) else history[:1])
        seq = feats.tolist() if hasattr(feats, 'tolist') else feats
        while len(seq) < LOOKBACK:
            seq.insert(0, [0.0] * len(FEATURE_COLS))
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.FloatTensor([seq[-LOOKBACK:]])).numpy()[0]

    def score(self, combo, history):
        probs = self._predict_probs(history)
        return float(np.mean([probs[n-1] for n in combo]))

    def score_batch(self, candidates, history):
        probs = self._predict_probs(history)
        return [float(np.mean([probs[n-1] for n in c])) for c in candidates]

    def save(self, suffix=""):
        if not TORCH_AVAILABLE or self.net is None: return
        import torch; os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(MODELS_DIR, f"m9_lstm{suffix}.pt"))

    def load(self, suffix=""):
        from engine.features_v2 import FEATURE_COLS
        if not TORCH_AVAILABLE: return False
        import torch
        path = os.path.join(MODELS_DIR, f"m9_lstm{suffix}.pt")
        if not os.path.exists(path): return False
        self.net = _LSTMNet(input_dim=len(FEATURE_COLS))
        self.net.load_state_dict(torch.load(path, map_location='cpu'))
        self._trained = True; return True
