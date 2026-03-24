"""M6 — Deep Q-Network RL (state=last 10 draws, action=weight blend, reward=match quality)"""
import numpy as np, os, joblib
from collections import deque
from engine.features_v2 import build_features, FEATURE_COLS

MODELS_DIR=os.path.join(os.path.dirname(__file__),"..","..","saved_models")
WEIGHT_OPTIONS=[
    [0.15,0.10,0.20,0.10,0.25,0.10,0.05,0.05],[0.25,0.05,0.15,0.05,0.30,0.10,0.05,0.05],
    [0.10,0.20,0.15,0.05,0.20,0.15,0.05,0.10],[0.10,0.05,0.30,0.05,0.30,0.05,0.05,0.10],
    [0.10,0.10,0.15,0.20,0.20,0.10,0.05,0.10],[0.15,0.10,0.15,0.10,0.15,0.20,0.05,0.10],
    [0.10,0.10,0.20,0.10,0.20,0.10,0.10,0.10],[0.10,0.10,0.15,0.10,0.20,0.10,0.05,0.20],
]
N_ACTIONS=len(WEIGHT_OPTIONS)

try:
    import torch, torch.nn as nn, torch.optim as optim
    TORCH_AVAILABLE=True
except ImportError:
    TORCH_AVAILABLE=False

class DQNAgent:
    def __init__(self, state_dim=None, gamma=0.95, epsilon=0.1, lr=1e-3, buffer_capacity=500, tau=0.01, seed=42):
        self.state_dim=state_dim or len(FEATURE_COLS)*10
        self.gamma=gamma; self.epsilon=epsilon; self.tau=tau; self.seed=seed
        np.random.seed(seed); self.replay_buffer=deque(maxlen=buffer_capacity)
        self.q_table=np.zeros((1000,N_ACTIONS)); self._online_net=None; self._target_net=None
        self._optimizer=None; self._trained=False
        if TORCH_AVAILABLE: self._build_networks(lr)

    def _build_networks(self, lr):
        class QNet(nn.Module):
            def __init__(self,sd,na): super().__init__(); self.net=nn.Sequential(nn.Linear(sd,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,na))
            def forward(self,x): return self.net(x)
        self._online_net=QNet(self.state_dim,N_ACTIONS); self._target_net=QNet(self.state_dim,N_ACTIONS)
        self._target_net.load_state_dict(self._online_net.state_dict())
        self._optimizer=optim.Adam(self._online_net.parameters(),lr=lr)

    def _state_from_history(self, history):
        recent=history[-10:] if len(history)>=10 else history
        vecs=[[build_features(d["nums"],[]).get(c,0) for c in FEATURE_COLS] for d in recent]
        while len(vecs)<10: vecs.insert(0,[0.0]*len(FEATURE_COLS))
        return np.array(vecs,dtype=np.float32).flatten()

    def _state_hash(self, state): return abs(hash(state.tobytes()))%1000

    def get_best_weights(self, history):
        state=self._state_from_history(history)
        if TORCH_AVAILABLE and self._online_net and self._trained:
            import torch
            with torch.no_grad(): q_vals=self._online_net(torch.FloatTensor(state)).numpy()
            return WEIGHT_OPTIONS[int(np.argmax(q_vals))]
        return WEIGHT_OPTIONS[int(np.argmax(self.q_table[self._state_hash(state)]))]

    def store_transition(self, h_before, action_idx, reward, h_after):
        self.replay_buffer.append((self._state_from_history(h_before),action_idx,reward,self._state_from_history(h_after)))

    def _soft_update(self):
        if not TORCH_AVAILABLE or self._online_net is None: return
        for tp,op in zip(self._target_net.parameters(),self._online_net.parameters()):
            tp.data.copy_(self.tau*op.data+(1-self.tau)*tp.data)

    def train_step(self, batch_size=32):
        if len(self.replay_buffer)<batch_size: return
        if TORCH_AVAILABLE and self._online_net: self._train_torch(batch_size)
        else: self._train_tabular(batch_size)

    def _train_torch(self, batch_size):
        import torch, torch.nn.functional as F
        idxs=np.random.choice(len(self.replay_buffer),batch_size,replace=False)
        batch=[self.replay_buffer[i] for i in idxs]
        states=torch.FloatTensor(np.array([b[0] for b in batch])); actions=torch.LongTensor([b[1] for b in batch])
        rewards=torch.FloatTensor([b[2] for b in batch]); next_states=torch.FloatTensor(np.array([b[3] for b in batch]))
        q_vals=self._online_net(states).gather(1,actions.unsqueeze(1)).squeeze()
        with torch.no_grad(): next_q=self._target_net(next_states).max(1)[0]
        targets=rewards+self.gamma*next_q; loss=F.mse_loss(q_vals,targets)
        self._optimizer.zero_grad(); loss.backward(); self._optimizer.step(); self._soft_update()

    def _train_tabular(self, batch_size):
        for i in np.random.choice(len(self.replay_buffer),batch_size,replace=False):
            s,a,r,s2=self.replay_buffer[i]; idx=self._state_hash(s); idx2=self._state_hash(s2)
            self.q_table[idx][a]+=0.1*(r+self.gamma*np.max(self.q_table[idx2])-self.q_table[idx][a])

    def fit(self, draws, train_end_idx, n_episodes=3):
        print(f"  [M6] Training DQN on {train_end_idx} draws, {n_episodes} episodes...")
        from engine.candidate_gen_v2 import generate_candidates
        for episode in range(n_episodes):
            total_reward=0
            for i in range(10,train_end_idx-1):
                history=draws[:i]; action_idx=np.random.randint(N_ACTIONS) if np.random.rand()<self.epsilon else 0
                actual=set(draws[i+1]["nums"]); additional=draws[i+1].get("additional")
                candidates=generate_candidates(history,n_candidates=500)
                if not candidates: continue
                combo=list(candidates[0]); match=len(set(combo)&actual); bonus=1 if additional and additional in combo else 0
                reward=50 if match>=5 else (10 if match==4 else (3+bonus*2 if match>=3 else -1))
                total_reward+=reward; self.store_transition(history,action_idx,reward,draws[:i+1]); self.train_step()
            print(f"  [M6] Episode {episode+1}/{n_episodes} | total_reward={total_reward}")
        self._trained=True; return self

    def score(self, combo, history):
        state=self._state_from_history(history)
        if TORCH_AVAILABLE and self._online_net and self._trained:
            import torch
            with torch.no_grad(): q_vals=self._online_net(torch.FloatTensor(state)).numpy()
            return float(np.max(q_vals)/50.0)
        return float(np.max(self.q_table[self._state_hash(state)])/50.0)

    def save(self, suffix=""):
        os.makedirs(MODELS_DIR,exist_ok=True)
        if TORCH_AVAILABLE and self._online_net:
            import torch; torch.save(self._online_net.state_dict(),os.path.join(MODELS_DIR,f"m6_dqn{suffix}.pt"))
        joblib.dump(self.q_table,os.path.join(MODELS_DIR,f"m6_qtable{suffix}.pkl"))

    def load(self, suffix=""):
        path=os.path.join(MODELS_DIR,f"m6_qtable{suffix}.pkl")
        if os.path.exists(path): self.q_table=joblib.load(path); self._trained=True; return True
        return False
