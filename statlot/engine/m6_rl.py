"""
M6 — Reinforcement Learning Agent (Q-Learning / Policy Gradient)
50,000 iterations. Learns which structural features to prioritize.

State:  feature vector of last N draws (frequency, sum, odd/even, decade distribution)
Action: select weight vector for candidate scoring
Reward: +10 for 4+ match, +3 for 3+ match, -1 for <3 match

Architecture:
- Tabular Q-learning on discretized state space (fast, interpretable)
- Epsilon-greedy exploration (ε=1.0 → 0.01 over 50k steps)
- Experience replay buffer (size=2000)
- Learning rate: 0.001, discount γ=0.95
"""
import numpy as np
import json
import os
import random
from collections import deque, Counter
from typing import List, Tuple, Optional


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Action space: weight presets for [freq, pair, aging, hot_cold, decade]
# Each action is a normalized weight vector
ACTIONS = [
    {"freq": 3.0, "pair": 2.0, "aging": 1.5, "hot_cold": 1.0, "decade": 0.5},   # A0: freq-heavy
    {"freq": 2.0, "pair": 3.0, "aging": 1.5, "hot_cold": 1.0, "decade": 0.5},   # A1: pair-heavy
    {"freq": 2.0, "pair": 2.0, "aging": 3.0, "hot_cold": 1.0, "decade": 0.5},   # A2: aging-heavy
    {"freq": 2.0, "pair": 2.0, "aging": 1.5, "hot_cold": 3.0, "decade": 0.5},   # A3: hot-heavy
    {"freq": 2.0, "pair": 2.0, "aging": 1.5, "hot_cold": 1.0, "decade": 3.0},   # A4: decade-heavy
    {"freq": 2.5, "pair": 2.5, "aging": 2.0, "hot_cold": 1.5, "decade": 1.0},   # A5: balanced+
    {"freq": 1.5, "pair": 1.5, "aging": 1.0, "hot_cold": 2.0, "decade": 2.0},   # A6: structure-heavy
    {"freq": 3.5, "pair": 1.5, "aging": 2.0, "hot_cold": 0.5, "decade": 1.0},   # A7: freq+aging
]
N_ACTIONS = len(ACTIONS)


def _state_from_draws(draws: list, lookback: int = 20) -> tuple:
    """
    Discretize the last `lookback` draws into a compact state tuple.
    State components (all discretized to integers for Q-table):
    - avg_sum_bucket (0-7)
    - dominant_odd (0-6)
    - top3_hot_numbers (as sorted tuple of 3)
    - decade_distribution (5-tuple)
    """
    if len(draws) < lookback:
        lookback = len(draws)
    window = draws[-lookback:]

    sums = [d["n1"]+d["n2"]+d["n3"]+d["n4"]+d["n5"]+d["n6"] for d in window]
    avg_sum = np.mean(sums)
    # Bucket: 0=<100, 1=100-120, ..., 7=220+
    sum_bucket = min(7, max(0, int((avg_sum - 70) / 20)))

    all_odds = [sum(1 for x in [d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]] if x%2!=0)
                for d in window]
    dom_odd = int(Counter(all_odds).most_common(1)[0][0])

    freq = Counter()
    for d in window:
        for n in [d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]]:
            freq[n] += 1
    top3 = tuple(sorted([n for n, _ in freq.most_common(3)]))

    decade_counts = [0, 0, 0, 0, 0]
    for d in window:
        for n in [d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]]:
            decade_counts[min(4, (n - 1) // 10)] += 1
    decade_dist = tuple(min(9, c // lookback) for c in decade_counts)

    return (sum_bucket, dom_odd) + top3 + decade_dist


def _score_combo(combo: tuple, draws: list, weights: dict) -> float:
    """Score a combo using the given weight vector."""
    freq = Counter()
    pair_freq = Counter()
    n = len(draws)
    for i, d in enumerate(draws):
        w = 1.0 + (i / n) * 2.0
        nums = [d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]]
        for num in nums:
            freq[num] += w
        nums_s = sorted(nums)
        for a in range(6):
            for b in range(a+1, 6):
                pair_freq[(nums_s[a], nums_s[b])] += w

    # Aging
    last_seen = {}
    for i, d in enumerate(draws):
        for num in [d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]]:
            last_seen[num] = i
    aging = {num: min(n - last_seen.get(num, 0), 30) for num in range(1, 50)}

    # Hot/cold
    freq_50 = Counter()
    for d in draws[-50:]:
        for num in [d["n1"],d["n2"],d["n3"],d["n4"],d["n5"],d["n6"]]:
            freq_50[num] += 1
    hot_set = set([num for num, _ in freq_50.most_common(10)])

    nums = sorted(combo[:6])
    f = sum(freq.get(num, 0.1) for num in nums)
    p = sum(pair_freq.get((nums[a], nums[b]), 0)
            for a in range(6) for b in range(a+1, 6))
    ag = sum(aging.get(num, 1) for num in nums)
    hc = sum(1 for num in nums if num in hot_set)
    dec = len(set(min(4, (num-1)//10) for num in nums))

    return (weights["freq"]*f + weights["pair"]*p + weights["aging"]*ag +
            weights["hot_cold"]*hc + weights["decade"]*dec)


class RLAgent:
    """
    Tabular Q-learning agent for weight selection.
    Trained over 50,000 iterations on historical draw windows.
    """

    def __init__(self, n_actions: int = N_ACTIONS, lr: float = 0.001,
                 gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.9999):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}   # state -> np.array(n_actions)
        self.replay_buffer = deque(maxlen=2000)
        self.total_steps = 0
        self.training_history = []

    def _get_q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def select_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self._get_q(state)))

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def replay(self, batch_size: int = 32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state in batch:
            q = self._get_q(state)
            q_next = self._get_q(next_state)
            target = reward + self.gamma * np.max(q_next)
            q[action] += self.lr * (target - q[action])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, draws: list, n_iterations: int = 50000,
              n_candidates_per_step: int = 100, min_history: int = 100,
              progress_every: int = 5000):
        """
        Train the RL agent on historical draw data.
        At each step:
        1. Sample a random draw window from history
        2. Build state from that window
        3. Select action (weight vector)
        4. Generate candidates using those weights
        5. Score candidates against the NEXT draw (ground truth)
        6. Compute reward from best match
        7. Update Q-table
        """
        from engine.candidate_gen import generate_candidates

        draws_sorted = sorted(draws, key=lambda d: d["draw_number"])
        N = len(draws_sorted)

        if N < min_history + 10:
            print(f"Not enough draws ({N}) to train RL agent")
            return

        print(f"Training RL agent: {n_iterations} iterations on {N} draws...")
        rewards_window = deque(maxlen=500)
        best_avg_reward = -999

        for step in range(n_iterations):
            # Sample a random position in history
            idx = random.randint(min_history, N - 2)
            history = draws_sorted[:idx]
            test_draw = draws_sorted[idx]
            actual = set([test_draw["n1"], test_draw["n2"], test_draw["n3"],
                          test_draw["n4"], test_draw["n5"], test_draw["n6"]])

            # Build state from window
            state = _state_from_draws(history)
            action = self.select_action(state)
            weights = ACTIONS[action]

            # Generate a small set of candidates
            try:
                candidates = generate_candidates(history, pool_size=6,
                                                  n_candidates=n_candidates_per_step,
                                                  weights=weights)
                if not candidates:
                    continue

                # Score candidates against actual draw
                best_match = max(len(set(c) & actual) for c in candidates[:20])
            except Exception:
                continue

            # Reward function
            if best_match >= 4:
                reward = 10.0
            elif best_match == 3:
                reward = 3.0
            elif best_match == 2:
                reward = 0.5
            else:
                reward = -1.0

            rewards_window.append(reward)

            # Next state (one draw later)
            next_history = draws_sorted[:idx + 1]
            next_state = _state_from_draws(next_history)

            self.remember(state, action, reward, next_state)
            self.replay(batch_size=32)
            self.total_steps += 1

            if (step + 1) % progress_every == 0:
                avg_r = np.mean(rewards_window) if rewards_window else 0
                win_rate = sum(1 for r in rewards_window if r >= 3.0) / max(len(rewards_window), 1)
                print(f"  Step {step+1}/{n_iterations} | ε={self.epsilon:.4f} | "
                      f"avg_reward={avg_r:.3f} | win_rate(3+)={win_rate:.2%} | "
                      f"q_states={len(self.q_table)}")
                self.training_history.append({
                    "step": step + 1,
                    "epsilon": self.epsilon,
                    "avg_reward": round(float(avg_r), 4),
                    "win_rate": round(float(win_rate), 4),
                })
                if avg_r > best_avg_reward:
                    best_avg_reward = avg_r
                    self.save()

        print(f"RL training complete. Best avg reward: {best_avg_reward:.3f}")
        print(f"Final win rate (last 500 steps): "
              f"{sum(1 for r in rewards_window if r >= 3) / max(len(rewards_window),1):.2%}")

    def get_best_weights(self, draws: list) -> dict:
        """Get the best weight vector for the current state of draws."""
        state = _state_from_draws(draws)
        action = int(np.argmax(self._get_q(state)))
        return ACTIONS[action]

    def save(self, path: str = None):
        os.makedirs(MODELS_DIR, exist_ok=True)
        if path is None:
            path = os.path.join(MODELS_DIR, "rl_agent.json")
        data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_history": self.training_history,
            "n_actions": self.n_actions,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"RL agent saved: {path} ({len(self.q_table)} states)")

    def load(self, path: str = None) -> bool:
        if path is None:
            path = os.path.join(MODELS_DIR, "rl_agent.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.q_table = {eval(k): np.array(v) for k, v in data["q_table"].items()}
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.total_steps = data.get("total_steps", 0)
        self.training_history = data.get("training_history", [])
        print(f"RL agent loaded: {len(self.q_table)} states, {self.total_steps} steps trained")
        return True
