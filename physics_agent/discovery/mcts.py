"""
Monte Carlo Tree Search (MCTS) for token sequence generation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


class MCTS:
    def __init__(self, policy_net, vocab_size: int, max_len: int, device: str = "cpu", c_puct: float = 1.0):
        self.policy_net = policy_net
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = device
        self.c_puct = c_puct

        self.Q: Dict[Tuple[Tuple[int, ...], int], float] = defaultdict(float)
        self.N: Dict[Tuple[Tuple[int, ...], int], int] = defaultdict(int)
        self.P: Dict[Tuple[int, ...], List[float]] = {}

    def search(self, state: List[int]) -> float:
        if len(state) == self.max_len:
            return self.evaluate(state)

        key = tuple(state)
        if key not in self.P:
            state_tensor = torch.tensor([state], dtype=torch.long, device=self.device)
            log_probs, value = self.policy_net(state_tensor)
            probs = torch.exp(log_probs).detach().cpu().numpy()[0].tolist()
            self.P[key] = probs
            return float(value.item())

        # Select action with maximum UCB
        total_n = sum(self.N.get((key, a), 0) for a in range(self.vocab_size)) + 1
        best_score, best_a = -float("inf"), 0
        for a in range(self.vocab_size):
            q = self.Q.get((key, a), 0.0)
            n = self.N.get((key, a), 0)
            p = self.P[key][a]
            ucb = q / (1 + n) + self.c_puct * p * math.sqrt(total_n) / (1 + n)
            if ucb > best_score:
                best_score, best_a = ucb, a

        next_state = state + [best_a]
        v = self.search(next_state)
        self.N[(key, best_a)] = self.N.get((key, best_a), 0) + 1
        self.Q[(key, best_a)] = self.Q.get((key, best_a), 0.0) + v
        return v

    def evaluate(self, state: List[int]) -> float:
        # Terminal value placeholder (neutral)
        return 0.0


__all__ = ["MCTS"]


