"""
Math-space Discovery Engine.

Guided symbolic candidate generation using a Transformer policy and MCTS.
Builds expressions from a finite DSL in reverse Polish notation (stack).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from sympy import Basic, simplify

from .dsl import VOCAB, SYMBOL_TABLE, is_operand, is_unary, is_binary
from .pruning import passes_pruning
from .lagrangian_classifier import classify_lagrangian
from .policy_net import PolicyNet
from .mcts import MCTS
from .validators import validate_expression


@dataclass
class Candidate:
    expression: Basic
    classification: str


class DiscoveryEngine:
    def __init__(
        self,
        max_symbols: int = 6,
        device: str | None = None,
        operands_override: list[str] | None = None,
        unary_override: list[str] | None = None,
        binary_override: list[str] | None = None,
        mode: str = "lagrangian",  # "lagrangian" or "force_free"
    ):
        # Resolve device using existing CLI semantics if not provided
        if device is None:
            try:
                from physics_agent.cli import get_cli_parser, determine_device_and_dtype
                parser = get_cli_parser()
                args, _ = parser.parse_known_args([])
                device, _ = determine_device_and_dtype(args)
            except Exception:
                device = "cpu"

        # Prefer CPU float64 on Apple MPS for precision, but keep CUDA fast path
        if device == "mps":
            # MPS lacks float64; we keep the policy on CPU for determinism
            device = "cpu"

        self.max_symbols = int(max_symbols)
        self.device = device
        # Allow restriction of token space to reduce search
        if any([operands_override, unary_override, binary_override]):
            operands = operands_override if operands_override is not None else []
            unary = unary_override if unary_override is not None else []
            binary = binary_override if binary_override is not None else []
            # Fallback to full sets when empty list provided
            from .dsl import OPERANDS as ALL_OPERANDS, UNARY_FUNCS as ALL_UNARY, BINARY_OPS as ALL_BINARY
            if not operands:
                operands = list(ALL_OPERANDS)
            if not unary:
                unary = list(ALL_UNARY)
            if not binary:
                binary = list(ALL_BINARY)
            vocab = operands + unary + binary
        else:
            vocab = VOCAB

        # De-duplicate while preserving order
        seen = set()
        self.vocab = [t for t in vocab if not (t in seen or seen.add(t))]
        self.token_to_idx = {tok: i for i, tok in enumerate(self.vocab)}
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}
        self.mode = mode
        self.policy_net = PolicyNet(len(self.vocab)).to(self.device)

    def _decode_tokens(self, token_ids: List[int]) -> List[str]:
        return [self.idx_to_token[i] for i in token_ids]

    def _tokens_to_expr(self, tokens: List[str]) -> Basic | None:
        # Stack-based evaluation (RPN)
        stack: List[Basic] = []
        for tok in tokens:
            if is_operand(tok):
                stack.append(SYMBOL_TABLE[tok])
            elif is_unary(tok):
                if len(stack) < 1:
                    return None
                a = stack.pop()
                _, fn = SYMBOL_TABLE[tok]
                try:
                    stack.append(fn(a))
                except Exception:
                    return None
            elif is_binary(tok):
                if len(stack) < 2:
                    return None
                b = stack.pop()
                a = stack.pop()
                _, fn = SYMBOL_TABLE[tok]
                try:
                    stack.append(fn(a, b))
                except Exception:
                    return None
            else:
                return None

        if len(stack) != 1:
            return None
        return stack[0]

    def run(self, num_candidates: int = 50, mcts_sims: int = 50) -> List[Dict[str, Any]]:
        mcts = MCTS(self.policy_net, len(self.vocab), self.max_symbols, self.device)
        results: List[Dict[str, Any]] = []

        for _ in range(num_candidates):
            state: List[int] = []
            for _ in range(self.max_symbols):
                for _ in range(mcts_sims):
                    mcts.search(state)
                visits = [mcts.N.get((tuple(state), a), 0) for a in range(len(self.vocab))]
                action = int(max(range(len(self.vocab)), key=lambda a: visits[a]))
                state.append(action)

            tokens = self._decode_tokens(state)
            expr = self._tokens_to_expr(tokens)
            if expr is None:
                continue
            if not passes_pruning(expr):
                continue
            try:
                expr_simplified = simplify(expr)
            except Exception:
                expr_simplified = expr

            if not passes_pruning(expr_simplified):
                continue

            # Mode-specific validation and classification
            if self.mode == 'force_free':
                validation = validate_expression(expr_simplified, mode='force_free')
                if not validation['valid']:
                    continue
                classification = validation.get('classification', 'Unknown')
            else:
                classification = classify_lagrangian(expr_simplified)
                
            results.append({
                "expression": str(expr_simplified),
                "classification": classification,
                "mode": self.mode,
            })

        return results


__all__ = ["DiscoveryEngine", "Candidate"]


