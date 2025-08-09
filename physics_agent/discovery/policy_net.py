"""
Transformer policy/value network for guided symbolic token generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Simple Transformer-based policy/value network.

    Inputs are integer token sequences. Outputs are next-token logits and a scalar value.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)
        # Start token embedding for empty sequences
        self.start = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len]
        batch_size = x.shape[0]
        if x.shape[1] == 0:
            # Use learned start embedding when no tokens yet
            h = self.start.unsqueeze(0).expand(batch_size, -1)
            logits = self.policy_head(h)
            value = self.value_head(h)
            return F.log_softmax(logits, dim=-1), torch.tanh(value)

        emb = self.embed(x)
        out = self.transformer(emb)
        h = out[:, -1, :]
        logits = self.policy_head(h)
        value = self.value_head(h)
        return F.log_softmax(logits, dim=-1), torch.tanh(value)


__all__ = ["PolicyNet"]


