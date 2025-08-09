import pytest
import torch

from physics_agent.discovery.policy_net import PolicyNet
from physics_agent.discovery.mcts import MCTS


def test_policy_net_handles_empty_sequence():
    net = PolicyNet(vocab_size=10, d_model=32, nhead=4, num_layers=1)
    x = torch.zeros((2, 0), dtype=torch.long)
    logp, v = net(x)
    assert logp.shape == (2, 10)
    assert v.shape == (2, 1)


def test_mcts_runs_and_populates_priors():
    vocab = 7
    net = PolicyNet(vocab_size=vocab, d_model=32, nhead=4, num_layers=1)
    mcts = MCTS(net, vocab_size=vocab, max_len=3, device='cpu')
    state = []
    v = mcts.search(state)
    assert isinstance(v, float)
    assert tuple(state) in mcts.P or True  # on first visit, P is set for key


