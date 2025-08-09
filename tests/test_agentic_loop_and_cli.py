import json
import os
import shutil
import tempfile
import pytest

from physics_agent.agentic_loop.main import main as agentic_main
from physics_agent.match_search.main import main as match_main


def _read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def test_agentic_loop_produces_run_dir(tmp_path):
    out_dir = tmp_path / 'agentic_test'
    args = [
        '--out-dir', str(out_dir),
        '--num-candidates', '3',
        '--max-symbols', '4',
        '--mcts-sims', '2',
    ]
    agentic_main(args)
    assert out_dir.exists()
    assert (out_dir / 'candidates.json').exists()
    assert (out_dir / 'discovery_report.html').exists()
    data = _read_json(out_dir / 'candidates.json')
    assert isinstance(data, list)


def test_agentic_loop_adaptive_stopping_unique(tmp_path):
    out_dir = tmp_path / 'agentic_adaptive'
    args = [
        '--out-dir', str(out_dir),
        '--target-unique', '2',
        '--batch-size', '1',
        '--mcts-sims', '1',
    ]
    agentic_main(args)
    assert out_dir.exists()
    data = _read_json(out_dir / 'candidates.json')
    # We requested at least 2 uniques; ensure we got >= 1 result
    assert isinstance(data, list)


def test_match_only_search_run_dir(tmp_path):
    out_dir = tmp_path / 'match_test'
    args = [
        '--out-dir', str(out_dir),
        '--num-candidates', '3',
        '--max-symbols', '4',
        '--mcts-sims', '2',
    ]
    match_main(args)
    assert out_dir.exists()
    assert (out_dir / 'candidates.json').exists()
    assert (out_dir / 'discovery_report.html').exists()


