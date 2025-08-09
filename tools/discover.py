#!/usr/bin/env python3
"""
CLI entrypoint for math-space discovery.

Selects device via existing CLI logic (CUDA on A100, CPU on M3 by default),
then runs the DiscoveryEngine and prints JSON results.
"""

from __future__ import annotations

import json
import sys

from datetime import datetime
import os
import shutil

from physics_agent.discovery import DiscoveryEngine
from physics_agent.cli import get_cli_parser, determine_device_and_dtype
from physics_agent.evaluation import generate_discovery_report


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Reuse existing CLI parser; we only care about device/dtype flags here
    parser = get_cli_parser()
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--max-symbols", type=int, default=6)
    parser.add_argument("--mcts-sims", type=int, default=30)
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for report and JSON")
    # Adaptive stopping
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--target-unique", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    # Token overrides
    parser.add_argument("--operands", type=str, default=None)
    parser.add_argument("--unary", type=str, default=None)
    parser.add_argument("--binary", type=str, default=None)
    args = parser.parse_args(argv)

    device, _ = determine_device_and_dtype(args)

    # On Apple M-series, prefer CPU to preserve float64 in SymPy stages
    if device == "mps":
        device = "cpu"

    operands_override = [t.strip() for t in args.operands.split(',')] if args.operands else None
    unary_override = [t.strip() for t in args.unary.split(',')] if args.unary else None
    binary_override = [t.strip() for t in args.binary.split(',')] if args.binary else None

    engine = DiscoveryEngine(
        max_symbols=args.max_symbols,
        device=device,
        operands_override=operands_override,
        unary_override=unary_override,
        binary_override=binary_override,
    )
    # Adaptive loop or single call
    use_adaptive = any([
        args.time_budget_seconds and args.time_budget_seconds > 0,
        args.target_unique and args.target_unique > 0,
        args.patience and args.patience > 0,
    ])

    if not use_adaptive:
        candidates = engine.run(num_candidates=args.num_candidates, mcts_sims=args.mcts_sims)
    else:
        import time
        start_time = time.time()
        candidates = []
        seen = set()
        no_improve = 0
        while True:
            batch = engine.run(num_candidates=max(1, args.batch_size), mcts_sims=args.mcts_sims)
            new_added = 0
            for item in batch:
                key = str(item.get('expression', ''))
                if key and key not in seen:
                    seen.add(key)
                    candidates.append(item)
                    new_added += 1
            if args.target_unique and args.target_unique > 0 and len(seen) >= args.target_unique:
                break
            if args.time_budget_seconds and args.time_budget_seconds > 0 and (time.time() - start_time) >= args.time_budget_seconds:
                break
            if args.patience and args.patience > 0:
                if new_added == 0:
                    no_improve += 1
                    if no_improve >= args.patience:
                        break
                else:
                    no_improve = 0

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir or os.path.join("runs", f"discovery_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(run_dir, "candidates.json")
    with open(json_path, "w") as f:
        json.dump(candidates, f, indent=2)

    # Generate simple HTML report
    html_path = os.path.join(run_dir, "discovery_report.html")
    generate_discovery_report(candidates, html_path, run_dir)

    print(json.dumps({
        "device": device,
        "run_dir": run_dir,
        "json": json_path,
        "html": html_path,
    }, indent=2))

    # Copy to docs/latest_run for quick access (best-effort)
    try:
        docs_latest = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "latest_run")
        os.makedirs(docs_latest, exist_ok=True)
        shutil.copy(html_path, os.path.join(docs_latest, "discovery_latest.html"))
    except Exception:
        pass


if __name__ == "__main__":
    main()


