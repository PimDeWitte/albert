#!/bin/bash
# Example script showing different parallelism configurations

echo "=== Comprehensive Theory Test - Parallelism Examples ==="
echo

echo "1. Default (4 particles in parallel, theories sequential):"
echo "   python evaluation.py"
echo

echo "2. Run 8 trajectories in parallel (2 theories × 4 particles):"
echo "   python evaluation.py --max-concurrent-trajectories 8"
echo

echo "3. Run 12 trajectories in parallel (3 theories × 4 particles):"
echo "   python evaluation.py --max-concurrent-trajectories 12"
echo

echo "4. Attempt 15 (will adjust to 12 - nearest multiple of 4):"
echo "   python evaluation.py --max-concurrent-trajectories 15"
echo

echo "5. Attempt 7 (will adjust to 4 - single theory):"
echo "   python evaluation.py --max-concurrent-trajectories 7"
echo

# Run with default settings
python evaluation.py