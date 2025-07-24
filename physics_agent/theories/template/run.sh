#!/bin/bash
# Run this theory with comparisons to defaults

# Go to project root
cd "$(dirname "$0")/../.."

# Run this theory - replace 'template' with your theory directory name
./run_theory.sh theories/template "$@" 