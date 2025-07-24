#!/bin/bash
# Setup Script for Physics Agent Albert
# <reason>chain: Environment setup for Physics Agent Albert - Einstein's legacy computational framework</reason>

set -e  # Exit on error

echo "=== Physics Agent Albert Setup ==="
echo "Setting up environment for gravitational theory computation and validation..."
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check for Python 3.9+
if ! command -v python3 &> /dev/null
then
    print_error "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MIN_VERSION="3.9"

# Correctly compare versions using sort -V
if [ "$(printf '%s\n' "$MIN_VERSION" "$PY_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    print_error "Python 3.9 or higher is required. You have version $PY_VERSION."
    exit 1
fi
print_status "Python version $PY_VERSION found."

echo "Creating virtual environment in .venv..."
python3 -m venv .venv
source .venv/bin/activate
print_status "Virtual environment created and activated."

echo "Upgrading pip..."
python3 -m pip install --upgrade pip
print_status "pip has been upgraded."

echo "Installing Physics Agent dependencies from requirements.txt..."
python3 -m pip install -r requirements.txt
print_status "All Physics Agent dependencies from requirements.txt are installed."

echo "Installing Physics Agent UI dependencies..."
python3 -m pip install -r physics_agent/ui/requirements.txt
print_status "Physics Agent UI dependencies installed (Flask, requests, werkzeug)."

echo -e "\n${GREEN}=== Physics Agent Albert Setup Complete ===${NC}"
echo
echo "To activate the environment in your current shell, run:"
echo -e "${YELLOW}source .venv/bin/activate${NC}"
echo
echo "To start the Physics Agent web UI, run:"
echo -e "${YELLOW}python -m physics_agent.ui.server${NC}"
echo
echo "Then open: ${YELLOW}http://localhost:8000${NC}"
echo
echo "To run Albert and validate theories, use:"
echo -e "${YELLOW}./albert run${NC}"
echo
echo "For more options:"
echo -e "${YELLOW}./albert --help${NC}" 