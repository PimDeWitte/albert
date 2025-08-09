#!/bin/bash

# Mathematical Theorem Scraper - Setup Script
# This script installs all necessary dependencies for the theorem scraper

echo "Installing Python packages for theorem scraper..."

# Install Python packages
pip install requests beautifulsoup4 selenium webdriver-manager arxiv pymupdf

# For Ubuntu/Debian systems (uncomment if running on Linux server):
# sudo apt update
# sudo apt install -y wget unzip chromium-chromedriver

# For macOS (using Homebrew):
# brew install --cask chromedriver

echo "Dependencies installed successfully!"
echo "Now run the scraper with: python scraper.py"