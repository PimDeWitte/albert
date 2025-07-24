#!/usr/bin/env python3
"""
Update Checker - Checks for new versions of Albert
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Tuple
import os
import sys
from datetime import datetime, timedelta

# Current version - update this when releasing new versions
CURRENT_VERSION = "0.1.0"
GITHUB_API_URL = "https://api.github.com/repos/pimdewitte/albert/releases/latest"
UPDATE_CHECK_FILE = os.path.expanduser("~/.albert_update_check")
CHECK_INTERVAL_DAYS = 1


def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse version string into tuple for comparison"""
    try:
        # Remove 'v' prefix if present
        version = version.lstrip('v')
        parts = version.split('.')
        return tuple(int(p) for p in parts[:3])
    except:
        return (0, 0, 0)


def should_check_for_updates() -> bool:
    """Check if we should check for updates based on last check time"""
    if not os.path.exists(UPDATE_CHECK_FILE):
        return True
    
    try:
        with open(UPDATE_CHECK_FILE, 'r') as f:
            data = json.load(f)
            last_check = datetime.fromisoformat(data.get('last_check', '2000-01-01'))
            return datetime.now() - last_check > timedelta(days=CHECK_INTERVAL_DAYS)
    except:
        return True


def save_check_timestamp():
    """Save the timestamp of the last update check"""
    try:
        with open(UPDATE_CHECK_FILE, 'w') as f:
            json.dump({'last_check': datetime.now().isoformat()}, f)
    except:
        pass  # Fail silently


def get_latest_version() -> Optional[str]:
    """Fetch the latest version from GitHub"""
    try:
        # Create request with timeout
        req = urllib.request.Request(
            GITHUB_API_URL,
            headers={'Accept': 'application/vnd.github.v3+json'}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return data.get('tag_name', '').lstrip('v')
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        # Fail silently - don't interrupt user's work
        return None


def check_for_updates(silent: bool = False) -> bool:
    """
    Check for updates and notify user if available.
    
    Args:
        silent: If True, only show notification if update is available
        
    Returns:
        True if update is available, False otherwise
    """
    if not should_check_for_updates():
        return False
    
    # Save that we checked
    save_check_timestamp()
    
    latest_version = get_latest_version()
    if not latest_version:
        return False
    
    current_tuple = parse_version(CURRENT_VERSION)
    latest_tuple = parse_version(latest_version)
    
    if latest_tuple > current_tuple:
        # Update available!
        print("\n" + "="*70)
        print(f"🚀 A new version of Albert is available! (v{latest_version})")
        print("="*70)
        print(f"You are running: v{CURRENT_VERSION}")
        print(f"Latest version:  v{latest_version}")
        print("\nTo update, run:")
        print("  albert update")
        print("\nOr manually:")
        print("  curl -fsSL https://github.com/pimdewitte/albert/releases/download/stable/download_cli.sh | bash")
        print("="*70 + "\n")
        return True
    elif not silent:
        print(f"✓ Albert is up to date (v{CURRENT_VERSION})")
    
    return False


def check_on_startup():
    """Check for updates on startup (called from main entry points)"""
    # Only check in interactive sessions, not in CI/automated environments
    if os.environ.get('CI') or not sys.stdout.isatty():
        return
    
    # Check silently
    check_for_updates(silent=True) 