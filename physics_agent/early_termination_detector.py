#!/usr/bin/env python3
"""
Early termination detector for stuck or diverging particles.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional

class TrajectoryMonitor:
    """Monitor trajectory for early termination conditions."""
    
    def __init__(self, 
                 r0: float,
                 length_scale: float,
                 horizon_threshold: float = 2.1,
                 escape_threshold: float = 1000.0,
                 stuck_threshold_ratio: float = 1e-6,
                 check_interval: int = 10):
        """
        Initialize trajectory monitor.
        
        Args:
            r0: Initial radius in SI units
            length_scale: GM/c^2 for unit conversion
            horizon_threshold: Minimum r in units of M (default 2.1M)
            escape_threshold: Maximum r in units of M (default 1000M)
            stuck_threshold_ratio: Motion threshold as fraction of r0
            check_interval: Check every N steps
        """
        self.r0 = r0
        self.length_scale = length_scale
        self.horizon_threshold = horizon_threshold
        self.escape_threshold = escape_threshold
        self.stuck_threshold = stuck_threshold_ratio * r0
        self.check_interval = check_interval
        
        # History for stuck detection
        self.position_history: List[Tuple[float, float, float]] = []
        self.max_history_size = 50
        
    def check_trajectory(self, step: int, y: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """
        Check if trajectory should be terminated early.
        
        Args:
            step: Current integration step
            y: Current state [t, r, theta, phi, ...] in SI units
            
        Returns:
            (should_terminate, reason)
        """
        # Only check at intervals
        if step % self.check_interval != 0:
            return False, None
            
        r_si = y[1].item() if torch.is_tensor(y[1]) else y[1]
        r_geom = r_si / self.length_scale
        
        # Check 1: Horizon crossing
        if r_geom < self.horizon_threshold:
            return True, f"Crossed horizon at r={r_geom:.2f}M"
            
        # Check 2: Escape to infinity
        if r_geom > self.escape_threshold:
            return True, f"Escaped to r={r_geom:.1f}M"
            
        # Check 3: Stuck detection (no motion)
        if step >= 3 * self.check_interval:
            theta = y[2].item() if len(y) > 2 else np.pi/2
            phi = y[3].item() if len(y) > 3 else 0.0
            
            # Convert to Cartesian for motion detection
            x = r_si * np.sin(theta) * np.cos(phi)
            y_cart = r_si * np.sin(theta) * np.sin(phi)
            z = r_si * np.cos(theta)
            
            current_pos = (x, y_cart, z)
            self.position_history.append(current_pos)
            
            # Keep history size limited
            if len(self.position_history) > self.max_history_size:
                self.position_history.pop(0)
                
            # Check if particle has moved significantly
            if len(self.position_history) >= 3:
                # Compare current position to older positions
                total_motion = 0.0
                for old_pos in self.position_history[:-1]:
                    dx = current_pos[0] - old_pos[0]
                    dy = current_pos[1] - old_pos[1]
                    dz = current_pos[2] - old_pos[2]
                    total_motion += np.sqrt(dx**2 + dy**2 + dz**2)
                    
                avg_motion = total_motion / len(self.position_history)
                
                if avg_motion < self.stuck_threshold:
                    return True, f"Particle stuck (motion < {self.stuck_threshold:.2e} m)"
                    
        return False, None
        
    def get_diagnostics(self) -> dict:
        """Get diagnostic information about the trajectory."""
        if not self.position_history:
            return {}
            
        positions = np.array(self.position_history)
        r_values = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
        
        return {
            'r_min': r_values.min(),
            'r_max': r_values.max(),
            'r_std': r_values.std(),
            'num_checks': len(self.position_history),
            'avg_r_geom': r_values.mean() / self.length_scale
        }


def test_monitor():
    """Test the trajectory monitor."""
    print("Testing TrajectoryMonitor...\n")
    
    # Test parameters (primordial mini black hole)
    length_scale = 7.426e-13  # GM/c^2
    r0 = 10 * length_scale    # 10M
    
    monitor = TrajectoryMonitor(r0, length_scale)
    
    # Test case 1: Normal circular orbit
    print("Test 1: Normal circular orbit")
    for i in range(50):
        phi = i * 0.1
        y = torch.tensor([i*1e-22, r0, np.pi/2, phi])
        should_stop, reason = monitor.check_trajectory(i, y)
        if should_stop:
            print(f"  Stopped at step {i}: {reason}")
            break
    else:
        print("  ✓ Completed normally")
        
    # Test case 2: Stuck particle
    print("\nTest 2: Stuck particle")
    monitor2 = TrajectoryMonitor(r0, length_scale, stuck_threshold_ratio=1e-4)
    for i in range(50):
        # Particle doesn't move
        y = torch.tensor([i*1e-22, r0, np.pi/2, 0.0])
        should_stop, reason = monitor2.check_trajectory(i, y)
        if should_stop:
            print(f"  ✓ Detected stuck at step {i}: {reason}")
            break
    else:
        print("  ✗ Failed to detect stuck particle")
        
    # Test case 3: Horizon crossing
    print("\nTest 3: Horizon crossing")
    monitor3 = TrajectoryMonitor(r0, length_scale)
    for i in range(50):
        # Particle falls inward
        r = r0 * (1 - 0.02 * i)
        y = torch.tensor([i*1e-22, r, np.pi/2, i*0.1])
        should_stop, reason = monitor3.check_trajectory(i, y)
        if should_stop:
            print(f"  ✓ Detected at step {i}: {reason}")
            print(f"    Final r = {r/length_scale:.2f}M")
            break
            
    print("\nDiagnostics:")
    print(monitor3.get_diagnostics())


if __name__ == "__main__":
    test_monitor()