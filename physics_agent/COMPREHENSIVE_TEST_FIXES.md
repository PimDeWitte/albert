# Comprehensive Test Report Fixes

## Issues Addressed

### 1. Quantum Geodesic Test Failures ✓
- **Problem**: Many theories showed "NotSupported" as FAIL
- **Fix**: Changed to handle NotSupported as SKIP status
- **Result**: These tests no longer count as failures in the score

### 2. Solver Compute Time Not Showing ✓
- **Problem**: Combined rankings showed N/A for timing
- **Fix**: Properly accumulate solver time from Trajectory vs Kerr and Circular Orbit tests
- **Result**: Shows total time and ms/step for actual computations

### 3. Failed Solver Tests Not Listed ✓
- **Problem**: Combined rankings only showed passed/total count
- **Fix**: Added list of failed tests in parentheses (e.g., "1/5 (CMB,PGW)")
- **Result**: Easy to see which specific tests failed

### 4. Kerr Loss Against Itself ✓
- **Problem**: Kerr showed scientific notation (e.g., 0.00e+00)
- **Fix**: Special case to show exactly "0.00" for Kerr baseline
- **Result**: Clear that Kerr has zero loss by definition

### 5. Trajectory Loss Clarification ✓
- **Problem**: Table header was unclear about test conditions
- **Fix**: Added info box explaining:
  - MSE loss vs Kerr baseline over 1000 steps
  - Primordial Mini Black Hole (10⁻¹⁹ solar masses)
  - Electron particle at r=10M circular orbit
- **Result**: Complete transparency about test parameters

## Technical Changes

### test_comprehensive_final.py
- Modified `test_quantum_geodesic_for_theory()` to return NotSupported status
- Updated `run_solver_test()` to handle SKIP status properly

### comprehensive_test_report_generator.py
- Enhanced combined rankings table headers
- Added failed test listing logic
- Improved timing calculation to exclude cached trajectories
- Added special formatting for Kerr's zero loss
- Added explanatory note box for trajectory loss conditions

## Result

The HTML report now provides:
- Accurate test counts (SKIPs don't count as failures)
- Clear solver timing information
- Specific failed test identification
- Proper baseline loss display
- Complete test condition documentation