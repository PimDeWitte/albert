# Comprehensive Test is Now Default

## Summary

The comprehensive validation test now runs **by default** when executing the theory engine. This change prioritizes quick scientific validation over lengthy trajectory simulations.

## What This Means

### Before (Old Behavior)
```bash
python -m physics_agent.theory_engine_core  # Ran full trajectory simulations
```

### Now (New Behavior) 
```bash
python -m physics_agent.theory_engine_core  # Runs comprehensive validation test only
```

## Key Commands

```bash
# Default: Run comprehensive test only (NEW!)
python -m physics_agent.theory_engine_core

# Run test + continue with simulations
python -m physics_agent.theory_engine_core --continue-after-test

# Skip test (use old behavior)
python -m physics_agent.theory_engine_core --skip-comprehensive-test
```

## Why This Change?

1. **Immediate Results** - Get theory rankings in minutes, not hours
2. **Scientific Rigor** - 12 standardized tests per theory
3. **Resource Efficient** - Don't waste compute on invalid theories
4. **Better UX** - HTML report provides clear, actionable results

## Output

- **HTML Report**: Beautiful scientific scorecard
- **Rankings**: Both analytical-only and combined scores
- **Timing**: Accurate performance metrics (no more 0.0ms confusion)
- **Location**: `physics_agent/reports/latest_comprehensive_validation.html`

This change makes theory validation more accessible and scientific, providing researchers with immediate feedback on theory validity.