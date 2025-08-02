# Migration Guide: Comprehensive Test as Default

## What Changed

As of this update, the theory engine now runs the **comprehensive validation test by default** instead of full trajectory simulations.

## Quick Reference

| Old Command | New Equivalent |
|-------------|----------------|
| `python -m physics_agent.theory_engine_core` | `python -m physics_agent.theory_engine_core --skip-comprehensive-test` |
| `python -m physics_agent.theory_engine_core --comprehensive-test` | `python -m physics_agent.theory_engine_core` |
| `python -m physics_agent.theory_engine_core --comprehensive-test-only` | `python -m physics_agent.theory_engine_core` |

## Why This Change?

1. **Faster initial assessment** - Comprehensive test runs in minutes, not hours
2. **Standardized evaluation** - All theories tested with same criteria
3. **Scientific rigor** - 12 different validation tests per theory
4. **Better resource usage** - Avoid running expensive simulations on invalid theories

## Common Scenarios

### "I just want to see theory rankings"
```bash
# NEW DEFAULT - This is what you want!
python -m physics_agent.theory_engine_core
```

### "I need full trajectory simulations"
```bash
# Option 1: Run tests first, then continue
python -m physics_agent.theory_engine_core --continue-after-test

# Option 2: Skip tests entirely (not recommended)
python -m physics_agent.theory_engine_core --skip-comprehensive-test
```

### "I want to test specific theories"
```bash
# Tests still respect theory filters
python -m physics_agent.theory_engine_core --theories "Kerr,Quantum Corrected"
```

### "I need the old leaderboard format"
```bash
# Run full simulation to generate traditional leaderboard
python -m physics_agent.theory_engine_core --skip-comprehensive-test --final
```

## Benefits of New Default

1. **Immediate feedback** - Know which theories pass/fail in minutes
2. **HTML scorecard** - Beautiful report with all test details
3. **Computational savings** - Don't waste GPU time on failing theories
4. **Consistent metrics** - Same tests for all theories

## Output Location

- HTML Report: `comprehensive_theory_validation_[timestamp].html`
- Latest: `physics_agent/reports/latest_comprehensive_validation.html`
- JSON Data: `theory_validation_comprehensive_[timestamp].json`

## Need Help?

The comprehensive test includes:
- 7 analytical validators
- 5 solver-based tests
- Performance metrics
- Loss comparisons

View the HTML report for complete details on each theory's performance.