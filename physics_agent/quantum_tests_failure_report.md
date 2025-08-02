# Quantum Tests Failure Analysis Report

## Executive Summary

The g-2 Muon and Scattering Amplitudes tests are failing across ALL theories in the comprehensive test suite due to fundamental implementation issues. The validators exist but are incomplete stubs that cannot be instantiated, and the theories lack the required quantum interface methods.

## Root Causes Identified

### 1. Incomplete Validator Implementations

The validators in `validations/g_minus_2_validator.py` and `validations/scattering_amplitude_validator.py` are **stub implementations** that:

- **Missing Required Abstract Methods**: They don't implement the abstract methods required by `PredictionValidator`:
  - `fetch_dataset()`
  - `get_observational_data()`
  - `get_sota_benchmark()`

- **Missing Core Calculation Methods**: The g-2 validator references methods that don't exist:
  - `self.experimental_values` (undefined)
  - `self.calculate_one_loop_correction()` (undefined)

- **Cannot Be Instantiated**: When `test_comprehensive_final.py` tries to create `GMinus2Validator()`, it fails with:
  ```
  TypeError: Can't instantiate abstract class GMinus2Validator without an implementation 
  for abstract methods 'fetch_dataset', 'get_observational_data', 'get_sota_benchmark'
  ```

### 2. Theories Lack Quantum Interface

None of the tested theories implement the quantum methods that validators expect:
- `get_coupling_constants()` - Required for g-2 calculations
- `calculate_scattering_amplitude()` - Required for scattering tests

This is evident from the investigation output:
```
✓ Has get_coupling_constants: False
✓ Has calculate_scattering_amplitude: False
```

### 3. Test Functions Assume Working Validators

The test functions in `test_comprehensive_final.py` are simple wrappers that:
```python
def test_g_minus_2(theory):
    validator = GMinus2Validator()  # This fails!
    result = validator.validate(theory)
    return {...}
```

They don't handle the instantiation failure, leading to ERROR status in reports.

## Why Tests Show as "ERROR" Not "FAIL"

In the comprehensive report, all quantum tests show:
- Status: ERROR
- Solver: Failed
- Time: 0.000s

This indicates the tests are **crashing during setup** (validator instantiation) rather than running and failing validation.

## Contrast with Test File Success

The quantum validators created in `solver_tests/test_quantum_validators.py` work because:
1. They are **complete implementations** with all required methods
2. They use a **test-specific mixin** (`QuantumInterfaceMixin`) to add quantum methods to theories
3. They are **self-contained** and don't rely on external validator files

## Required Changes

### Option 1: Complete the Stub Validators
1. Implement all abstract methods in `g_minus_2_validator.py` and `scattering_amplitude_validator.py`
2. Add the missing calculation methods
3. Include proper experimental data storage

### Option 2: Copy Working Validators from Test
1. Move `GMinus2ValidatorTest` and `ScatteringAmplitudeValidatorTest` from the test file to the validations directory
2. Rename them to match expected class names
3. Ensure they properly subclass `PredictionValidator`

### Option 3: Add Quantum Interface to Theories
1. Create a quantum interface mixin or base class
2. Update relevant theories to implement quantum methods
3. Provide default/stub implementations for classical theories

### Option 4: Make Tests Optional/Conditional
1. Modify test functions to check if theory supports quantum interface
2. Return SKIP status for theories without quantum methods
3. Handle validator instantiation failures gracefully

## Recommended Approach

**Immediate Fix** (for comprehensive tests to pass):
1. Update test functions to handle instantiation failures:
   ```python
   def test_g_minus_2(theory):
       try:
           validator = GMinus2Validator()
       except TypeError:
           return {
               'name': 'g-2 Muon',
               'status': 'SKIP',
               'passed': False,
               'notes': 'Validator not fully implemented'
           }
   ```

**Long-term Solution**:
1. Complete the validator implementations using the working code from test file
2. Add quantum interface as optional mixin for theories that support it
3. Update test framework to distinguish between "not applicable" and "failed"

## Validation

The investigation script clearly shows:
- All theories fail with the same `TypeError` about abstract methods
- The validators cannot be imported from the test script due to relative import issues
- The working test validators use different class names (`GMinus2ValidatorTest` vs `GMinus2Validator`)

This confirms the validators in the main codebase are incomplete implementations that need to be finished before the quantum tests can run successfully.