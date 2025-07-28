#!/usr/bin/env python3
"""
Central Validator Registry

This module provides a single source of truth for:
1. Which validators exist
2. Which validators have been tested in solver tests
3. Performance metrics for each validator
4. Test coverage status

A validator CANNOT be added to the main validation suite until it has been
tested against circularity in test_geodesic_validator_comparison.py
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class ValidatorRegistry:
    """Central registry for all validators in the physics agent system."""
    
    # List of validators that have been tested in solver tests
    TESTED_VALIDATORS = {
        # Constraint validators
        'ConservationValidator': {
            'category': 'constraint',
            'test_function': None,  # Tested indirectly
            'description': 'Tests energy and angular momentum conservation',
            'performance_metrics': {}
        },
        'MetricPropertiesValidator': {
            'category': 'constraint', 
            'test_function': None,  # Tested indirectly
            'description': 'Tests metric signature and asymptotic flatness',
            'performance_metrics': {}
        },
        
        # Observational validators with solver tests
        'MercuryPrecessionValidator': {
            'category': 'observational',
            'test_function': 'test_mercury_comparison',
            'description': 'Tests perihelion precession of Mercury',
            'performance_metrics': {}
        },
        'LightDeflectionValidator': {
            'category': 'observational',
            'test_function': 'test_light_deflection_comparison',
            'description': 'Tests light bending near the Sun',
            'performance_metrics': {}
        },
        'PpnValidator': {
            'category': 'observational',
            'test_function': 'test_ppn_comparison',
            'description': 'Tests PPN parameters gamma and beta',
            'performance_metrics': {}
        },
        'PhotonSphereValidator': {
            'category': 'observational',
            'test_function': 'test_photon_sphere_comparison',
            'description': 'Tests photon sphere radius and black hole shadow',
            'performance_metrics': {}
        },
        'GwValidator': {
            'category': 'observational',
            'test_function': 'test_gravitational_wave_inspiral',
            'description': 'Tests gravitational wave waveforms',
            'performance_metrics': {}
        },
        'COWInterferometryValidator': {
            'category': 'observational',
            'test_function': 'test_quantum_interferometry',
            'description': 'Tests quantum phase shifts in neutron interferometry',
            'performance_metrics': {}
        },
        'PsrJ0740Validator': {
            'category': 'observational',
            'test_function': 'test_psr_j0740_validation',
            'description': 'Tests Shapiro delay in PSR J0740+6620',
            'performance_metrics': {}
        },
        
        # Prediction validators
        'CMBPowerSpectrumValidator': {
            'category': 'prediction',
            'test_function': 'test_cmb_power_spectrum',
            'description': 'Tests CMB power spectrum against Planck data',
            'performance_metrics': {}
        },
        'PrimordialGWsValidator': {
            'category': 'prediction',
            'test_function': 'test_bicep_keck_primordial_gws',
            'description': 'Tests primordial gravitational wave constraints',
            'performance_metrics': {}
        }
    }
    
    # Validators that exist but are NOT tested (and thus cannot be used)
    UNTESTED_VALIDATORS = [
        'LagrangianValidator',
        'AtomInterferometryValidator',
        'QuantumClockValidator',
        'GravitationalDecoherenceValidator',
        'QuantumLagrangianGroundingValidator',
        'HawkingValidator',
        'CosmologyValidator',
        'RenormalizabilityValidator',
        'UnificationScaleValidator',
        'PTAStochasticGWValidator',
        'QEDPrecisionValidator'
    ]
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the validator registry."""
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'cache', 'validator_registry'
            )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.performance_file = os.path.join(self.cache_dir, 'performance_metrics.json')
        self.load_performance_metrics()
    
    def load_performance_metrics(self):
        """Load cached performance metrics."""
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                data = json.load(f)
                for validator, metrics in data.items():
                    if validator in self.TESTED_VALIDATORS:
                        self.TESTED_VALIDATORS[validator]['performance_metrics'] = metrics
    
    def save_performance_metrics(self):
        """Save performance metrics to cache."""
        data = {}
        for validator, info in self.TESTED_VALIDATORS.items():
            data[validator] = info['performance_metrics']
        
        with open(self.performance_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_performance_metric(self, validator_name: str, theory_name: str, 
                                 execution_time: float, loss: float, 
                                 status: str, additional_data: Dict[str, Any] = None):
        """Update performance metrics for a validator on a specific theory."""
        if validator_name not in self.TESTED_VALIDATORS:
            raise ValueError(f"Validator {validator_name} is not in the tested validators list")
        
        if theory_name not in self.TESTED_VALIDATORS[validator_name]['performance_metrics']:
            self.TESTED_VALIDATORS[validator_name]['performance_metrics'][theory_name] = []
        
        metric = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'loss': loss,
            'status': status
        }
        
        if additional_data:
            metric.update(additional_data)
        
        self.TESTED_VALIDATORS[validator_name]['performance_metrics'][theory_name].append(metric)
        self.save_performance_metrics()
    
    def get_tested_validators(self, category: Optional[str] = None) -> List[str]:
        """Get list of tested validators, optionally filtered by category."""
        if category:
            return [name for name, info in self.TESTED_VALIDATORS.items() 
                   if info['category'] == category]
        return list(self.TESTED_VALIDATORS.keys())
    
    def get_untested_validators(self) -> List[str]:
        """Get list of validators that exist but haven't been tested."""
        return self.UNTESTED_VALIDATORS.copy()
    
    def is_validator_tested(self, validator_name: str) -> bool:
        """Check if a validator has been tested in solver tests."""
        return validator_name in self.TESTED_VALIDATORS
    
    def get_validator_info(self, validator_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific validator."""
        return self.TESTED_VALIDATORS.get(validator_name)
    
    def get_performance_summary(self, validator_name: str) -> Dict[str, Any]:
        """Get performance summary for a validator across all theories."""
        if validator_name not in self.TESTED_VALIDATORS:
            return {}
        
        metrics = self.TESTED_VALIDATORS[validator_name]['performance_metrics']
        if not metrics:
            return {}
        
        summary = {
            'validator': validator_name,
            'category': self.TESTED_VALIDATORS[validator_name]['category'],
            'description': self.TESTED_VALIDATORS[validator_name]['description'],
            'theories_tested': len(metrics),
            'average_execution_time': 0,
            'success_rate': 0,
            'total_runs': 0
        }
        
        total_time = 0
        total_success = 0
        total_runs = 0
        
        for theory, runs in metrics.items():
            for run in runs:
                total_runs += 1
                total_time += run['execution_time']
                if run['status'] in ['PASS', 'WARNING']:
                    total_success += 1
        
        if total_runs > 0:
            summary['average_execution_time'] = total_time / total_runs
            summary['success_rate'] = total_success / total_runs
            summary['total_runs'] = total_runs
        
        return summary
    
    def generate_registry_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the validator registry."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_validators': len(self.TESTED_VALIDATORS) + len(self.UNTESTED_VALIDATORS),
            'tested_validators': len(self.TESTED_VALIDATORS),
            'untested_validators': len(self.UNTESTED_VALIDATORS),
            'categories': {
                'constraint': self.get_tested_validators('constraint'),
                'observational': self.get_tested_validators('observational'),
                'prediction': self.get_tested_validators('prediction')
            },
            'performance_summaries': {}
        }
        
        for validator in self.TESTED_VALIDATORS:
            report['performance_summaries'][validator] = self.get_performance_summary(validator)
        
        return report


# Global registry instance
validator_registry = ValidatorRegistry() 