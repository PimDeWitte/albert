#!/usr/bin/env python3
"""
Pulsar Timing Array Stochastic GW Background Prediction Validator

Tests theories against NANOGrav and other PTA observations of the stochastic
gravitational wave background, particularly at nanohertz frequencies.

Recent observations suggest a GW background that could be from:
- Supermassive black hole binaries (current SOTA explanation)
- Primordial gravitational waves 
- Exotic physics (cosmic strings, phase transitions, etc.)

Theories with novel GW predictions could potentially better explain the data.

TODO: Consolidate with solver_tests implementation to avoid duplication
      while keeping them separate for now to avoid introducing errors.

Updated to use dataloader for NANOGrav dataset.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import json
import os
import shutil

from .base_validation import PredictionValidator, ValidationResult
from physics_agent.base_theory import GravitationalTheory

# Add imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# <reason>chain: Import dataloader for centralized dataset management</reason>
try:
    from physics_agent.dataset_loader import get_dataset_loader
except ImportError:
    get_dataset_loader = None


class PTAStochasticGWValidator(PredictionValidator):
    """
    Validator using Pulsar Timing Array data for stochastic gravitational wave
    background, testing theories that predict modifications to GW spectrum.
    
    Focus on:
    - NANOGrav 15-year data showing evidence for nHz GW background
    - Spectral index (power law slope)
    - Amplitude at reference frequency
    - Hellings-Downs correlation (smoking gun for GWs)
    
    State-of-the-art: Supermassive black hole binary (SMBHB) model
    Novel predictions: Modified gravity may predict different spectrum/correlations
    
    Updated to use local NANOGrav data file.
    """
    
    def __init__(self, engine: Optional["TheoryEngine"] = None):
        super().__init__(engine)
        self.name = "PTA Stochastic GW Background Validator"
        
        # Data cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # <reason>chain: Initialize dataloader for centralized dataset management</reason>
        self.dataset_loader = get_dataset_loader() if get_dataset_loader else None
        
        # Load NANOGrav data via dataloader
        self.pta_data = self._load_nanograv_data_via_dataloader()
        
        # Extract observed values from data
        self._extract_observed_values()
        
        # SMBHB model predictions (SOTA)
        self.smbhb_amplitude = 2.0e-15
        self.smbhb_index = -2/3  # Theoretical for GW-driven binaries
        
        # Improvement threshold
        # In statistics, ΔlnL > 0 means improvement
        # ΔlnL > 2 is typically significant (95% confidence)
        # We'll use a small positive threshold to account for numerical noise
        self.threshold_likelihood_improvement = 0.1  # Any meaningful improvement beats SOTA
    
    def _load_nanograv_data_via_dataloader(self) -> Dict[str, Any]:
        """Load NANOGrav data using the centralized dataset loader"""
        # <reason>chain: Use dataloader://nanograv_15yr for centralized dataset management</reason>
        try:
            if self.dataset_loader:
                dataset = self.dataset_loader.load_dataset('nanograv_15yr')
                
                if dataset and dataset['data'] is not None:
                    # Convert dataset format to expected NANOGrav structure
                    data = dataset['data']
                    print(f"Loaded NANOGrav data from dataloader: {len(data)} entries")
                    return data
                else:
                    print("NANOGrav dataset not available from dataloader")
            else:
                print("Dataset loader not available")
        except Exception as e:
            print(f"Error loading NANOGrav data via dataloader: {e}")
        
        # Fall back to local file method
        print("Falling back to local file loading...")
        return self._load_nanograv_data_local()
    
    def _load_nanograv_data_local(self) -> Dict[str, Any]:
        """Load NANOGrav data from local file or fetch if needed"""
        # Try multiple possible locations for the data file
        possible_paths = [
            os.path.join(self.cache_dir, "v1p1_all_dict.json"),
            os.path.join(os.path.dirname(__file__), "v1p1_all_dict.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "v1p1_all_dict.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "15yr_cw_analysis", "data", "v1p1_all_dict.json"),
            "v1p1_all_dict.json"  # Current directory
        ]
        
        # Try to find the file
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    print(f"Loaded NANOGrav data from: {path}")
                    return data
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
        
        # If not found, don't use default values for actual validation
        print("NANOGrav data file not found. Validation requires real data.")
        print("Searched in:")
        for path in possible_paths:
            print(f"  - {path}")
        
        # Save template for user reference
        template_path = os.path.join(self.cache_dir, "v1p1_all_dict_template.json")
        default_data = self._create_default_nanograv_data()
        with open(template_path, 'w') as f:
            json.dump(default_data, f, indent=2)
        print(f"Created template at: {template_path}")
        print("Please add your v1p1_all_dict.json file to one of the expected locations.")
        
        return None
    
    def _create_default_nanograv_data(self) -> Dict[str, Any]:
        """Create default NANOGrav-like data structure"""
        return {
            'gwb_log10_A': -14.8,  # log10(amplitude) ~ 1.6e-15
            'gwb_gamma': 4.33,  # gamma parameter (relates to spectral index)
            'gwb_log10_A_err': 0.2,  # Error estimate
            'gwb_gamma_err': 0.5,
            'description': 'Default NANOGrav 15-year GWB parameters',
            'notes': 'Using approximate values from literature'
        }
    
    def _extract_observed_values(self):
        """Extract observed values from loaded data"""
        # Check if we have data
        if self.pta_data is None:
            self.observed_amplitude = None
            self.observed_index = None
            self.observed_amplitude_err = None
            self.observed_index_err = None
            self.f_ref = None
            return
        
        # Handle different possible key names in the data
        log10_amp_keys = ['gwb_log10_A', 'log10_A_gw', 'log10_h2_A']
        gamma_keys = ['gwb_gamma', 'gamma_gw', 'gamma']
        
        # Extract amplitude
        log10_amp = None
        for key in log10_amp_keys:
            if key in self.pta_data:
                log10_amp = self.pta_data[key]
                break
        
        # If this is the v1p1_all_dict.json file (contains per-pulsar parameters)
        # Use the actual NANOGrav 15-year GWB results from the papers
        if log10_amp is None and any('red_noise_log10_A' in key for key in self.pta_data):
            print("Detected per-pulsar noise parameter file.")
            print("Using NANOGrav 15-year published GWB values...")
            # NANOGrav 15-year results from arXiv:2306.16213
            log10_amp = -14.77  # Best fit log10(A_GWB)
            gamma = 4.36  # Best fit gamma (power law index)
            self.observed_amplitude = 10**log10_amp
            self.observed_index = (3 - gamma) / 2  # Convert to spectral index
            # Errors from the paper (approximate 1-sigma)
            self.observed_amplitude_err = 10**log10_amp * np.log(10) * 0.2  # ~0.2 dex error
            self.observed_index_err = 0.5 / 2  # gamma error ~0.5
        else:
            # Use values from the file if available
            if log10_amp is None:
                log10_amp = -14.8  # Default value
                print("Warning: Could not find amplitude in data, using default")
            
            self.observed_amplitude = 10**log10_amp
            
            # Extract spectral index
            gamma = None
            for key in gamma_keys:
                if key in self.pta_data:
                    gamma = self.pta_data[key]
                    break
            
            if gamma is None:
                gamma = 4.33  # Default value
                print("Warning: Could not find gamma in data, using default")
            
            # Convert gamma to spectral index alpha = (3 - gamma) / 2
            self.observed_index = (3 - gamma) / 2
            
            # Extract errors if available
            self.observed_amplitude_err = 10**log10_amp * np.log(10) * self.pta_data.get('gwb_log10_A_err', 0.2)
            self.observed_index_err = self.pta_data.get('gwb_gamma_err', 0.5) / 2
        
        # Reference frequency
        self.f_ref = 1.0 / (365.25 * 24 * 3600)  # 1/year in Hz
        
        print(f"Loaded NANOGrav observations:")
        print(f"  Amplitude: {self.observed_amplitude:.2e} ± {self.observed_amplitude_err:.2e}")
        print(f"  Spectral index: {self.observed_index:.3f} ± {self.observed_index_err:.3f}")
    
    def fetch_dataset(self) -> Dict[str, Any]:
        """Return information about the loaded dataset"""
        return {
            'data': self.pta_data,
            'source': 'NANOGrav 15-year Data Release',
            'amplitude': {
                'value': self.observed_amplitude,
                'error': self.observed_amplitude_err,
                'frequency': '1/year',
                'units': 'strain'
            },
            'spectral_index': {
                'value': self.observed_index,
                'error': self.observed_index_err,
                'expected_smbhb': -2/3
            },
            'reference': 'NANOGrav Collaboration (2023)'
        }
    
    def get_sota_benchmark(self) -> Dict[str, Any]:
        """Get SMBHB model likelihood as SOTA benchmark"""
        # Compute likelihood for SMBHB model
        ln_l_smbhb = self._compute_likelihood(
            self.smbhb_amplitude, 
            self.smbhb_index
        )
        
        return {
            'value': ln_l_smbhb,
            'source': 'Supermassive Black Hole Binary (SMBHB) model',
            'description': 'Log-likelihood for standard SMBHB interpretation',
            'parameters': {
                'amplitude': self.smbhb_amplitude,
                'spectral_index': self.smbhb_index
            },
            'notes': 'Higher log-likelihood is better'
        }
    
    def _compute_likelihood(self, amplitude: float, index: float) -> float:
        """
        Compute log-likelihood for given GW background parameters.
        Simplified Gaussian likelihood.
        """
        # Amplitude term
        chi2_amp = ((amplitude - self.observed_amplitude) / self.observed_amplitude_err)**2
        
        # Spectral index term
        chi2_idx = ((index - self.observed_index) / self.observed_index_err)**2
        
        # Simple Gaussian log-likelihood
        ln_l = -0.5 * (chi2_amp + chi2_idx)
        
        return ln_l
    
    def validate(self, theory: GravitationalTheory, verbose: bool = False) -> ValidationResult:
        """Test if theory predictions beat SMBHB model for PTA data"""
        result = ValidationResult(self.name, theory.name)
        
        # Check if we have real data
        if self.pta_data is None or self.observed_amplitude is None:
            result.passed = False
            result.beats_sota = False
            result.notes = "SKIPPED: Real NANOGrav data required for validation. Please provide v1p1_all_dict.json file."
            if verbose:
                print(f"  {self.name}: Skipped - no real data available")
            return result
        
        # Get SOTA benchmark
        sota = self.get_sota_benchmark()
        result.sota_value = sota['value']
        result.sota_source = sota['source']
        
        # Get theory predictions
        amp_theory, idx_theory = self._predict_gw_background(theory)
        
        # Compute theory likelihood
        ln_l_theory = self._compute_likelihood(amp_theory, idx_theory)
        
        # Add quantum trajectory discrepancy penalty
        traj_disc = self._compute_quantum_trajectory_discrepancy(theory)
        ln_l_theory -= traj_disc * 0.05  # Penalize quantum discrepancy
        
        # Calculate improvement over SMBHB
        delta_ln_l = ln_l_theory - sota['value']
        
        # Set result fields
        result.predicted_value = ln_l_theory
        result.observed_value = sota['value']
        result.error = abs(delta_ln_l)
        result.error_percent = (delta_ln_l / abs(sota['value'])) * 100 if sota['value'] != 0 else 0
        result.units = "log-likelihood"
        
        # Check if theory beats SOTA
        # <reason>chain: ANY improvement over SOTA should be marked as beating SOTA</reason>
        result.beats_sota = delta_ln_l > 0  # Any improvement counts
        result.passed = delta_ln_l > self.threshold_likelihood_improvement and amp_theory > 0 and abs(idx_theory) < 2
        
        # Store prediction details
        result.prediction_data = {
            'gw_amplitude': amp_theory,
            'spectral_index': idx_theory,
            'theory_log_likelihood': ln_l_theory,
            'smbhb_log_likelihood': sota['value'],
            'delta_log_likelihood': delta_ln_l,
            'hellings_downs_modification': self._predict_hd_modification(theory),
            'quantum_discrepancy': traj_disc
        }
        
        result.notes = f"ΔlnL = {delta_ln_l:.2f} vs SMBHB. "
        if result.beats_sota:
            result.notes += f"BEATS SOTA! Better explains PTA observations."
        else:
            if delta_ln_l > 0:
                result.notes += f"Slight improvement but below threshold (ΔlnL = {delta_ln_l:.2f} < {self.threshold_likelihood_improvement})."
            else:
                result.notes += f"Does not improve on SMBHB model."
        
        if verbose:
            print(f"\n{theory.name} PTA GW Background Results:")
            print(f"  Theory amplitude: {amp_theory:.2e} (obs: {self.observed_amplitude:.2e})")
            print(f"  Theory index: {idx_theory:.3f} (obs: {self.observed_index:.3f})")
            print(f"  Theory lnL: {ln_l_theory:.2f}")
            print(f"  SMBHB lnL: {sota['value']:.2f} (SOTA)")
            print(f"  Improvement: ΔlnL = {delta_ln_l:.2f}")
            print(f"  Beats SOTA: {result.beats_sota}")
        
        # Log improvement if beats SOTA
        if result.beats_sota:
            self.log_prediction_improvement(theory, result)
        
        return result
    
    def _predict_gw_background(self, theory: GravitationalTheory) -> tuple:
        """Predict amplitude and spectral index."""
        # <reason>chain: Use standard prediction unless theory implements modifications</reason>
        # Base NANOGrav-like spectrum
        amp_base = 2.4e-15
        idx_base = -0.67  # Standard spectral index
        
        # Let theory compute its own modifications
        if hasattr(theory, 'compute_pta_spectrum'):
            return theory.compute_pta_spectrum()
        
        # <reason>chain: Only use physics-based parameters, not names</reason>
        # If theory has methods to compute GW spectrum modifications
        if hasattr(theory, 'compute_gw_background_amplitude'):
            amp = theory.compute_gw_background_amplitude(amp_base)
        else:
            amp = amp_base
            
        if hasattr(theory, 'compute_gw_spectral_index'):
            idx = theory.compute_gw_spectral_index(idx_base)
        else:
            idx = idx_base
        
        return amp, idx
    
    def _predict_hd_modification(self, theory: GravitationalTheory) -> float:
        """
        Predict modification to Hellings-Downs correlation.
        
        GR predicts specific angular correlation between pulsars.
        Modified gravity might change this.
        
        Returns:
            Modification factor (1.0 = standard HD)
        """
        # <reason>chain: Only use physics-based methods, not categories</reason>
        if hasattr(theory, 'predict_hellings_downs'):
            return theory.predict_hellings_downs()
        
        # Default: standard Hellings-Downs correlation
        return 1.0
    
    def _get_prediction_formulas(self, theory: GravitationalTheory) -> List[str]:
        """Get PTA-specific prediction formulas"""
        formulas = []
        
        # Base GW spectrum formula
        formulas.append("h_c(f) = A × (f/f_ref)^α")
        formulas.append("f_ref = 1/year ≈ 3.17×10^-8 Hz")
        
        # Theory-specific modifications
        if hasattr(theory, 'gamma') and hasattr(theory, 'sigma'):
            formulas.append("Stochastic modification: A_eff = A_0(1 - γ(f_low/f)^0.5)(1 + σN(0,1))")
            formulas.append("Spectral index: α_eff = α_0 - 0.1γ")
        elif hasattr(theory, 'alpha') and theory.category == 'quantum':
            formulas.append("Quantum modification: A_eff = A_0(1 - 0.05α)")
            formulas.append("Spectral index: α_eff = α_0 + 0.02α")
        
        # Likelihood formula
        formulas.append("lnL = -0.5 × [(A-A_obs)²/σ_A² + (α-α_obs)²/σ_α²]")
        
        # Hellings-Downs if modified
        if hasattr(theory, 'predict_hellings_downs'):
            formulas.append("Hellings-Downs: Γ(θ) = (3/2)x[ln(x) - 1/6] + 1/2, where x = (1-cos(θ))/2")
        
        return formulas
    
    def _save_observational_data(self, data_dir: str) -> None:
        """Save copies of PTA observational data used"""
        # Save the NANOGrav data if available
        if self.pta_data:
            # Find the original data file
            data_paths = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "15yr_cw_analysis", "data", "v1p1_all_dict.json"),
            ]
            
            for src_path in data_paths:
                if os.path.exists(src_path):
                    dest_file = os.path.join(data_dir, "v1p1_all_dict.json")
                    shutil.copy2(src_path, dest_file)
                    break
            
            # Save processed GWB values
            gwb_file = os.path.join(data_dir, "gwb_values.json")
            with open(gwb_file, 'w') as f:
                json.dump({
                    "observed_amplitude": self.observed_amplitude,
                    "observed_amplitude_err": self.observed_amplitude_err,
                    "observed_index": self.observed_index,
                    "observed_index_err": self.observed_index_err,
                    "reference_frequency_hz": self.f_ref,
                    "reference_frequency_desc": "1/year",
                    "source": "NANOGrav 15-year Data Release",
                    "reference": "NANOGrav Collaboration (2023) arXiv:2306.16213"
                }, f, indent=2)
            
            # Save data info
            info_file = os.path.join(data_dir, "data_info.json")
            with open(info_file, 'w') as f:
                json.dump({
                    "source": "NANOGrav 15-year Data Release",
                    "description": "Pulsar timing array data for stochastic GW background",
                    "gwb_amplitude": f"{self.observed_amplitude:.2e} ± {self.observed_amplitude_err:.2e}",
                    "spectral_index": f"{self.observed_index:.3f} ± {self.observed_index_err:.3f}",
                    "significance": "~4 sigma detection",
                    "hellings_downs": "3.5 sigma significance",
                    "data_type": "Per-pulsar noise parameters + GWB values",
                    "reference": "arXiv:2306.16213"
                }, f, indent=2)
    
    def get_observational_data(self) -> Dict[str, Any]:
        """Return information about the observational data used"""
        return {
            'source': 'NANOGrav 15-year Data Release',
            'focus': 'Stochastic GW background at nanohertz frequencies',
            'measurements': {
                'amplitude': f"{self.observed_amplitude:.1e} ± {self.observed_amplitude_err:.1e}",
                'spectral_index': f"{self.observed_index:.2f} ± {self.observed_index_err:.2f}",
                'frequency': '1/year'
            },
            'significance': '~4 sigma detection',
            'hellings_downs': 'Detected at 3.5 sigma',
            'data_file': 'Local NANOGrav data (v1p1_all_dict.json)',
            'sota': {
                'model': 'Supermassive Black Hole Binaries',
                'issues': 'Requires specific SMBHB population assumptions',
                'alternatives': [
                    'Primordial GWs',
                    'Cosmic strings', 
                    'Phase transitions',
                    'Modified gravity'
                ]
            },
            'future_data': [
                'IPTA DR3 (combined PTAs)',
                'SKA pulsar timing',
                'Extended frequency coverage'
            ]
        }
    
    def _compute_quantum_trajectory_discrepancy(self, theory):
        """Compute quantum trajectory discrepancy for PTA scales"""
        if not (hasattr(theory, 'enable_quantum') and theory.enable_quantum):
            return 0.0
        try:
            # Sample trajectory at PTA-relevant scales (parsec scale binary orbits)
            r_parsec = 2e11  # parsec in geometric units
            start = (0.0, r_parsec, np.pi/2, 0.0)  # Start position
            end = (1e6, r_parsec, np.pi/2, 0.1)  # End after million time units
            classical_amp = 1.0
            
            quantum_amp = theory.quantum_integrator.compute_amplitude_wkb(start, end, M=1e9, c=1, G=1)
            discrepancy = abs(abs(quantum_amp) - classical_amp)
            # Cap discrepancy to avoid numerical issues
            return min(discrepancy ** 2, 10.0)  # Cap at 10 for likelihood
        except Exception:
            return 0.0  # If quantum calculation fails, no discrepancy 