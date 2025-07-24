#!/usr/bin/env python3
"""
Centralized Dataset Loader

Manages all datasets used in the gravity compression framework:
- Tracks dataset sources (remote URLs or local paths)
- Manages downloads and caching
- Provides standardized access to all data
- Maintains provenance records

<reason>chain: Einstein demanded absolute precision in data management - this ensures complete traceability</reason>
"""

import os
import json
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

from .constants import *


class DatasetLoader:
    """
    Central dataset management for all physics validation data.
    
    <reason>chain: Centralized data management ensures reproducibility and traceability</reason>
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Base directory for data storage. Defaults to physics_agent/data/dataloader
        """
        if data_dir is None:
            # Default to physics_agent/data/dataloader
            module_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(module_dir, 'data', 'dataloader')
        
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize dataset registry
        self._initialize_registry()
        
    def _initialize_registry(self):
        """Initialize the comprehensive dataset registry"""
        self.registry = {
            # ========== OBSERVATIONAL DATASETS ==========
            'planck_cmb_2018': {
                'name': 'Planck 2018 CMB TT Power Spectrum',
                'reason': 'CMB temperature anisotropies test early universe physics and quantum gravity effects',
                'remote_url': 'https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/previews/COM_PowerSpect_CMB-TT-binned_R3.01.txt',
                'local_path': 'planck_cmb_2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt',
                'format': 'text',
                'columns': ['l', 'D_l', 'err_low', 'err_high'],
                'reference': 'Planck Collaboration 2020, A&A 641, A6'
            },
            
            'nanograv_15yr': {
                'name': 'NANOGrav 15-year Pulsar Timing Array Data',
                'reason': 'Tests low-frequency gravitational wave background from supermassive black holes',
                'remote_url': 'https://data.nanograv.org/15yr/v1.1.0/15yr_cw_analysis/v1p1_all_dict.json',
                'local_path': 'nanograv_15yr/v1p1_all_dict.json',
                'format': 'json',
                'reference': 'NANOGrav Collaboration 2023, ApJL 951, L8'
            },
            
            'bicep_keck_2021': {
                'name': 'BICEP/Keck Array Primordial GW Constraints',
                'reason': 'Constrains tensor-to-scalar ratio and tests inflationary predictions',
                'remote_url': 'http://bicepkeck.org/bk18_2021_release/BK18_r_likelihood.txt',
                'local_path': 'bicep_keck_2021/BK18_r_likelihood.txt',
                'format': 'text',
                'columns': ['r', 'minus_2_ln_L'],
                'reference': 'BICEP/Keck Collaboration 2022, Phys. Rev. Lett. 129, 201301'
            },
            
            # ========== HARDCODED EXPERIMENTAL VALUES ==========
            'mercury_perihelion': {
                'name': 'Mercury Perihelion Advance',
                'reason': 'Classic GR test of post-Newtonian corrections in planetary orbits',
                'remote_url': None,  # Hardcoded from literature
                'local_path': 'mercury_perihelion/data.json',
                'format': 'json',
                'data': MERCURY_PERIHELION_ADVANCE,
                'reference': 'Shapiro et al. (1976), improved by MESSENGER mission'
            },
            
            'solar_light_deflection': {
                'name': 'Solar Light Deflection',
                'reason': 'Tests null geodesics in curved spacetime, validates gravitational lensing',
                'remote_url': None,
                'local_path': 'solar_light_deflection/data.json',
                'format': 'json',
                'data': SOLAR_LIGHT_DEFLECTION,
                'reference': 'Shapiro et al. (2004), Phys. Rev. Lett. 92, 121101'
            },
            
            'shapiro_delay': {
                'name': 'Shapiro Time Delay (Cassini)',
                'reason': 'Most precise test of GR time delay, PPN parameter gamma',
                'remote_url': None,
                'local_path': 'shapiro_delay/data.json',
                'format': 'json',
                'data': SHAPIRO_TIME_DELAY,
                'reference': 'Bertotti et al. (2003), Cassini spacecraft'
            },
            
            'lunar_laser_ranging': {
                'name': 'Lunar Laser Ranging',
                'reason': 'Tests strong equivalence principle and PPN beta parameter',
                'remote_url': None,
                'local_path': 'lunar_laser_ranging/data.json',
                'format': 'json',
                'data': LUNAR_LASER_RANGING,
                'reference': 'Williams et al. (2012), Class. Quantum Grav. 29, 184004'
            },
            
            # ========== QUANTUM GRAVITY EXPERIMENTS ==========
            'cow_interferometry': {
                'name': 'COW Neutron Interferometry',
                'reason': 'Tests quantum effects in gravity, gravitational phase shift',
                'remote_url': None,
                'local_path': 'cow_interferometry/data.json',
                'format': 'json',
                'data': COW_INTERFEROMETRY,
                'reference': 'Colella, Overhauser & Werner (1975), Phys. Rev. Lett. 34, 1472'
            },
            
            'atom_interferometry': {
                'name': 'Atom Interferometry Gravitational Redshift',
                'reason': 'Ultra-precise gravitational redshift at meter scales',
                'remote_url': None,
                'local_path': 'atom_interferometry/data.json',
                'format': 'json',
                'data': ATOM_INTERFEROMETRY,
                'reference': 'Müller et al. (2010), Nature 463, 926'
            },
            
            'quantum_clock': {
                'name': 'Quantum Clock Time Dilation',
                'reason': 'Tests gravitational time dilation at quantum scales',
                'remote_url': None,
                'local_path': 'quantum_clock/data.json',
                'format': 'json',
                'data': QUANTUM_CLOCK,
                'reference': 'Chou et al. (2010), Science 329, 1630'
            },
            
            'gravitational_decoherence': {
                'name': 'Gravitational Decoherence Bounds',
                'reason': 'Tests gravity-induced collapse models and quantum decoherence',
                'remote_url': None,
                'local_path': 'gravitational_decoherence/data.json',
                'format': 'json',
                'data': GRAVITATIONAL_DECOHERENCE,
                'reference': 'Bassi et al. (2013), Class. Quantum Grav. 30, 035002'
            },
            
            # ========== PULSAR DATASETS ==========
            'psr_j0740_6620': {
                'name': 'PSR J0740+6620 Binary Pulsar',
                'reason': 'Most massive neutron star, tests GR in strong field with Shapiro delay',
                'remote_url': None,
                'local_path': 'psr_j0740_6620/data.json',
                'format': 'json',
                'data': {
                    'mass_companion': 2.08 * SOLAR_MASS,
                    'mass_companion_err': 0.004 * SOLAR_MASS,
                    'mass_pulsar': 0.25 * SOLAR_MASS,
                    'mass_pulsar_err': 0.01 * SOLAR_MASS,
                    'orbital_period': 4.77 * 86400,  # days to seconds
                    'eccentricity': 0.0,
                    'inclination': 87.0,  # degrees
                    'rms_residual': 0.18e-6  # microseconds
                },
                'reference': 'Fonseca et al. (2021), ApJL 915, L12'
            },
            
            'psr_j0952_0607': {
                'name': 'PSR J0952-0607 Black Widow Pulsar',
                'reason': 'Most massive neutron star known, tests equation of state limits',
                'remote_url': None,
                'local_path': 'psr_j0952_0607/data.json',
                'format': 'json',
                'data': PSR_J0952_0607,
                'reference': 'Romani et al. (2022), ApJ Lett. 934, L17'
            },
            
            'psr_b1913_16': {
                'name': 'PSR B1913+16 Hulse-Taylor Binary',
                'reason': 'Nobel Prize system, orbital decay confirms GW emission',
                'remote_url': None,
                'local_path': 'psr_b1913_16/data.json',
                'format': 'json',
                'data': PSR_B1913_16,
                'reference': 'Weisberg & Huang (2016), ApJ 829, 55'
            },
            
            # ========== COSMOLOGICAL PARAMETERS ==========
            'planck_cosmology_2018': {
                'name': 'Planck 2018 Cosmological Parameters',
                'reason': 'Comprehensive cosmological model parameters including CMB anomalies',
                'remote_url': None,
                'local_path': 'planck_cosmology_2018/parameters.json',
                'format': 'json',
                'data': PLANCK_COSMOLOGY,
                'reference': 'Planck Collaboration 2020, A&A 641, A6'
            },
            
            # ========== TEST CONFIGURATIONS ==========
            'test_radii': {
                'name': 'Standard Test Radii',
                'reason': 'Standard radii for metric testing in Schwarzschild units',
                'remote_url': None,
                'local_path': 'test_configurations/test_radii.json',
                'format': 'json',
                'data': {'factors': TEST_RADII_FACTORS},
                'reference': 'Internal standard configuration'
            },
            
            'standard_orbits': {
                'name': 'Standard Orbit Parameters',
                'reason': 'Standard orbital configurations for conservation tests',
                'remote_url': None,
                'local_path': 'test_configurations/standard_orbits.json',
                'format': 'json',
                'data': STANDARD_ORBITS,
                'reference': 'Internal standard configuration'
            }
        }
    
    def get_dataset_path(self, dataset_id: str) -> str:
        """Get the full local path for a dataset"""
        if dataset_id not in self.registry:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        
        info = self.registry[dataset_id]
        return os.path.join(self.data_dir, info['local_path'])
    
    def get_dataset_uri(self, dataset_id: str) -> str:
        """Get a URI for the dataset that can be used in documentation"""
        info = self.registry[dataset_id]
        
        # For hardcoded constants, return the code reference
        if not info['remote_url'] and dataset_id in [
            'mercury_perihelion', 'solar_light_deflection', 'shapiro_delay',
            'lunar_laser_ranging', 'cow_interferometry', 'atom_interferometry',
            'quantum_clock', 'gravitational_decoherence', 'psr_j0952_0607',
            'psr_b1913_16', 'planck_cosmology_2018', 'test_radii', 'standard_orbits'
        ]:
            # Map dataset ID to constant name
            constant_map = {
                'mercury_perihelion': 'MERCURY_PERIHELION_ADVANCE',
                'solar_light_deflection': 'SOLAR_LIGHT_DEFLECTION',
                'shapiro_delay': 'SHAPIRO_TIME_DELAY',
                'lunar_laser_ranging': 'LUNAR_LASER_RANGING',
                'cow_interferometry': 'COW_INTERFEROMETRY',
                'atom_interferometry': 'ATOM_INTERFEROMETRY',
                'quantum_clock': 'QUANTUM_CLOCK',
                'gravitational_decoherence': 'GRAVITATIONAL_DECOHERENCE',
                'psr_j0952_0607': 'PSR_J0952_0607',
                'psr_b1913_16': 'PSR_B1913_16',
                'planck_cosmology_2018': 'PLANCK_COSMOLOGY',
                'test_radii': 'TEST_RADII_FACTORS',
                'standard_orbits': 'STANDARD_ORBITS'
            }
            constant_name = constant_map.get(dataset_id, dataset_id.upper())
            return f"constants.py::{constant_name}"
        
        # For remote datasets, use dataloader URI
        return f"dataloader://{dataset_id}"
    
    def load_dataset(self, dataset_id: str, force_download: bool = False) -> Dict[str, Any]:
        """
        Load a dataset, downloading if necessary.
        
        Args:
            dataset_id: Dataset identifier from registry
            force_download: Force re-download even if cached
            
        Returns:
            Dictionary containing the dataset and metadata
        """
        if dataset_id not in self.registry:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        
        info = self.registry[dataset_id]
        local_path = self.get_dataset_path(dataset_id)
        dataset_dir = os.path.dirname(local_path)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Check if we need to download
        if info['remote_url'] and (force_download or not os.path.exists(local_path)):
            self._download_dataset(dataset_id)
        elif not info['remote_url'] and not os.path.exists(local_path):
            # For hardcoded data, create the file
            self._save_hardcoded_data(dataset_id)
        
        # Update source tracking
        self._update_source_tracking(dataset_id)
        
        # Load the data
        data = self._load_data_file(local_path, info['format'])
        
        return {
            'data': data,
            'metadata': info,
            'local_path': local_path,
            'uri': self.get_dataset_uri(dataset_id)
        }
    
    def _download_dataset(self, dataset_id: str):
        """Download a dataset from its remote URL"""
        info = self.registry[dataset_id]
        url = info['remote_url']
        local_path = self.get_dataset_path(dataset_id)
        
        print(f"Downloading {info['name']} from {url}...")
        
        try:
            # Create request with headers
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
                
                # Save to file
                with open(local_path, 'wb') as f:
                    f.write(data)
                
                print(f"Downloaded {len(data)} bytes to {local_path}")
                
        except Exception as e:
            print(f"Failed to download {dataset_id}: {e}")
            raise
    
    def _save_hardcoded_data(self, dataset_id: str):
        """Save hardcoded data to file"""
        info = self.registry[dataset_id]
        local_path = self.get_dataset_path(dataset_id)
        
        if 'data' in info:
            with open(local_path, 'w') as f:
                json.dump(info['data'], f, indent=2)
            print(f"Saved hardcoded data for {info['name']} to {local_path}")
    
    def _update_source_tracking(self, dataset_id: str):
        """Update the source.txt file for a dataset"""
        info = self.registry[dataset_id]
        dataset_dir = os.path.dirname(self.get_dataset_path(dataset_id))
        source_file = os.path.join(dataset_dir, 'source.txt')
        
        source_info = {
            'dataset_id': dataset_id,
            'name': info['name'],
            'last_updated': datetime.now().isoformat(),
            'source': info['remote_url'] or 'hardcoded from constants.py',
            'reference': info['reference']
        }
        
        with open(source_file, 'w') as f:
            json.dump(source_info, f, indent=2)
    
    def _load_data_file(self, path: str, format: str) -> Any:
        """Load data from file based on format"""
        if format == 'json':
            with open(path, 'r') as f:
                return json.load(f)
        elif format == 'text':
            # For text files, return as numpy array if numeric
            data = []
            with open(path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        try:
                            values = [float(x) for x in line.split()]
                            data.append(values)
                        except ValueError:
                            pass
            return np.array(data) if data else None
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def get_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered datasets"""
        return {
            dataset_id: {
                'name': info['name'],
                'reason': info['reason'],
                'source': info['remote_url'] or 'hardcoded',
                'uri': self.get_dataset_uri(dataset_id),
                'local_path': os.path.join('data/dataloader', info['local_path'])
            }
            for dataset_id, info in self.registry.items()
        }
    
    def verify_all_datasets(self) -> Dict[str, bool]:
        """Verify all datasets are available locally"""
        results = {}
        for dataset_id in self.registry:
            local_path = self.get_dataset_path(dataset_id)
            results[dataset_id] = os.path.exists(local_path)
        return results
    
    def download_all_remote_datasets(self):
        """Download all remote datasets that aren't already cached"""
        for dataset_id, info in self.registry.items():
            if info['remote_url']:
                local_path = self.get_dataset_path(dataset_id)
                if not os.path.exists(local_path):
                    try:
                        self._download_dataset(dataset_id)
                    except Exception as e:
                        print(f"Failed to download {dataset_id}: {e}")
    
    def create_all_hardcoded_datasets(self):
        """Create all hardcoded dataset files"""
        for dataset_id, info in self.registry.items():
            if not info['remote_url'] and 'data' in info:
                local_path = self.get_dataset_path(dataset_id)
                if not os.path.exists(local_path):
                    self._save_hardcoded_data(dataset_id)
                    self._update_source_tracking(dataset_id)


# Singleton instance
_dataset_loader = None

def get_dataset_loader(data_dir: Optional[str] = None) -> DatasetLoader:
    """Get the singleton DatasetLoader instance"""
    global _dataset_loader
    if _dataset_loader is None:
        _dataset_loader = DatasetLoader(data_dir)
    return _dataset_loader 