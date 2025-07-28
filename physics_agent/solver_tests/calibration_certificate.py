#!/usr/bin/env python3
"""
Calibration Certificate System

Creates a "certificate" of calibration that guarantees:
1. Engine correctness
2. Solver accuracy  
3. Environment stability
4. Device performance benchmarks

This certificate is included in all run reports to ensure transparency.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class CalibrationCertificate:
    """Manages calibration certificates for physics runs."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the certificate manager."""
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'cache', 'certificates'
            )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.current_certificate = None
        self.certificate_path = os.path.join(self.cache_dir, 'current_certificate.json')
    
    def create_certificate(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a calibration certificate from test results.
        
        Args:
            calibration_results: Results from calibration tests
            
        Returns:
            Certificate dictionary with all validation info
        """
        # Extract test results
        tests = calibration_results.get('results', {})
        
        # Calculate overall health score
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test.get('passed', False))
        health_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Extract device recommendations if available
        device_benchmarks = tests.get('device_benchmarks', {})
        device_recommendations = None
        if device_benchmarks.get('passed') and 'results' in device_benchmarks:
            recommendations = device_benchmarks['results'].get('recommendations', {})
            device_recommendations = {
                'overall': recommendations.get('summary', {}),
                'per_test': recommendations.get('detailed', {})
            }
        
        # Create certificate
        certificate = {
            'certificate_id': self._generate_certificate_id(),
            'timestamp': datetime.now().isoformat(),
            'status': 'VALID' if health_score == 100 else 'WARNING' if health_score >= 80 else 'FAILED',
            'health_score': health_score,
            'tests_summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': total_tests - passed_tests
            },
            'test_details': {
                test_name: {
                    'passed': result.get('passed', False),
                    'error': result.get('error'),
                    'duration': calibration_results.get('duration', 0)
                }
                for test_name, result in tests.items()
            },
            'environment': {
                'platform': self._get_platform_info(),
                'device_recommendations': device_recommendations
            },
            'guarantees': self._generate_guarantees(tests)
        }
        
        # Save certificate
        self.current_certificate = certificate
        self._save_certificate(certificate)
        
        return certificate
    
    def _generate_certificate_id(self) -> str:
        """Generate a unique certificate ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information."""
        import platform
        import torch
        
        info = {
            'python_version': platform.python_version(),
            'os': f"{platform.system()} {platform.release()}",
            'architecture': platform.machine(),
            'pytorch_version': torch.__version__,
            'cuda_available': str(torch.cuda.is_available()),
            'mps_available': str(torch.backends.mps.is_available())
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
        
        return info
    
    def _generate_guarantees(self, tests: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate specific guarantees based on test results."""
        guarantees = {}
        
        # Engine correctness guarantee
        if tests.get('Geodesic Solver', {}).get('passed') and tests.get('Theory Engine', {}).get('passed'):
            guarantees['engine_correctness'] = {
                'status': 'GUARANTEED',
                'details': 'Geodesic integration and theory engine verified'
            }
        else:
            guarantees['engine_correctness'] = {
                'status': 'NOT_GUARANTEED',
                'details': 'Engine tests failed - results may be unreliable'
            }
        
        # Solver accuracy guarantee
        if tests.get('Basic Validation', {}).get('passed'):
            guarantees['solver_accuracy'] = {
                'status': 'GUARANTEED',
                'details': 'Solver validation tests passed'
            }
        else:
            guarantees['solver_accuracy'] = {
                'status': 'NOT_GUARANTEED',
                'details': 'Solver validation failed - numerical accuracy uncertain'
            }
        
        # Environment stability guarantee
        if tests.get('Environment Check', {}).get('passed'):
            guarantees['environment_stability'] = {
                'status': 'GUARANTEED',
                'details': 'All required dependencies and environment checks passed'
            }
        else:
            guarantees['environment_stability'] = {
                'status': 'NOT_GUARANTEED',
                'details': 'Environment issues detected - may affect results'
            }
        
        # Performance optimization guarantee
        if tests.get('device_benchmarks', {}).get('passed'):
            guarantees['performance_optimized'] = {
                'status': 'GUARANTEED',
                'details': 'Device benchmarks completed - using optimal configuration'
            }
        else:
            guarantees['performance_optimized'] = {
                'status': 'NOT_TESTED',
                'details': 'Device benchmarks not run - using default configuration'
            }
        
        return guarantees
    
    def _save_certificate(self, certificate: Dict[str, Any]):
        """Save certificate to disk."""
        # Save current certificate
        with open(self.certificate_path, 'w') as f:
            json.dump(certificate, f, indent=2)
        
        # Also save timestamped copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(
            self.cache_dir, 
            f"certificate_{timestamp}.json"
        )
        with open(archive_path, 'w') as f:
            json.dump(certificate, f, indent=2)
    
    def load_current_certificate(self) -> Optional[Dict[str, Any]]:
        """Load the current calibration certificate."""
        if os.path.exists(self.certificate_path):
            try:
                with open(self.certificate_path, 'r') as f:
                    self.current_certificate = json.load(f)
                return self.current_certificate
            except:
                return None
        return None
    
    def validate_certificate(self, max_age_hours: int = 24) -> bool:
        """
        Validate if the current certificate is still valid.
        
        Args:
            max_age_hours: Maximum age of certificate in hours
            
        Returns:
            True if certificate is valid and recent
        """
        if not self.current_certificate:
            self.load_current_certificate()
        
        if not self.current_certificate:
            return False
        
        # Check certificate status
        if self.current_certificate.get('status') == 'FAILED':
            return False
        
        # Check age
        try:
            cert_time = datetime.fromisoformat(self.current_certificate['timestamp'])
            age = datetime.now() - cert_time
            if age.total_seconds() > max_age_hours * 3600:
                return False
        except:
            return False
        
        return True
    
    def get_certificate_summary(self) -> str:
        """Get a human-readable summary of the certificate."""
        if not self.current_certificate:
            self.load_current_certificate()
        
        if not self.current_certificate:
            return "No calibration certificate available"
        
        cert = self.current_certificate
        lines = [
            "=== CALIBRATION CERTIFICATE ===",
            f"Certificate ID: {cert['certificate_id']}",
            f"Status: {cert['status']}",
            f"Health Score: {cert['health_score']:.1f}%",
            f"Generated: {cert['timestamp']}",
            "",
            "Guarantees:"
        ]
        
        for guarantee_name, guarantee in cert['guarantees'].items():
            status = guarantee['status']
            details = guarantee['details']
            lines.append(f"  {guarantee_name.replace('_', ' ').title()}: {status}")
            lines.append(f"    {details}")
        
        if cert.get('environment', {}).get('device_recommendations'):
            recs = cert['environment']['device_recommendations']['overall']
            lines.append("")
            lines.append(f"Device Recommendation: {'GPU' if recs.get('gpu_recommended') else 'CPU'}")
            lines.append(f"  {recs.get('reason', 'No recommendation available')}")
        
        return "\n".join(lines)
    
    def generate_html_badge(self) -> str:
        """Generate an HTML badge showing calibration status."""
        if not self.current_certificate:
            self.load_current_certificate()
        
        if not self.current_certificate:
            status = "UNKNOWN"
            color = "#999"
            health = "N/A"
        else:
            status = self.current_certificate['status']
            health = f"{self.current_certificate['health_score']:.0f}%"
            
            if status == 'VALID':
                color = "#4CAF50"  # Green
            elif status == 'WARNING':
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
        
        return f"""
        <div style="display: inline-block; background: {color}; color: white; padding: 8px 16px; 
                    border-radius: 4px; font-family: monospace; margin: 10px 0;">
            <strong>CALIBRATION: {status}</strong> | Health: {health}
        </div>
        """


# Global certificate manager
certificate_manager = CalibrationCertificate()


def create_calibration_certificate(calibration_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a calibration certificate from results."""
    return certificate_manager.create_certificate(calibration_results)


def get_current_certificate() -> Optional[Dict[str, Any]]:
    """Get the current calibration certificate."""
    return certificate_manager.load_current_certificate()


def validate_calibration() -> bool:
    """Check if current calibration is valid."""
    return certificate_manager.validate_certificate()


if __name__ == '__main__':
    # Test certificate generation
    test_results = {
        'results': {
            'Environment Check': {'passed': True},
            'Geodesic Solver': {'passed': True},
            'Theory Engine': {'passed': True},
            'Validator Registry': {'passed': True},
            'Basic Validation': {'passed': True},
            'device_benchmarks': {
                'passed': True,
                'results': {
                    'recommendations': {
                        'summary': {
                            'gpu_recommended': False,
                            'reason': 'CPU provides better precision/performance balance'
                        }
                    }
                }
            }
        },
        'duration': 1.7
    }
    
    cert = create_calibration_certificate(test_results)
    print(certificate_manager.get_certificate_summary())
    print("\nHTML Badge:")
    print(certificate_manager.generate_html_badge()) 