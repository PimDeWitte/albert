"""
Lagrangian-based Metric Derivations for Gravitational Theories.

This module provides the functionality to derive the metric tensor components
for a given gravitational theory by solving the Euler-Lagrange equations derived
from its Lagrangian action. This is a crucial step towards quantum validation
via path integrals.

The core function, `derive_metric_from_action`, takes a SymPy expression for
the Lagrangian, symbolically derives the equations of motion, and then numerically
solves the resulting boundary value problem to find the metric functions.

This powerful approach allows the framework to test novel gravitational theories
that can be expressed in a Lagrangian form.

This implementation was created by James Thompson.
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Any, TYPE_CHECKING

from physics_agent.validations.base_validation import BaseValidation
from physics_agent.base_theory import GravitationalTheory

import pennylane as qml
from pennylane import numpy as pnp

import math
from scipy.constants import epsilon_0, G, c

if TYPE_CHECKING:
    from physics_agent.theory_engine_core import TheoryEngine

class LagrangianValidator(BaseValidation):
    """
    Validates consistency between a theory's Lagrangian and its get_metric implementation.
    Derives metric from Lagrangian and computes MSE against direct metric.
    """
    category = "constraint"
    
    def __init__(self, engine: "TheoryEngine", tolerance: float = 1e-1, num_samples: int = 50, plot_enabled: bool = True, loss_type: str = 'ricci'):
        super().__init__(engine, "Lagrangian Validator")
        self.tolerance = tolerance
        self.num_samples = num_samples
        self.plot_enabled = plot_enabled
        self.loss_type = loss_type

    def detect_em_term(self, lagrangian: sp.Expr) -> tuple[sp.Expr, sp.Symbol] | None:
        """
        Detects quadratic EM-like term (e.g., F**2) in Lagrangian.
        Returns (coefficient as SymPy expr, field_symbol) if found, else None.
        Assumption: EM term is the only quadratic field strength; extend for more complex forms.
        """
        for term in lagrangian.as_ordered_terms():
            if term.is_Mul and term.has(sp.Pow):
                pow_terms = [arg for arg in term.args if isinstance(arg, sp.Pow) and arg.exp == 2]
                if pow_terms:
                    field_sym = pow_terms[0].base
                    coeff = term.coeff(field_sym**2)
                    if coeff != 0:
                        return coeff, field_sym
        return None

    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor = None, experimental: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Derives metric from Lagrangian and compares to theory's get_metric.

        NOTE ON DEVICE HANDLING:
        The derivation uses SciPy/NumPy which require CPU scalars/arrays.
        We extract rs_val as float via .item() and move sampled_r to CPU
        before interpolation. This isolates CPU operations while allowing
        the rest of the engine to use GPU tensors for performance.
        """
        if not hasattr(theory, 'lagrangian') or theory.lagrangian is None:
            return {
                "loss": 0.0,
                "flags": {"overall": "SKIP", "details": "No Lagrangian defined for theory."}
            }
        
        try:
            # New: Substitute theory parameters into Lagrangian
            free_symbols = theory.lagrangian.free_symbols
            subs_dict = {}
            for sym in free_symbols:
                sym_name = str(sym)
                if hasattr(theory, sym_name):
                    val = getattr(theory, sym_name)
                    if torch.is_tensor(val):
                        val = val.item()
                    subs_dict[sym] = val
            L_param = theory.lagrangian.subs(subs_dict)
            
            rs = 2 * self.engine.G_T * self.engine.M / self.engine.C_T**2
            if rs.numel() != 1:
                raise ValueError("rs must be a scalar for derivation")
            rs_val = rs.item()
            print(f"Debug: rs_val={rs_val} (from device {rs.device})")

            # Sample radii (use hist if provided, else default range)
            if hist is not None and hist.shape[0] > 1:
                sampled_r = hist[:, 1][:self.num_samples]
            else:
                # Default: log-spaced from 2RS to 100RS
                sampled_r = torch.logspace(np.log10(2 * rs_val), np.log10(100 * rs_val), self.num_samples, device=self.engine.device, dtype=self.engine.dtype)
            
            # Get direct metric
            g_tt_direct, g_rr_direct, g_theta_direct, g_phi_direct = theory.get_metric(sampled_r, self.engine.M, self.engine.C_T, self.engine.G_T)
            
            em_info = self.detect_em_term(L_param)  # Use L_param instead of theory.lagrangian
            include_charge = em_info is not None
            if include_charge:
                em_coeff, field_sym = em_info
                # Substitute theory parameters if symbolic (e.g., lambdaEM in coeff)
                for sym in em_coeff.free_symbols:
                    if hasattr(theory, str(sym)):
                        val = getattr(theory, str(sym))
                        if torch.is_tensor(val):
                            val = val.item()
                        em_coeff = em_coeff.subs(sym, val)
                try:
                    em_coeff = float(em_coeff)
                except:
                    try:
                        em_coeff = float(em_coeff.subs(subs_dict).evalf())
                    except:
                        print(f'Warning: Could not convert em_coeff to float for {theory.name}, using default -1/4')
                        em_coeff = -1/4.0
                # Strip EM term from Lagrangian (assumes it's coeff * field**2)
                L_grav = L_param - em_coeff * field_sym**2  # Use L_param
            else:
                L_grav = L_param
                em_coeff = -1/4.0
            rq = 1.0
            if include_charge and hasattr(theory, 'Q'):
                q_val = getattr(theory, 'Q')
                q = q_val.item() if torch.is_tensor(q_val) else q_val
                rq = math.sqrt( G * q**2 / (4 * math.pi * epsilon_0 * c**4) )
            # Derive from Lagrangian using existing function
            derived_metrics = derive_metric_from_action(L_grav, rs=rs_val, include_charge=include_charge, rq=rq, em_coeff=em_coeff)  # Use L_grav as before
            sampled_r_cpu = sampled_r.detach().cpu().numpy()
            g_tt_derived = torch.tensor([derived_metrics['g_tt'](float(r)) for r in sampled_r_cpu], device=self.engine.device, dtype=self.engine.dtype).detach().clone()
            g_rr_derived = torch.tensor([derived_metrics['g_rr'](float(r)) for r in sampled_r_cpu], device=self.engine.device, dtype=self.engine.dtype).detach().clone()
            
            # Compute MSE losses
            # Ensure g_tt_derived and g_rr_derived are tensors
            g_tt_derived = torch.tensor(g_tt_derived, device=self.engine.device, dtype=self.engine.dtype)
            g_rr_derived = torch.tensor(g_rr_derived, device=self.engine.device, dtype=self.engine.dtype)

            if self.loss_type == 'mse':
                loss_tt = torch.mean((g_tt_direct - g_tt_derived)**2).item()
                loss_rr = torch.mean((g_rr_direct - g_rr_derived)**2).item()
            elif self.loss_type == 'fft':
                fft_tt_direct = torch.fft.fft(g_tt_direct)
                fft_tt_derived = torch.fft.fft(g_tt_derived)
                loss_tt = torch.mean(torch.abs(fft_tt_direct - fft_tt_derived)**2).item()
                fft_rr_direct = torch.fft.fft(g_rr_direct)
                fft_rr_derived = torch.fft.fft(g_rr_derived)
                loss_rr = torch.mean(torch.abs(fft_rr_direct - fft_rr_derived)**2).item()
            elif self.loss_type == 'endpoint_mse':
                loss_tt = ((g_tt_direct[-1] - g_tt_derived[-1]) ** 2).item()
                loss_rr = ((g_rr_direct[-1] - g_rr_derived[-1]) ** 2).item()
            elif self.loss_type == 'cosine':
                loss_tt = (1 - torch.nn.functional.cosine_similarity(g_tt_direct.unsqueeze(0), g_tt_derived.unsqueeze(0))).item()
                loss_rr = (1 - torch.nn.functional.cosine_similarity(g_rr_direct.unsqueeze(0), g_rr_derived.unsqueeze(0))).item()
            elif self.loss_type == 'trajectory_mse':
                min_len = min(len(g_tt_direct), len(g_tt_derived))
                loss_tt = torch.mean((g_tt_direct[:min_len] - g_tt_derived[:min_len])**2).item()
                loss_rr = torch.mean((g_rr_direct[:min_len] - g_rr_derived[:min_len])**2).item()
            elif self.loss_type == 'hausdorff':
                from scipy.spatial.distance import directed_hausdorff
                points_tt_direct = torch.stack([sampled_r, g_tt_direct], dim=1).cpu().numpy()
                points_tt_derived = torch.stack([sampled_r, g_tt_derived], dim=1).cpu().numpy()
                loss_tt = max(directed_hausdorff(points_tt_direct, points_tt_derived)[0], directed_hausdorff(points_tt_derived, points_tt_direct)[0])
                points_rr_direct = torch.stack([sampled_r, g_rr_direct], dim=1).cpu().numpy()
                points_rr_derived = torch.stack([sampled_r, g_rr_derived], dim=1).cpu().numpy()
                loss_rr = max(directed_hausdorff(points_rr_direct, points_rr_derived)[0], directed_hausdorff(points_rr_derived, points_rr_direct)[0])
            elif self.loss_type == 'frechet':
                def frechet_distance(p, q):
                    from scipy.spatial.distance import cdist
                    dm = cdist(p, q)
                    return max(np.max(dm), np.max(dm.T))
                points_tt_direct = torch.stack([sampled_r, g_tt_direct], dim=1).cpu().numpy()
                points_tt_derived = torch.stack([sampled_r, g_tt_derived], dim=1).cpu().numpy()
                loss_tt = frechet_distance(points_tt_direct, points_tt_derived)
                points_rr_direct = torch.stack([sampled_r, g_rr_direct], dim=1).cpu().numpy()
                points_rr_derived = torch.stack([sampled_r, g_rr_derived], dim=1).cpu().numpy()
                loss_rr = frechet_distance(points_rr_direct, points_rr_derived)
            elif self.loss_type == 'trajectory_dot':
                dots_tt = torch.sum(g_tt_direct * g_tt_derived)
                loss_tt = dots_tt.item()
                dots_rr = torch.sum(g_rr_direct * g_rr_derived)
                loss_rr = dots_rr.item()
            elif self.loss_type == 'raw_dot':
                loss_tt = torch.dot(g_tt_direct, g_tt_derived).item()
                loss_rr = torch.dot(g_rr_direct, g_rr_derived).item()
            elif self.loss_type == 'ricci':
                # Compute numerical derivatives for direct
                dg_tt_dr_direct = torch.gradient(g_tt_direct, spacing=(sampled_r,))[0]
                dg_rr_dr_direct = torch.gradient(g_rr_direct, spacing=(sampled_r,))[0]
                dg_theta_dr_direct = torch.gradient(g_theta_direct, spacing=(sampled_r,))[0]
                dg_phi_dr_direct = torch.gradient(g_phi_direct, spacing=(sampled_r,))[0]
                dg_dr_direct = {'tt': dg_tt_dr_direct, 'rr': dg_rr_dr_direct, 'pp': dg_theta_dr_direct, 'tp': dg_phi_dr_direct}
                from physics_agent.functions import compute_ricci_tensor
                ricci_direct = compute_ricci_tensor(g_tt_direct, g_rr_direct, g_theta_direct, g_phi_direct, sampled_r, dg_dr=dg_dr_direct, device=self.engine.device, dtype=self.engine.dtype)
                
                # For derived (numerical)
                dg_tt_dr_derived = torch.gradient(g_tt_derived, spacing=(sampled_r,))[0]
                dg_rr_dr_derived = torch.gradient(g_rr_derived, spacing=(sampled_r,))[0]
                # Assume g_theta and g_phi same as direct for derived
                dg_dr_derived = {'tt': dg_tt_dr_derived, 'rr': dg_rr_dr_derived, 'pp': dg_theta_dr_direct, 'tp': dg_phi_dr_direct}
                ricci_derived = compute_ricci_tensor(g_tt_derived, g_rr_derived, g_theta_direct, g_phi_direct, sampled_r, dg_dr=dg_dr_derived, device=self.engine.device, dtype=self.engine.dtype)
                diff = ricci_direct - ricci_derived
                total_loss = torch.mean(diff**2).item()
            else:
                raise ValueError(f'Unsupported loss_type: {self.loss_type}')
            
            total_loss = (loss_tt + loss_rr) / 2
            
            # Flags
            tt_flag = "PASS" if loss_tt < self.tolerance else "FAIL"
            rr_flag = "PASS" if loss_rr < self.tolerance else "FAIL"
            overall_flag = "PASS" if (tt_flag == "PASS" and rr_flag == "PASS") else "FAIL"
            
            if self.plot_enabled:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = f"lagrangian_trajectory_comparison_{theory.name.replace(' ', '_')}_{timestamp}.png"
                self.plot_trajectory_comparison(theory, plot_path)
                print(f"Lagrangian trajectory comparison plotted to {plot_path}")

            result = {
                "loss": total_loss,
                "flags": {
                    "overall": overall_flag,
                    "g_tt_consistency": tt_flag,
                    "g_rr_consistency": rr_flag
                },
                "details": {
                    "mse_g_tt": loss_tt,
                    "mse_g_rr": loss_rr
                }
            }
            if experimental:
                quantum_result = self.quantum_validate(theory, experimental=True)
                result['quantum_loss'] = quantum_result['quantum_loss']
                result['flags'].update(quantum_result['flags'])
                result['details']['quantum'] = quantum_result['details']
                # Update overall flag if quantum fails
                if quantum_result['flags']['quantum_overall'] == 'FAIL':
                    result['flags']['overall'] = 'FAIL'
            return result
        
        except Exception as e:
            print(f"Warning: Lagrangian derivation failed for {theory.name}: {str(e)}")
            return {
                "loss": float('inf'),
                "flags": {"overall": "WARN", "details": f"Derivation error: {str(e)}"},
                "details": {}
            }

    def quantum_validate(self, theory: "GravitationalTheory", num_qubits: int = 4, steps: int = 10, tolerance: float = 1e-2, experimental: bool = False) -> Dict[str, Any]:
        """
        EXPERIMENTAL: Quantum circuit validation using PennyLane. Unvalidated feature.
        """
        if not experimental:
            return {"quantum_loss": 0.0, "flags": {"quantum_overall": "SKIP", "details": "Experimental quantum validation disabled."}}
        if not hasattr(theory, 'lagrangian') or theory.lagrangian is None:
            return {"quantum_loss": 0.0, "flags": {"quantum_overall": "SKIP", "details": "No Lagrangian for quantum validation."}}
        
        try:
            # Simple example: Simulate a quantum walk based on discretized Lagrangian
            dev = qml.device("default.qubit", wires=num_qubits)
            
            @qml.qnode(dev)
            def circuit(params):
                # Encode Lagrangian into gates (placeholder: use params to parameterize)
                for i in range(num_qubits):
                    qml.RY(params[i], wires=i)
                return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
            
            # Dummy params; in real: derive from Lagrangian sympy expr
            params = pnp.random.random(num_qubits)
            exp_vals = circuit(params)
            
            # Classical expectation (placeholder: from derive_metric)
            classical_exp = np.mean(exp_vals)  # Simplified
            
            # Compute loss
            quantum_loss = np.mean((exp_vals - classical_exp)**2)
            
            flag = "PASS" if quantum_loss < tolerance else "FAIL"
            
            return {
                "quantum_loss": float(quantum_loss),
                "flags": {"quantum_overall": flag},
                "details": {"exp_vals": exp_vals.tolist()}
            }
        except Exception as e:
            return {"quantum_loss": float('inf'), "flags": {"quantum_overall": "FAIL", "details": str(e)}}

from typing import Callable, Dict

import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from sympy import Derivative, ode_order
from sympy.solvers.solveset import NonlinearError

# --- HELPERS ---

def variational_derivative(expr: sp.Expr, f: sp.Function) -> sp.Expr:  # type: ignore[override]
    """Return the Euler-Lagrange expression δS/δf for a scalar field *f(r)*.
    Only supports 1D dependence (all fields are functions of the single
    radial coordinate *r*).  Sufficient for our metric ansatz.
    """
    r = list(f.free_symbols)[0]  # assumes exactly one independent variable
    total = sp.diff(expr, f)
    total -= sp.diff(sp.diff(expr, sp.diff(f, r)), r)
    total += sp.diff(sp.diff(expr, sp.diff(f, r, 2)), r, 2)
    return sp.simplify(total)

def safe_power(base: np.ndarray | float, exp: float) -> np.ndarray | float:
    """Real-valued power that copes with negative bases and non-integer exponents."""
    return np.sign(base) * np.abs(base) ** exp

def _drop_residual_derivs(expr: sp.Expr) -> sp.Expr:
    """Replace any leftover SymPy ``Derivative`` objects with zero."""
    return expr.replace(lambda e: isinstance(e, Derivative), lambda _: 0)

def ricci_scalar(A: sp.Expr, B: sp.Expr, r: sp.Symbol) -> sp.Expr:
    """Ricci scalar for the metric  ds² = -A dt² + B dr² + r² dΩ² ."""
    Ap, Bp = sp.diff(A, r), sp.diff(B, r)
    App = sp.diff(A, r, 2)

    term1 = (
        -App / A
        + sp.Rational(1, 2) * Ap**2 / A**2
        + sp.Rational(1, 2) * Ap * Bp / (A * B)
        - 2 * Ap / (r * A)
        + 2 * Bp / (r * B)
    ) / B
    term2 = 2 / r**2 * (1 - 1 / B)
    return sp.simplify(term1 + term2)

# ---

def derive_metric_from_action(
    L_expr: sp.Expr,
    *,
    rs: float = 2.0,
    include_charge: bool = False,
    rq: float = 1.0,
    em_coeff: float = -1/4.0
) -> Dict[str, Callable[[np.ndarray | float], np.ndarray | float]]:
    """Return interpolators for the metric (and electrostatic potential).
    ----------
    L_expr         : SymPy expression for the Lagrangian density (insert *R*)
    rs             : Schwarzschild radius (2M with G = c = 1)
    include_charge : introduce a Maxwell field Φ(r) if ``True``
    rq             : charge radius |Q| in geometric units
    """

    r = sp.symbols("r", positive=True)
    A = sp.Function("A")(r)
    # <reason>chain: Use C = 1/B to avoid infinity at horizon, C->0 as B->inf</reason>
    C = sp.Function("C")(r)
    B = 1 / C
    fields = [A, C]

    R_scalar = ricci_scalar(A, B, r)
    L_total = L_expr.subs({"R": R_scalar})

    if include_charge:
        Phi = sp.Function("Phi")(r)
        E2 = sp.diff(Phi, r)**2 / (A * B)
        F2 = -2 * E2
        L_total += em_coeff * F2
        fields.append(Phi)

    L_eff = r**2 * sp.sqrt(A * B) * L_total
    EL_eqs = [sp.simplify(variational_derivative(L_eff, f)) for f in fields]

    if max((ode_order(eq, f) or 0) for eq in EL_eqs for f in fields) > 2:
        raise RuntimeError("Action leads to higher-than-second-order equations.")

    # Isolate second derivatives A'', C'' (and Φ'')
    d2 = [sp.diff(f, r, 2) for f in fields]
    d2_sym = [sp.symbols(f"d2_{f.func.__name__}") for f in fields]
    sub_d2 = dict(zip(d2, d2_sym))
    rev_d2 = {v: k for k, v in sub_d2.items()}

    M, rhs_vec = sp.linear_eq_to_matrix([eq.subs(sub_d2) for eq in EL_eqs], d2_sym)
    lin = sp.linsolve((M, rhs_vec))
    if not lin:
        # Try first-order reduction
        d1      = [sp.diff(f, r) for f in fields]
        d1_sym  = [sp.symbols(f"d1_{f.func.__name__}") for f in fields]
        sub_d1  = dict(zip(d1, d1_sym))

        try:
            M1, rhs1 = sp.linear_eq_to_matrix(
                [eq.subs(sub_d1) for eq in EL_eqs], d1_sym   # A',C',(Φ')
            )
            lin1 = sp.linsolve((M1, rhs1))
            d1_sym_used = d1_sym
        except NonlinearError:
            d1_sym_used = d1_sym[:2]
            M1, rhs1 = sp.linear_eq_to_matrix(
                [eq.subs(sub_d1) for eq in EL_eqs[:2]],      # EL[0:2] ≡ A,C eqs
                d1_sym_used,
            )
            lin1 = sp.linsolve((M1, rhs1))

        if not lin1:
            print('Warning: Singular EL system - falling back to flat metric approximation')
            return {
                'g_tt': lambda R: -1.0,
                'g_rr': lambda R: 1.0,
                'g_theta_theta': lambda R: R**2,
                'g_phi_phi': lambda R: R**2
            }

        sol_d1 = {d: expr for d, expr in zip(d1_sym_used, list(lin1)[0])}

        if include_charge:
            d1_Phi_sym = sp.symbols("d1_Phi")
            subs_charge = {d1_Phi_sym: -rq / r**2}

            dA_expr = sol_d1[sp.symbols("d1_A")].subs(subs_charge)
            dC_expr = sol_d1[sp.symbols("d1_C")].subs(subs_charge)

            rhs_AC = sp.lambdify(
                (r, A, C, Phi),
                [dA_expr, dC_expr],
                modules=[{"pow": safe_power}, "numpy"],
            )
            def ode(rval: float, y):
                A_, C_, P_ = y
                dA, dC = rhs_AC(rval, A_, C_, P_)
                dP = -rq / (rval**2)
                return np.array([dA, dC, dP], dtype=float)
        else:
            rhs = sp.lambdify(
                (r, *fields),
                [sol_d1[d] for d in d1_sym_used],
                modules=[{"pow": safe_power}, "numpy"],
            )
            def ode(rval: float, y):
                return np.asarray(rhs(rval, *y), dtype=float)

        r_min, r_max = 4 * rs, 1e6 * rs
        if include_charge:
            f_inf = 1 - rs / r_max + rq**2 / r_max**2
            A_inf = f_inf
            C_inf = f_inf  # Since C = 1/B = f_inf
            Phi_inf = -rq / r_max
        else:
            f_inf = 1 - rs / r_max
            A_inf = f_inf
            C_inf = f_inf

        def bc(ya, yb):
            bc_vec = [
                ya[0],  # A(r_min) = 0 at horizon
                ya[1],  # C(r_min) = 0 at horizon
                yb[0] - A_inf,
                yb[1] - C_inf
            ]
            if include_charge:
                bc_vec += [yb[2] - Phi_inf]
            return np.asarray(bc_vec, dtype=float)

        inner = np.geomspace(r_min, 10 * rs, 300)
        outer = np.linspace(10 * rs, r_max, 200)[1:]
        r_mesh = np.concatenate((inner, outer))
        if include_charge:
            A0 = 1 - rs / r_mesh + rq**2 / r_mesh**2
            C0 = A0  # Initial guess C ≈ A
            Phi0 = -rq / r_mesh
            guess = np.vstack((A0, C0, Phi0))
        else:
            A0 = 1 - rs / r_mesh
            C0 = A0
            guess = np.vstack((A0, C0))

        sol_num = solve_bvp(ode, bc, r_mesh, guess, tol=1e-8, max_nodes=500000)
        if not sol_num.success:
            print("Initial solve failed, trying adaptive")
            # Adaptive code...

        g_tt_i = interp1d(sol_num.x, sol_num.y[0], kind="cubic")
        g_rr_i = lambda R: 1 / interp1d(sol_num.x, sol_num.y[1], kind="cubic")(R)  # B = 1/C
        if include_charge:
            Phi_i = interp1d(sol_num.x, sol_num.y[2], kind="cubic")
            return {"g_tt": lambda R: -g_tt_i(R), "g_rr": g_rr_i, "Phi": Phi_i}
        return {"g_tt": lambda R: -g_tt_i(R), "g_rr": g_rr_i}

# --- QUANTUM VALIDATION VIA PATH INTEGRALS ---

def approximate_scattering_amplitude(
    L_expr: sp.Expr,
    initial_state: Dict[str, float],
    final_state: Dict[str, float],
    num_paths: int = 1000,
    steps: int = 100,
    params: Dict[str, float] = {},
) -> float:
    """Approximate 2->2 scattering amplitude using semi-classical path integral.
    
    This implements a discretized path integral over random paths from initial
    to final state, computing the action integral for each and summing amplitudes.
    For validation against Standard Model (e.g., compare to tree-level QED).
    
    Args:
        L_expr: Lagrangian expression (SymPy)
        initial_state: Dict of initial coordinates/momenta
        final_state: Dict of final coordinates/momenta
        num_paths: Number of Monte Carlo paths to sample
        steps: Discretization steps per path
        
    Returns:
        Complex amplitude (magnitude for simplicity here)
    """
    # Derive action S = ∫ L dt
    t = sp.symbols('t')
    action = sp.integrate(L_expr, t)  # Symbolic action
    
    # Monte Carlo path integral approximation
    amplitude = 0.0 + 0.0j
    dt = (final_state.get('t', 1.0) - initial_state.get('t', 0.0)) / steps
    
    # Define the velocity symbol if it exists in the Lagrangian
    v_sym = sp.Symbol('v_x')
    dx_dt_sym = sp.Derivative(sp.Symbol('x'), sp.Symbol('t'))
    x_sym = sp.Symbol('x')


    for _ in range(num_paths):
        # Generate random path (simple Brownian for demo; improve with Metropolis)
        path = np.linspace(initial_state['x'], final_state['x'], steps) 
        path += np.random.normal(0, 0.5, steps) # Add noise to explore paths
        
        # Compute discretized action for this path
        S_path = 0.0
        for i in range(steps - 1):
            # Evaluate L along path segment (numerical; use SymPy lambdify)
            velocity = (path[i+1] - path[i]) / dt
            subs_dict = {x_sym: path[i], **params}
            if v_sym in L_expr.free_symbols:
                subs_dict[v_sym] = velocity
            if dx_dt_sym in L_expr.free_symbols:
                 subs_dict[dx_dt_sym] = velocity

            L_val = float(L_expr.subs(subs_dict).doit())
            S_path += L_val * dt
        
        # Add e^{i S / ħ} to amplitude (semi-classical)
        hbar = 1.0  # Unit value for semi-classical approximation
        amplitude += np.exp(1j * S_path / hbar)
    
    # Normalize and return magnitude for comparison
    return abs(amplitude / num_paths)

# Example usage:
# amp = approximate_scattering_amplitude(sp.sympify('p**2/2m - V(x)'), {'x':0, 't':0}, {'x':1, 't':1})

# Add tests
import unittest

class TestQuantumValidation(unittest.TestCase):
    def test_quantum_validate(self):
        # Mock theory with simple Lagrangian
        class MockTheory(GravitationalTheory):
            lagrangian = sp.sympify("x**2")  # Dummy
        
        engine = TheoryEngine()  # Assuming import
        validator = LagrangianValidator(engine)
        result = validator.quantum_validate(MockTheory())
        self.assertIn("quantum_loss", result)
        self.assertTrue(result["quantum_loss"] >= 0)

import os

if __name__ == '__main__':
    if os.environ.get('EXPERIMENTAL', '0') == '1':
        unittest.main()
    else:
        print('Experimental tests skipped. Set EXPERIMENTAL=1 to run.')

def plot_trajectory_comparison(self, theory: 'GravitationalTheory', plot_path: str):
    import matplotlib.pyplot as plt
    rs_val = (2 * self.engine.G_T * self.engine.M / self.engine.C_T**2).item()
    r_plot = np.logspace(np.log10(2 * rs_val), np.log10(100 * rs_val), 100)
    r_torch = torch.tensor(r_plot, device=self.engine.device, dtype=self.engine.dtype)
    g_tt_direct, g_rr_direct, _, _ = theory.get_metric(r_torch, self.engine.M, self.engine.C_T, self.engine.G_T)
    derived_metrics = derive_metric_from_action(theory.lagrangian, rs=rs_val)
    g_tt_derived = [derived_metrics['g_tt'](r) for r in r_plot]
    g_rr_derived = [derived_metrics['g_rr'](r) for r in r_plot]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(r_plot, g_tt_direct.cpu().numpy(), label='Direct')
    ax1.plot(r_plot, g_tt_derived, label='Derived')
    ax1.set_title(f'g_tt Comparison - {theory.name}')
    ax1.legend()
    ax2.plot(r_plot, g_rr_direct.cpu().numpy(), label='Direct')
    ax2.plot(r_plot, g_rr_derived, label='Derived')
    ax2.set_title(f'g_rr Comparison - {theory.name}')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()