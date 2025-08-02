"""
Scientific descriptions for trajectory plots to help students understand the physics.
"""

def get_theory_description(theory_name: str) -> str:
    """Get a student-friendly description of each theory."""
    
    descriptions = {
        "Schwarzschild": """
The Schwarzschild metric describes spacetime around a non-rotating, uncharged black hole.
Key features:
• Event horizon at r = 2M (Schwarzschild radius)
• Photon sphere at r = 3M (unstable circular orbits for light)
• ISCO (Innermost Stable Circular Orbit) at r = 6M for massive particles
• Purely radial gravitational effects with spherical symmetry
        """,
        
        "Kerr": """
The Kerr metric describes spacetime around a rotating black hole.
Key features:
• Event horizon location depends on spin parameter a
• Frame-dragging effect: spacetime itself rotates with the black hole
• Ergosphere region where particles must co-rotate
• Can extract energy via the Penrose process
        """,
        
        "Kerr-Newman": """
The Kerr-Newman metric describes a rotating, electrically charged black hole.
Key features:
• Combines rotation (Kerr) and charge (Reissner-Nordström) effects
• Multiple horizons possible depending on charge and spin
• Electromagnetic fields affect charged particle trajectories
• Most general black hole solution in classical GR
        """,
        
        "String Theory": """
String theory modifies gravity at the Planck scale with quantum corrections.
Key features:
• Extra dimensions compactified at small scales
• α' parameter controls string tension and quantum corrections
• Modifies strong-field gravity near the horizon
• Predicts small deviations from classical GR
        """,
        
        "Loop Quantum Gravity": """
Loop quantum gravity quantizes spacetime itself into discrete units.
Key features:
• Spacetime has a minimum length scale (Planck length)
• γ parameter (Immirzi parameter) controls quantum effects
• Prevents singularities through quantum geometry
• Modifies physics near the Planck scale
        """,
        
        "Quantum Corrected": """
Effective quantum corrections to classical general relativity.
Key features:
• α parameter controls strength of quantum corrections
• Becomes important near the horizon
• Can prevent information loss in black holes
• Smooth transition between quantum and classical regimes
        """,
        
        "Yukawa": """
Yukawa-type modification adds exponential screening to gravity.
Key features:
• λ parameter sets the screening length scale
• Gravity weakens faster than 1/r² at large distances
• Originally from nuclear physics (meson exchange)
• Tests modifications to Newton's inverse square law
        """,
        
        "Asymptotic Safety": """
Asymptotic safety makes gravity well-behaved at all energy scales.
Key features:
• Λ_as is the asymptotic safety scale
• Gravity becomes weaker at very high energies
• Avoids infinities in quantum gravity
• Running coupling constants with energy
        """,
        
        "Non-Commutative Geometry": """
Space coordinates don't commute at the Planck scale.
Key features:
• θ parameter controls non-commutativity
• [x,y] ≠ 0 at small scales (like quantum mechanics)
• Modifies particle trajectories at high energies
• Natural minimum length scale
        """,
        
        "Twistor Theory": """
Reformulates spacetime in terms of light rays (twistors).
Key features:
• λ parameter controls twistor deformation
• Natural framework for massless particles
• Connects quantum theory and relativity
• Elegant description of null geodesics
        """,
        
        "Einstein Teleparallel": """
Reformulates gravity using torsion instead of curvature.
Key features:
• τ parameter controls torsion strength
• Mathematically equivalent to GR when τ=0
• Uses flat connection with torsion
• Alternative geometric interpretation of gravity
        """,
        
        "Newtonian Limit": """
Classical Newtonian gravity in the weak-field limit.
Key features:
• Valid for v << c and weak gravitational fields
• No relativistic effects or spacetime curvature
• Instantaneous action at a distance
• Baseline for comparing with relativistic theories
        """
    }
    
    # Find matching description
    for key, desc in descriptions.items():
        if key.lower() in theory_name.lower():
            return desc.strip()
    
    # Default description
    return f"""
{theory_name} is a modified theory of gravity.
This theory predicts deviations from general relativity
that may be testable through precision measurements
of particle trajectories near black holes.
    """.strip()

def get_motion_analysis(trajectory_stats: dict) -> str:
    """Provide physics interpretation of trajectory statistics."""
    
    orbits = trajectory_stats.get('orbits_completed', 0)
    r_variation = trajectory_stats.get('r_variation', 0)
    avg_speed = trajectory_stats.get('avg_speed_c', 0)
    
    analysis = []
    
    # Orbit type
    if orbits < 0.1:
        analysis.append("• Plunging trajectory: particle falling toward horizon")
    elif orbits < 1:
        analysis.append("• Partial orbit: trajectory interrupted or escaping")
    elif r_variation < 5:
        analysis.append(f"• Stable circular orbit: {orbits:.1f} complete revolutions")
    elif r_variation < 20:
        analysis.append(f"• Elliptical orbit: {orbits:.1f} orbits with {r_variation:.1f}% eccentricity")
    else:
        analysis.append(f"• Highly eccentric orbit: large radial variations ({r_variation:.1f}%)")
    
    # Speed regime
    if avg_speed < 0.1:
        analysis.append("• Non-relativistic motion (v << c)")
    elif avg_speed < 0.5:
        analysis.append(f"• Mildly relativistic: v ≈ {avg_speed:.2f}c")
    else:
        analysis.append(f"• Highly relativistic: v ≈ {avg_speed:.2f}c")
    
    # Stability
    if trajectory_stats.get('final_r', 0) < 2.1:
        analysis.append("• Fate: Crossing event horizon")
    elif trajectory_stats.get('final_r', 0) > 50:
        analysis.append("• Fate: Escaping to infinity")
    else:
        analysis.append("• Fate: Bound orbit around black hole")
    
    return "\n".join(analysis)

def get_reference_explanation() -> str:
    """Explain the reference circles/spheres in the plots."""
    return """
Reference surfaces (for Schwarzschild metric):
• Black circle: Event horizon (r = 2M) - point of no return
• Orange dashed: Photon sphere (r = 3M) - unstable light orbits  
• Purple dotted: ISCO (r = 6M) - innermost stable orbit for matter

Note: These radii may differ for other theories due to quantum
corrections or modified gravity effects.
    """