"""
AI Feedback and Evolutionary System for Theory Discovery

<reason>chain: Implements evolutionary algorithms and AI feedback for unification discovery</reason>
"""

import numpy as np
import torch
import json
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass
import sympy as sp


@dataclass
class TheoryGenome:
    """
    <reason>chain: Genetic representation of a theory for evolution</reason>
    """
    category: str  # 'classical', 'quantum', 'quantum'
    parameters: Dict[str, float]
    lagrangian_terms: List[str]
    symmetries: List[str]
    fitness: float = 0.0
    generation: int = 0
    parents: List[str] = None
    
    def mutate(self, mutation_rate: float = 0.1) -> 'TheoryGenome':
        """<reason>chain: Mutate theory parameters and structure</reason>"""
        mutated = TheoryGenome(
            category=self.category,
            parameters=self.parameters.copy(),
            lagrangian_terms=self.lagrangian_terms.copy(),
            symmetries=self.symmetries.copy(),
            generation=self.generation + 1,
            parents=[self.to_id()]
        )
        
        # Mutate parameters
        for param, value in mutated.parameters.items():
            if random.random() < mutation_rate:
                # Log-normal mutation for scale parameters
                if 'scale' in param or 'mass' in param:
                    mutated.parameters[param] = value * np.exp(random.gauss(0, 0.3))
                else:
                    # Gaussian mutation for dimensionless parameters
                    mutated.parameters[param] = value + random.gauss(0, 0.1 * abs(value))
                    
        # Mutate Lagrangian structure
        if random.random() < mutation_rate / 2:
            # Add new term
            new_terms = ['R**2', 'R_μν**2', 'S_ent', 'F_μν * F^μν', 'ψ̄ψ']
            available = [t for t in new_terms if t not in mutated.lagrangian_terms]
            if available:
                mutated.lagrangian_terms.append(random.choice(available))
                
        # Evolve category
        if random.random() < mutation_rate / 10:
            if mutated.category == 'classical':
                mutated.category = 'quantum'
            elif mutated.category == 'quantum' and 'gauge' in ' '.join(mutated.lagrangian_terms):
                mutated.category = 'quantum'
                
        return mutated
        
    def crossover(self, other: 'TheoryGenome') -> 'TheoryGenome':
        """<reason>chain: Combine two theories to create offspring</reason>"""
        # Mix parameters
        child_params = {}
        for param in set(self.parameters.keys()) | set(other.parameters.keys()):
            if param in self.parameters and param in other.parameters:
                # Average with random weight
                w = random.random()
                child_params[param] = w * self.parameters[param] + (1-w) * other.parameters[param]
            elif param in self.parameters:
                child_params[param] = self.parameters[param]
            else:
                child_params[param] = other.parameters[param]
                
        # Combine Lagrangian terms
        child_terms = list(set(self.lagrangian_terms + other.lagrangian_terms))
        
        # Higher category wins
        categories = [self.category, other.category]
        if 'quantum' in categories:
            child_category = 'quantum'
        elif 'quantum' in categories:
            child_category = 'quantum'
        else:
            child_category = 'classical'
            
        return TheoryGenome(
            category=child_category,
            parameters=child_params,
            lagrangian_terms=child_terms,
            symmetries=list(set(self.symmetries + other.symmetries)),
            generation=max(self.generation, other.generation) + 1,
            parents=[self.to_id(), other.to_id()]
        )
        
    def to_id(self) -> str:
        """<reason>chain: Generate unique ID for theory</reason>"""
        param_str = '_'.join(f'{k}{v:.2e}' for k, v in sorted(self.parameters.items()))
        return f"{self.category}_{param_str}_gen{self.generation}"


class AIFeedbackEvolution:
    """
    <reason>chain: Main evolutionary system for theory discovery</reason>
    """
    
    def __init__(self, population_size: int = 100, elite_fraction: float = 0.1):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.population: List[TheoryGenome] = []
        self.generation = 0
        self.best_theories: List[TheoryGenome] = []
        
        # <reason>chain: Prompt templates for different unification approaches</reason>
        self.unification_prompts = {
            'gauge': "Explore gauge-quantum theories where gravity emerges from U(1)×SU(2)×SU(3) gauge symmetries",
            'entropy': "Investigate emergent gravity from quantum entanglement entropy, following ER=EPR conjecture",
            'information': "Develop theories where spacetime geometry encodes quantum information",
            'holographic': "Apply holographic principle: gravity in bulk equals CFT on boundary",
            'string': "Derive effective field theory from string/M-theory compactification"
        }
        
    def initialize_population(self, seed_theories: List[TheoryGenome] = None):
        """<reason>chain: Create initial population of theories</reason>"""
        if seed_theories:
            self.population.extend(seed_theories)
            
        # <reason>chain: Generate diverse initial population</reason>
        while len(self.population) < self.population_size:
            # Random category weighted toward unified
            category = random.choices(
                ['classical', 'quantum', 'quantum'],
                weights=[0.1, 0.3, 0.6]
            )[0]
            
            # Random parameters
            params = {}
            if category in ['quantum', 'quantum']:
                params['alpha'] = 10 ** random.uniform(-6, -2)  # Quantum correction strength
                params['beta'] = random.uniform(0, 1)  # Entanglement parameter
                
            if category == 'quantum':
                params['unification_scale'] = 10 ** random.uniform(15, 19)  # GeV
                params['coupling_constant'] = random.uniform(0.5, 1.0)
                
            # Random Lagrangian terms
            possible_terms = ['R', 'R**2', 'F_μν * F^μν', 'ψ̄ψ', 'S_ent', 'R_μν**2']
            num_terms = random.randint(2, 5)
            terms = random.sample(possible_terms, num_terms)
            
            # Symmetries
            symmetries = ['Lorentz']
            if 'F_μν' in ' '.join(terms):
                symmetries.extend(['U(1)', 'SU(2)', 'SU(3)'])
                
            genome = TheoryGenome(
                category=category,
                parameters=params,
                lagrangian_terms=terms,
                symmetries=symmetries
            )
            
            self.population.append(genome)
            
    def evaluate_fitness(self, genome: TheoryGenome, validation_results: Dict) -> float:
        """
        <reason>chain: Compute fitness score prioritizing unification</reason>
        
        Fitness criteria:
        1. Low loss against both classical and quantum baselines
        2. Passes renormalizability checks
        3. Makes novel testable predictions
        4. Unifies at reasonable scale
        """
        fitness = 0.0
        
        # <reason>chain: Base fitness from validation losses</reason>
        total_loss = validation_results.get('total_loss', float('inf'))
        if total_loss < float('inf'):
            fitness += 100 / (1 + total_loss)
            
        # <reason>chain: Bonus for unification properties</reason>
        if genome.category == 'quantum':
            fitness += 20
            
        if validation_results.get('is_renormalizable', False):
            fitness += 50
            
        if validation_results.get('is_gauge_invariant', False):
            fitness += 30
            
        # <reason>chain: Unification scale quality</reason>
        unif_scale = validation_results.get('unification_scale', 0)
        if unif_scale > 0:
            # Prefer scales near GUT scale
            scale_quality = np.exp(-abs(np.log10(unif_scale / 2e16)))
            fitness += 40 * scale_quality
            
        # <reason>chain: Novel predictions bonus</reason>
        num_predictions = len(validation_results.get('novel_predictions', []))
        fitness += 5 * num_predictions
        
        # <reason>chain: Penalty for violations</reason>
        if validation_results.get('flags', {}).get('proton_decay_excluded', False):
            fitness *= 0.1
            
        if validation_results.get('flags', {}).get('energy_violated', False):
            fitness *= 0.5
            
        genome.fitness = fitness
        return fitness
        
    def select_parents(self) -> Tuple[TheoryGenome, TheoryGenome]:
        """<reason>chain: Tournament selection for breeding</reason>"""
        tournament_size = 5
        
        # Select two parents via tournaments
        parents = []
        for _ in range(2):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda g: g.fitness)
            parents.append(winner)
            
        return tuple(parents)
        
    def evolve_generation(self, validation_func):
        """<reason>chain: Run one generation of evolution</reason>"""
        self.generation += 1
        
        # <reason>chain: Evaluate all genomes</reason>
        for genome in self.population:
            if genome.fitness == 0:  # Not yet evaluated
                # Convert genome to theory and validate
                theory = self.genome_to_theory(genome)
                results = validation_func(theory)
                self.evaluate_fitness(genome, results)
                
        # <reason>chain: Sort by fitness</reason>
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # <reason>chain: Save best theories</reason>
        num_elite = int(self.population_size * self.elite_fraction)
        self.best_theories.extend(self.population[:num_elite])
        
        # <reason>chain: Create next generation</reason>
        new_population = self.population[:num_elite]  # Elitism
        
        while len(new_population) < self.population_size:
            # Crossover
            if random.random() < 0.7:
                parent1, parent2 = self.select_parents()
                child = parent1.crossover(parent2)
            else:
                # Mutation only
                parent = random.choice(self.population[:self.population_size // 2])
                child = parent.mutate()
                
            new_population.append(child)
            
        self.population = new_population
        
    def generate_ai_prompt(self, focus: str = 'quantum') -> str:
        """<reason>chain: Generate prompts for AI theory generation</reason>"""
        best_genome = max(self.population, key=lambda g: g.fitness)
        
        prompt = f"""
Based on our evolutionary search for quantum theories of gravity, please generate a new theory.

Current best approach:
- Category: {best_genome.category}
- Lagrangian terms: {', '.join(best_genome.lagrangian_terms)}
- Key parameters: {json.dumps(best_genome.parameters, indent=2)}
- Fitness score: {best_genome.fitness:.2f}

Focus: {self.unification_prompts.get(focus, self.unification_prompts['gauge'])}

Requirements:
1. The theory must include a complete Lagrangian with gravity, matter, and gauge terms
2. It should be renormalizable (dimension ≤ 4 operators)
3. It must make specific predictions at the unification scale
4. Include mechanisms for resolving the hierarchy problem and dark energy

Please provide:
1. The metric tensor components g_μν(r)
2. Complete Lagrangian with all terms
3. Unification scale and coupling values
4. Novel predictions for experiments

Consider recent insights:
- Entanglement entropy may be the source of spacetime geometry
- Gauge symmetries could unify all forces including gravity
- Quantum information principles constrain the theory
"""
        
        return prompt
        
    def genome_to_theory(self, genome: TheoryGenome):
        """<reason>chain: Convert genome to executable theory class</reason>"""
        # This would create a dynamic theory class
        # For now, return a mock theory with genome properties
        
        class DynamicTheory:
            def __init__(self):
                self.category = genome.category
                self.name = f"Evolved_{genome.to_id()}"
                
                # Set parameters as attributes
                for param, value in genome.parameters.items():
                    setattr(self, param, value)
                    
                # Build Lagrangian from terms
                self.lagrangian = self._build_lagrangian(genome.lagrangian_terms)
                
            def _build_lagrangian(self, terms):
                L = sp.Symbol('0')
                for term in terms:
                    if term == 'R':
                        L += sp.Symbol('R')
                    elif term == 'R**2':
                        L += sp.Symbol('alpha') * sp.Symbol('R')**2
                    elif term == 'S_ent':
                        L += sp.Symbol('beta') * sp.Symbol('S_ent')
                    # ... etc
                return L
                
            def get_metric(self, r, M, c, G):
                # Simplified metric based on category
                rs = 2 * G * M / c**2
                if self.category == 'classical':
                    f = 1 - rs / r
                else:
                    # Add quantum corrections
                    alpha = getattr(self, 'alpha', 1e-4)
                    f = 1 - rs / r + alpha * (rs / r)**2
                    
                return -f, 1/f, r**2, torch.zeros_like(r)
                
        return DynamicTheory()
        
    def export_best_theories(self, filename: str = 'best_theories.json'):
        """<reason>chain: Save best discovered theories</reason>"""
        data = {
            'generation': self.generation,
            'best_theories': [
                {
                    'id': g.to_id(),
                    'fitness': g.fitness,
                    'category': g.category,
                    'parameters': g.parameters,
                    'lagrangian_terms': g.lagrangian_terms,
                    'symmetries': g.symmetries,
                    'parents': g.parents
                }
                for g in sorted(self.best_theories, key=lambda g: g.fitness, reverse=True)[:10]
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2) 