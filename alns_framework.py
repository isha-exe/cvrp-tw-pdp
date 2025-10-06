"""
alns_framework.py
Adaptive Large Neighborhood Search (ALNS) Framework

Phase 1: Classical ALNS with adaptive operator weights
"""

import random
import math
import copy
from typing import List, Tuple, Dict
from datetime import datetime, timedelta

# Import from Phase 0
from cvrptw_parser import ProblemInstance, Solution, Service, Task, Order
from constraint_validator import ConstraintValidator, ObjectiveCalculator


class ALNSFramework:
    """Adaptive Large Neighborhood Search for CVRPTW-PD"""
    
    def __init__(self, problem: ProblemInstance, 
                 destroy_operators: List,
                 repair_operators: List,
                 seed: int = 42):
        
        self.problem = problem
        self.destroy_ops = destroy_operators
        self.repair_ops = repair_operators
        
        # Initialize operator weights (all start equal)
        self.destroy_weights = {op.name: 1.0 for op in destroy_operators}
        self.repair_weights = {op.name: 1.0 for op in repair_operators}
        
        # Operator statistics
        self.destroy_stats = {op.name: {'used': 0, 'improved': 0} for op in destroy_operators}
        self.repair_stats = {op.name: {'used': 0, 'improved': 0} for op in repair_operators}
        
        # ALNS parameters
        self.reaction_factor = 0.1  # Weight update speed
        self.noise_factor = 0.025   # Randomness in selection
        
        # Score rewards for operator performance
        self.scores = {
            'new_best': 33,           # Found new global best
            'better': 9,              # Better than current
            'accepted': 3,            # Accepted but not better
            'rejected': 0             # Rejected
        }
        
        # Validators
        self.validator = ConstraintValidator(problem)
        self.objective = ObjectiveCalculator(problem)
        
        random.seed(seed)
    
    def optimize(self, initial_solution: Solution, 
                 max_iterations: int = 1000,
                 temperature_start: float = 500.0,
                 temperature_end: float = 0.1,
                 cooling_rate: float = 0.9975,
                 destroy_size_min: int = 1,
                 destroy_size_max: int = 4) -> Tuple[Solution, Dict]:
        """
        Run ALNS optimization
        
        Args:
            initial_solution: Starting solution
            max_iterations: Number of iterations
            temperature_start: Initial temperature for SA
            temperature_end: Final temperature
            cooling_rate: Temperature decay rate
            destroy_size_min: Minimum orders to remove
            destroy_size_max: Maximum orders to remove
            
        Returns:
            (best_solution, statistics)
        """
        
        print(f"\n{'='*70}")
        print(f"PHASE 1: ALNS OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Max iterations: {max_iterations}")
        print(f"Destroy operators: {len(self.destroy_ops)}")
        print(f"Repair operators: {len(self.repair_ops)}")
        print(f"Initial temperature: {temperature_start}")
        print(f"Cooling rate: {cooling_rate}")
        
        # Initialize
        current_solution = copy.deepcopy(initial_solution)
        best_solution = copy.deepcopy(initial_solution)
        
        current_cost, _ = self.objective.calculate(current_solution)
        best_cost = current_cost
        
        temperature = temperature_start
        
        # Statistics
        stats = {
            'iteration': [],
            'current_cost': [],
            'best_cost': [],
            'temperature': [],
            'accepted': 0,
            'rejected': 0,
            'new_best': 0,
            'destroy_op_used': {op.name: 0 for op in self.destroy_ops},
            'repair_op_used': {op.name: 0 for op in self.repair_ops}
        }
        
        print(f"\nInitial solution:")
        print(f"  Cost: {current_cost:.2f}")
        print(f"  Coverage: {current_solution.coverage_rate(len(self.problem.orders))*100:.1f}%")
        
        # Main ALNS loop
        print(f"\n{'='*70}")
        print(f"Starting optimization...")
        print(f"{'='*70}")
        
        for iteration in range(max_iterations):
            # Select operators
            destroy_op = self._select_operator(self.destroy_ops, self.destroy_weights)
            repair_op = self._select_operator(self.repair_ops, self.repair_weights)
            
            # Determine destroy size
            k = random.randint(destroy_size_min, min(destroy_size_max, 
                                                     len(current_solution.services) * 2))
            
            # Apply destroy-repair
            try:
                new_solution = self._destroy_repair(
                    copy.deepcopy(current_solution),
                    destroy_op,
                    repair_op,
                    k
                )
                
                # Calculate new cost
                new_cost, _ = self.objective.calculate(new_solution)
                
                # Acceptance criterion (Simulated Annealing)
                accept, score_type = self._accept(new_cost, current_cost, best_cost, temperature)
                
                if accept:
                    current_solution = new_solution
                    current_cost = new_cost
                    stats['accepted'] += 1
                    
                    if new_cost < best_cost:
                        best_solution = copy.deepcopy(new_solution)
                        best_cost = new_cost
                        stats['new_best'] += 1
                        score_type = 'new_best'
                else:
                    stats['rejected'] += 1
                
                # Update operator weights
                score = self.scores[score_type]
                self._update_weights(destroy_op.name, repair_op.name, score)
                
                # Update statistics
                stats['destroy_op_used'][destroy_op.name] += 1
                stats['repair_op_used'][repair_op.name] += 1
                
            except Exception as e:
                # If destroy-repair fails, skip this iteration
                score = 0
                stats['rejected'] += 1
            
            # Cool down temperature
            temperature = max(temperature * cooling_rate, temperature_end)
            
            # Record progress
            if iteration % 50 == 0:
                stats['iteration'].append(iteration)
                stats['current_cost'].append(current_cost)
                stats['best_cost'].append(best_cost)
                stats['temperature'].append(temperature)
                
                coverage = best_solution.coverage_rate(len(self.problem.orders)) * 100
                is_feasible, violations = self.validator.validate_solution(best_solution)
                
                print(f"Iter {iteration:4d} | Best: {best_cost:8.2f} | "
                      f"Current: {current_cost:8.2f} | Coverage: {coverage:5.1f}% | "
                      f"Violations: {len(violations):2d} | Temp: {temperature:6.2f}")
        
        # Final statistics
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total iterations: {max_iterations}")
        print(f"Accepted: {stats['accepted']} ({stats['accepted']/max_iterations*100:.1f}%)")
        print(f"Rejected: {stats['rejected']} ({stats['rejected']/max_iterations*100:.1f}%)")
        print(f"New best: {stats['new_best']}")
        
        print(f"\nImprovement:")
        initial_cost, _ = self.objective.calculate(initial_solution)
        improvement = (initial_cost - best_cost) / initial_cost * 100
        print(f"  Initial cost: {initial_cost:.2f}")
        print(f"  Final cost: {best_cost:.2f}")
        print(f"  Improvement: {improvement:.2f}%")
        
        print(f"\nOperator Usage:")
        print(f"  Destroy operators:")
        for name, count in stats['destroy_op_used'].items():
            pct = count / max_iterations * 100
            weight = self.destroy_weights[name]
            print(f"    {name:20s}: {count:4d} ({pct:5.1f}%), weight={weight:.3f}")
        
        print(f"  Repair operators:")
        for name, count in stats['repair_op_used'].items():
            pct = count / max_iterations * 100
            weight = self.repair_weights[name]
            print(f"    {name:20s}: {count:4d} ({pct:5.1f}%), weight={weight:.3f}")
        
        return best_solution, stats
    
    def _select_operator(self, operators: List, weights: Dict[str, float]):
        """Select operator using roulette wheel with weights"""
        # Add noise for exploration
        noisy_weights = {
            name: w * (1.0 + random.uniform(-self.noise_factor, self.noise_factor))
            for name, w in weights.items()
        }
        
        total = sum(noisy_weights.values())
        probabilities = {name: w/total for name, w in noisy_weights.items()}
        
        # Roulette wheel selection
        r = random.random()
        cumulative = 0.0
        
        for op in operators:
            cumulative += probabilities[op.name]
            if r <= cumulative:
                return op
        
        return operators[-1]  # Fallback
    
    def _destroy_repair(self, solution: Solution, 
                       destroy_op, repair_op, k: int) -> Solution:
        """Apply destroy and repair operators"""
        # Destroy: remove k orders
        removed_orders = destroy_op.destroy(solution, k)
        
        # Repair: reinsert removed orders
        solution = repair_op.repair(solution, removed_orders)
        
        return solution
    
    def _accept(self, new_cost: float, current_cost: float, 
                best_cost: float, temperature: float) -> Tuple[bool, str]:
        """
        Acceptance criterion using Simulated Annealing
        
        Returns: (accept, score_type)
        """
        if new_cost < best_cost:
            return True, 'new_best'
        elif new_cost < current_cost:
            return True, 'better'
        else:
            # Simulated annealing acceptance
            delta = new_cost - current_cost
            probability = math.exp(-delta / temperature)
            
            if random.random() < probability:
                return True, 'accepted'
            else:
                return False, 'rejected'
    
    def _update_weights(self, destroy_name: str, repair_name: str, score: float):
        """Update operator weights using exponential smoothing"""
        # Update destroy operator weight
        old_weight = self.destroy_weights[destroy_name]
        self.destroy_weights[destroy_name] = (
            old_weight * (1 - self.reaction_factor) + 
            score * self.reaction_factor
        )
        
        # Update repair operator weight
        old_weight = self.repair_weights[repair_name]
        self.repair_weights[repair_name] = (
            old_weight * (1 - self.reaction_factor) + 
            score * self.reaction_factor
        )
        
        # Ensure minimum weight
        self.destroy_weights[destroy_name] = max(0.1, self.destroy_weights[destroy_name])
        self.repair_weights[repair_name] = max(0.1, self.repair_weights[repair_name])


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("ALNS Framework ready!")
    print("\nTo use:")
    print("  1. Import destroy/repair operators")
    print("  2. Create ALNS instance")
    print("  3. Run optimize() with initial solution")