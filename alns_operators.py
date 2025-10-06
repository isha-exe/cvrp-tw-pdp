"""
alns_operators.py
ALNS operators compatible with RL training

Simplified wrappers around existing pdp_nlns_operators for RL integration
"""

import random
import numpy as np
from typing import List, Tuple
from cvrptw_parser import Solution, Order, ProblemInstance

# Import existing operators
try:
    from pdp_nlns_operators import (
        RandomRemoval, WorstRemoval, ShawRemoval,
        GreedyInsertion, Regret2Insertion
    )
    OPERATORS_AVAILABLE = True
except ImportError:
    OPERATORS_AVAILABLE = False
    print("Warning: pdp_nlns_operators not found")


class DestroyOperators:
    """
    Wrapper class for destroy operators
    Provides static methods for RL integration
    """
    
    @staticmethod
    def random_removal(solution: Solution, problem: ProblemInstance, 
                      removal_rate: float = 0.3) -> Tuple[Solution, List[Order]]:
        """Remove random orders"""
        if not OPERATORS_AVAILABLE:
            return solution, []
        
        op = RandomRemoval(problem)
        
        # Calculate number of orders to remove
        served_orders = [o for o in problem.orders if o not in solution.unserved_orders]
        num_to_remove = max(1, int(len(served_orders) * removal_rate))
        
        destroyed_solution = op.destroy(solution, num_destroy=num_to_remove)
        
        # Find which orders were removed (now in unserved)
        removed_orders = [o for o in destroyed_solution.unserved_orders 
                         if o not in solution.unserved_orders]
        
        return destroyed_solution, removed_orders
    
    @staticmethod
    def worst_removal(solution: Solution, problem: ProblemInstance,
                     removal_rate: float = 0.3) -> Tuple[Solution, List[Order]]:
        """Remove orders with highest cost contribution"""
        if not OPERATORS_AVAILABLE:
            return solution, []
        
        op = WorstRemoval(problem)
        
        served_orders = [o for o in problem.orders if o not in solution.unserved_orders]
        num_to_remove = max(1, int(len(served_orders) * removal_rate))
        
        destroyed_solution = op.destroy(solution, num_destroy=num_to_remove)
        removed_orders = [o for o in destroyed_solution.unserved_orders 
                         if o not in solution.unserved_orders]
        
        return destroyed_solution, removed_orders
    
    @staticmethod
    def shaw_removal(solution: Solution, problem: ProblemInstance,
                    removal_rate: float = 0.3) -> Tuple[Solution, List[Order]]:
        """Remove similar orders (spatial/temporal proximity)"""
        if not OPERATORS_AVAILABLE:
            return solution, []
        
        op = ShawRemoval(problem)
        
        served_orders = [o for o in problem.orders if o not in solution.unserved_orders]
        num_to_remove = max(1, int(len(served_orders) * removal_rate))
        
        destroyed_solution = op.destroy(solution, num_destroy=num_to_remove)
        removed_orders = [o for o in destroyed_solution.unserved_orders 
                         if o not in solution.unserved_orders]
        
        return destroyed_solution, removed_orders
    
    @staticmethod
    def route_removal(solution: Solution, problem: ProblemInstance,
                     removal_rate: float = 0.3) -> Tuple[Solution, List[Order]]:
        """Remove complete route"""
        if not solution.services:
            return solution, []
        
        # Select random service
        service = random.choice(solution.services)
        
        # Create copy and remove service
        destroyed_solution = solution.copy()
        removed_orders = service.orders.copy()
        
        # Remove service
        destroyed_solution.services = [s for s in destroyed_solution.services 
                                      if s.service_id != service.service_id]
        
        # Add orders to unserved
        for order in removed_orders:
            if order not in destroyed_solution.unserved_orders:
                destroyed_solution.unserved_orders.append(order)
        
        return destroyed_solution, removed_orders


class RepairOperators:
    """
    Wrapper class for repair operators
    Provides static methods for RL integration
    """
    
    @staticmethod
    def greedy_insertion(partial_solution: Solution, removed_orders: List[Order],
                        problem: ProblemInstance) -> Solution:
        """Greedy insertion of removed orders"""
        if not OPERATORS_AVAILABLE or not removed_orders:
            return partial_solution
        
        op = GreedyInsertion(problem)
        
        # Set unserved orders
        solution = partial_solution.copy()
        solution.unserved_orders = removed_orders.copy()
        
        # Repair
        repaired_solution = op.repair(solution)
        
        return repaired_solution
    
    @staticmethod
    def regret_insertion(partial_solution: Solution, removed_orders: List[Order],
                        problem: ProblemInstance) -> Solution:
        """Regret-2 insertion"""
        if not OPERATORS_AVAILABLE or not removed_orders:
            return partial_solution
        
        op = Regret2Insertion(problem)
        
        solution = partial_solution.copy()
        solution.unserved_orders = removed_orders.copy()
        
        repaired_solution = op.repair(solution)
        
        return repaired_solution
    
    @staticmethod
    def best_insertion(partial_solution: Solution, removed_orders: List[Order],
                      problem: ProblemInstance) -> Solution:
        """Best insertion (same as greedy for now)"""
        return RepairOperators.greedy_insertion(partial_solution, removed_orders, problem)


# Test if operators work
if __name__ == "__main__":
    print("ALNS Operators for RL")
    print("=" * 60)
    
    if OPERATORS_AVAILABLE:
        print("\n✓ Operators imported successfully")
        print("\nDestroy Operators:")
        print("  - random_removal")
        print("  - worst_removal")
        print("  - shaw_removal")
        print("  - route_removal")
        print("\nRepair Operators:")
        print("  - greedy_insertion")
        print("  - regret_insertion")
        print("  - best_insertion")
    else:
        print("\n✗ pdp_nlns_operators not found")
        print("  Make sure pdp_nlns_operators.py is in the same directory")