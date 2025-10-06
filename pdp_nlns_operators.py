"""
pdp_nlns_operators.py
Destroy and Repair operators for Pickup-Delivery Problem with NLNS

Phase 1: ALNS operators that handle pickup-delivery pairs
"""

import random
import copy
from typing import List, Tuple, Optional
from datetime import timedelta

# Import from Phase 0
from cvrptw_parser import (
    ProblemInstance, 
    Solution, 
    Service, 
    Task, 
    Order,
    Location
)


# ===========================================================================
# DESTROY OPERATORS - Remove pickup-delivery pairs together
# ===========================================================================

class DestroyOperator:
    """Base class for destroy operators"""
    
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.name = "Base"
    
    def destroy(self, solution: Solution, k: int) -> List[Order]:
        """
        Remove k orders from solution
        Returns: List of removed orders (for repair)
        """
        raise NotImplementedError


class RandomRemoval(DestroyOperator):
    """Randomly remove k orders (and their pickup-delivery tasks)"""
    
    def __init__(self, problem):
        super().__init__(problem)
        self.name = "RandomRemoval"
    
    def destroy(self, solution: Solution, k: int) -> List[Order]:
        """Remove k random orders"""
        # Collect all orders currently in solution
        all_orders = []
        for service in solution.services:
            all_orders.extend(service.orders)
        
        if not all_orders:
            return []
        
        # Randomly select k orders to remove
        k = min(k, len(all_orders))
        orders_to_remove = random.sample(all_orders, k)
        
        # Remove BOTH pickup and delivery tasks for each order
        for order in orders_to_remove:
            self._remove_order_from_solution(solution, order)
        
        return orders_to_remove
    
    def _remove_order_from_solution(self, solution: Solution, order: Order):
        """Remove order's pickup and delivery tasks from all services"""
        for service in solution.services:
            if order in service.orders:
                # Remove from order list
                service.orders.remove(order)
                
                # Remove pickup and delivery tasks
                service.tasks = [
                    t for t in service.tasks 
                    if not (t.order and t.order.id == order.id)
                ]
                
                # Recalculate metrics after removal
                self._recalculate_service_metrics(service)
    
    def _recalculate_service_metrics(self, service: Service):
        """Recalculate distance, driving time, and duration for a service"""
        from datetime import timedelta
        
        total_distance = 0.0
        total_driving_time = 0.0
        
        # Start from depot
        prev_node = service.start_node
        current_time = service.start_time if service.start_time else self.problem.base_date.replace(hour=6, minute=0)
        
        for task in service.tasks:
            if task.node:
                # Travel to this task's location
                dist, travel_time = self.problem.get_travel_time(prev_node, task.node)
                total_distance += dist
                total_driving_time += travel_time
                
                # Update timing
                current_time += timedelta(minutes=travel_time)
                
                # Task execution
                if task.task_type == 'PICKUP':
                    task.start_time = current_time
                    task.duration = 10
                    task.end_time = current_time + timedelta(minutes=task.duration)
                    current_time = task.end_time
                elif task.task_type == 'DELIVERY':
                    task.start_time = current_time
                    task.duration = 10
                    task.end_time = current_time + timedelta(minutes=task.duration)
                    current_time = task.end_time
                elif task.task_type == 'BREAK':
                    task.start_time = current_time
                    task.duration = 30
                    task.end_time = current_time + timedelta(minutes=task.duration)
                    current_time = task.end_time
                
                prev_node = task.node
        
        # Return to depot
        dist, travel_time = self.problem.get_travel_time(prev_node, service.end_node)
        total_distance += dist
        total_driving_time += travel_time
        service.end_time = current_time + timedelta(minutes=travel_time)
        
        # Update service metrics
        service.total_distance = total_distance
        service.total_driving_time = total_driving_time


class WorstRemoval(DestroyOperator):
    """Remove orders with highest cost contribution"""
    
    def __init__(self, problem):
        super().__init__(problem)
        self.name = "WorstRemoval"
    
    def destroy(self, solution: Solution, k: int) -> List[Order]:
        """Remove k orders with highest removal savings"""
        # Calculate removal savings for each order
        order_costs = []
        
        for service in solution.services:
            for order in service.orders:
                # Calculate cost of this order in the route
                cost = self._calculate_order_cost(service, order)
                order_costs.append((order, cost, service))
        
        if not order_costs:
            return []
        
        # Sort by cost (highest first)
        order_costs.sort(key=lambda x: x[1], reverse=True)
        
        # Remove top k
        k = min(k, len(order_costs))
        removed_orders = []
        
        for order, cost, service in order_costs[:k]:
            self._remove_order_from_solution(solution, order)
            removed_orders.append(order)
        
        return removed_orders
    
    def _calculate_order_cost(self, service: Service, order: Order) -> float:
        """Calculate cost contribution of an order in a service"""
        # Find pickup and delivery positions
        pickup_idx = None
        delivery_idx = None
        
        for i, task in enumerate(service.tasks):
            if task.order and task.order.id == order.id:
                if task.task_type == 'PICKUP':
                    pickup_idx = i
                elif task.task_type == 'DELIVERY':
                    delivery_idx = i
        
        if pickup_idx is None or delivery_idx is None:
            return 0.0
        
        # Calculate distance cost
        cost = 0.0
        
        # Get nodes for distance calculation
        pickup_node = order.from_node
        delivery_node = order.to_node
        
        # Distance from pickup to delivery
        dist, _ = self.problem.get_travel_time(pickup_node, delivery_node)
        cost += dist
        
        # Add penalty for time window tightness
        time_window_width = (order.to_time - order.from_time).total_seconds() / 60
        if time_window_width < 120:  # Less than 2 hours
            cost += 50  # Penalty for tight windows
        
        return cost
    
    def _remove_order_from_solution(self, solution: Solution, order: Order):
        """Same as RandomRemoval"""
        for service in solution.services:
            if order in service.orders:
                service.orders.remove(order)
                service.tasks = [
                    t for t in service.tasks 
                    if not (t.order and t.order.id == order.id)
                ]


class ShawRemoval(DestroyOperator):
    """Remove related orders (similar in time/space)"""
    
    def __init__(self, problem):
        super().__init__(problem)
        self.name = "ShawRemoval"
    
    def destroy(self, solution: Solution, k: int) -> List[Order]:
        """Remove k related orders"""
        all_orders = []
        for service in solution.services:
            all_orders.extend(service.orders)
        
        if not all_orders:
            return []
        
        # Start with random seed order
        seed_order = random.choice(all_orders)
        removed = [seed_order]
        remaining = [o for o in all_orders if o != seed_order]
        
        # Iteratively add most related orders
        while len(removed) < k and remaining:
            # Find most related order to any removed order
            best_order = None
            best_relatedness = -float('inf')
            
            for order in remaining:
                relatedness = max(
                    self._relatedness(order, r) for r in removed
                )
                if relatedness > best_relatedness:
                    best_relatedness = relatedness
                    best_order = order
            
            if best_order:
                removed.append(best_order)
                remaining.remove(best_order)
        
        # Remove from solution
        for order in removed:
            self._remove_order_from_solution(solution, order)
        
        return removed
    
    def _relatedness(self, order1: Order, order2: Order) -> float:
        """Calculate relatedness between two orders (higher = more related)"""
        # Spatial proximity (pickup locations)
        _, dist1 = self.problem.get_travel_time(order1.from_node, order2.from_node)
        
        # Temporal proximity (time windows)
        time_diff = abs((order1.from_time - order2.from_time).total_seconds() / 60)
        
        # Combine (normalize and weight)
        spatial_score = 1.0 / (1.0 + dist1 / 60.0)  # Normalize distance
        temporal_score = 1.0 / (1.0 + time_diff / 180.0)  # Normalize time
        
        return 0.6 * spatial_score + 0.4 * temporal_score
    
    def _remove_order_from_solution(self, solution: Solution, order: Order):
        """Same as RandomRemoval"""
        for service in solution.services:
            if order in service.orders:
                service.orders.remove(order)
                service.tasks = [
                    t for t in service.tasks 
                    if not (t.order and t.order.id == order.id)
                ]


# ===========================================================================
# REPAIR OPERATORS - Insert pickup-delivery pairs with precedence
# ===========================================================================

class RepairOperator:
    """Base class for repair operators"""
    
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.name = "Base"
    
    def repair(self, solution: Solution, removed_orders: List[Order]) -> Solution:
        """
        Reinsert removed orders into solution
        Returns: Modified solution
        """
        raise NotImplementedError


class GreedyInsertion(RepairOperator):
    """Insert orders at position with minimum cost increase"""
    
    def __init__(self, problem):
        super().__init__(problem)
        self.name = "GreedyInsertion"
    
    def repair(self, solution: Solution, removed_orders: List[Order]) -> Solution:
        """Greedily insert each order at best position"""
        
        for order in removed_orders:
            best_insertion = self._find_best_insertion(solution, order)
            
            if best_insertion:
                service, p_pos, d_pos, cost = best_insertion
                self._insert_order(service, order, p_pos, d_pos)
                # Recalculate service metrics after insertion
                self._recalculate_service_metrics(service)
            else:
                # Cannot insert - add to unserved
                if order not in solution.unserved_orders:
                    solution.unserved_orders.append(order)
        
        return solution
    
    def _find_best_insertion(self, solution: Solution, order: Order) \
            -> Optional[Tuple[Service, int, int, float]]:
        """
        Find best (service, pickup_pos, delivery_pos) for inserting order
        
        Returns: (service, pickup_pos, delivery_pos, cost) or None
        """
        best_cost = float('inf')
        best_insertion = None
        
        # Try inserting into each service
        for service in solution.services:
            # Check vehicle compatibility
            if not service.vehicle.can_visit(order.from_node):
                continue
            if not service.vehicle.can_visit(order.to_node):
                continue
            
            # Try all valid (pickup_pos, delivery_pos) pairs
            n_tasks = len(service.tasks)
            
            for p_pos in range(n_tasks + 1):
                for d_pos in range(p_pos + 1, n_tasks + 2):
                    # Check if this insertion is feasible
                    if self._is_feasible_insertion(service, order, p_pos, d_pos):
                        cost = self._insertion_cost(service, order, p_pos, d_pos)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_insertion = (service, p_pos, d_pos, cost)
        
        return best_insertion
    
    def _is_feasible_insertion(self, service: Service, order: Order,
                              pickup_pos: int, delivery_pos: int) -> bool:
        """Check if inserting order at these positions is feasible"""
        
        # 1. Precedence: delivery must come after pickup
        if delivery_pos <= pickup_pos:
            return False
        
        # 2. Capacity check: simulate load profile
        test_tasks = service.tasks.copy()
        
        # Create temporary tasks
        pickup_task = Task('PICKUP', order.from_node, order)
        delivery_task = Task('DELIVERY', order.to_node, order)
        
        test_tasks.insert(pickup_pos, pickup_task)
        test_tasks.insert(delivery_pos, delivery_task)
        
        # Check capacity at each point
        current_load = 0.0
        for task in test_tasks:
            if task.task_type == 'PICKUP':
                current_load += task.order.capacity_required
            elif task.task_type == 'DELIVERY':
                current_load -= task.order.capacity_required
            
            if current_load > service.vehicle.capacity:
                return False  # Exceeds capacity
        
        # 3. Check duration constraint (simplified)
        # Estimate additional time
        additional_time = 20  # Pickup + delivery time (simplified)
        _, travel_time = self.problem.get_travel_time(order.from_node, order.to_node)
        additional_time += travel_time
        
        estimated_duration = service.duration_minutes() + additional_time
        if estimated_duration > self.problem.params.worktime_standard:
            return False
        
        return True
    
    def _insertion_cost(self, service: Service, order: Order,
                       pickup_pos: int, delivery_pos: int) -> float:
        """Calculate cost increase of inserting order"""
        
        # Cost = additional distance traveled
        cost = 0.0
        
        # Cost at pickup position
        if pickup_pos == 0:
            prev_node = service.start_node
        else:
            prev_node = service.tasks[pickup_pos - 1].node
        
        if pickup_pos < len(service.tasks):
            next_node = service.tasks[pickup_pos].node
        else:
            next_node = service.end_node
        
        # Calculate detour cost
        dist_old, _ = self.problem.get_travel_time(prev_node, next_node)
        dist_new1, _ = self.problem.get_travel_time(prev_node, order.from_node)
        dist_new2, _ = self.problem.get_travel_time(order.from_node, next_node)
        
        cost += (dist_new1 + dist_new2 - dist_old)
        
        # Cost at delivery position (similar calculation)
        # Simplified: just add pickup to delivery distance
        dist_pd, _ = self.problem.get_travel_time(order.from_node, order.to_node)
        cost += dist_pd
        
        return cost
    
    def _insert_order(self, service: Service, order: Order,
                     pickup_pos: int, delivery_pos: int):
        """Insert order's pickup and delivery tasks into service"""
        
        # SAFETY CHECK: Ensure delivery comes after pickup
        assert delivery_pos > pickup_pos, \
            f"PRECEDENCE VIOLATION: delivery_pos ({delivery_pos}) must be > pickup_pos ({pickup_pos})"
        
        # Create tasks
        pickup_task = Task('PICKUP', order.from_node, order)
        delivery_task = Task('DELIVERY', order.to_node, order)
        
        # Insert pickup first
        service.tasks.insert(pickup_pos, pickup_task)
        
        # Insert delivery (position shifts by 1 after pickup insertion)
        service.tasks.insert(delivery_pos, delivery_task)
        
        # Add to service orders
        service.orders.append(order)
        
        # VERIFY: Double-check precedence in final route
        p_idx = service.tasks.index(pickup_task)
        d_idx = service.tasks.index(delivery_task)
        assert d_idx > p_idx, \
            f"POST-INSERT VIOLATION: delivery at {d_idx} not after pickup at {p_idx}"
    
    def _recalculate_service_metrics(self, service: Service):
        """Recalculate distance, driving time, and duration for a service"""
        total_distance = 0.0
        total_driving_time = 0.0
        
        # Start from depot
        prev_node = service.start_node
        current_time = service.start_time
        
        for task in service.tasks:
            if task.node:
                # Travel to this task's location
                dist, travel_time = self.problem.get_travel_time(prev_node, task.node)
                total_distance += dist
                total_driving_time += travel_time
                
                # Update timing
                current_time += timedelta(minutes=travel_time)
                
                # Task execution
                if task.task_type == 'PICKUP':
                    task.start_time = current_time
                    task.duration = 10  # Simplified
                    task.end_time = current_time + timedelta(minutes=task.duration)
                    current_time = task.end_time
                elif task.task_type == 'DELIVERY':
                    task.start_time = current_time
                    task.duration = 10  # Simplified
                    task.end_time = current_time + timedelta(minutes=task.duration)
                    current_time = task.end_time
                elif task.task_type == 'BREAK':
                    task.start_time = current_time
                    task.duration = 30
                    task.end_time = current_time + timedelta(minutes=task.duration)
                    current_time = task.end_time
                
                prev_node = task.node
        
        # Return to depot
        dist, travel_time = self.problem.get_travel_time(prev_node, service.end_node)
        total_distance += dist
        total_driving_time += travel_time
        service.end_time = current_time + timedelta(minutes=travel_time)
        
        # Update service metrics
        service.total_distance = total_distance
        service.total_driving_time = total_driving_time


class Regret2Insertion(RepairOperator):
    """Prioritize hard-to-insert orders using regret metric"""
    
    def __init__(self, problem):
        super().__init__(problem)
        self.name = "Regret2Insertion"
        self.greedy = GreedyInsertion(problem)
    
    def repair(self, solution: Solution, removed_orders: List[Order]) -> Solution:
        """Insert orders by regret (difference between best and 2nd best)"""
        
        uninserted = removed_orders.copy()
        
        while uninserted:
            # Calculate regret for each uninserted order
            best_order = None
            best_regret = -float('inf')
            
            for order in uninserted:
                regret = self._calculate_regret(solution, order)
                if regret > best_regret:
                    best_regret = regret
                    best_order = order
            
            if best_order:
                # Insert the order with highest regret
                insertion = self.greedy._find_best_insertion(solution, best_order)
                
                if insertion:
                    service, p_pos, d_pos, cost = insertion
                    self.greedy._insert_order(service, best_order, p_pos, d_pos)
                    self.greedy._recalculate_service_metrics(service)  # Recalculate metrics
                    uninserted.remove(best_order)
                else:
                    # Cannot insert
                    if best_order not in solution.unserved_orders:
                        solution.unserved_orders.append(best_order)
                    uninserted.remove(best_order)
            else:
                break
        
        return solution
    
    def _calculate_regret(self, solution: Solution, order: Order) -> float:
        """Calculate regret = cost of 2nd best - cost of best insertion"""
        costs = []
        
        for service in solution.services:
            if not service.vehicle.can_visit(order.from_node):
                continue
            if not service.vehicle.can_visit(order.to_node):
                continue
            
            # Find best insertion cost in this service
            best_cost = float('inf')
            n_tasks = len(service.tasks)
            
            for p_pos in range(n_tasks + 1):
                for d_pos in range(p_pos + 1, n_tasks + 2):
                    if self.greedy._is_feasible_insertion(service, order, p_pos, d_pos):
                        cost = self.greedy._insertion_cost(service, order, p_pos, d_pos)
                        best_cost = min(best_cost, cost)
            
            if best_cost < float('inf'):
                costs.append(best_cost)
        
        # Regret = difference between best and second-best
        if len(costs) >= 2:
            costs.sort()
            return costs[1] - costs[0]
        elif len(costs) == 1:
            return costs[0]  # High priority if only one option
        else:
            return float('inf')  # Highest priority if no feasible insertion


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":
    print("PDP-Aware NLNS Operators")
    print("=" * 60)
    print("\n✓ Destroy operators remove pickup-delivery PAIRS")
    print("✓ Repair operators insert with PRECEDENCE constraints")
    print("✓ Allows INTERLEAVED routes: P1→P2→D1→D2")
    print("\nOperators implemented:")
    print("  Destroy: RandomRemoval, WorstRemoval, ShawRemoval")
    print("  Repair: GreedyInsertion, Regret2Insertion")