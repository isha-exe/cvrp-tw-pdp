"""
constraint_validator.py
Constraint validation and objective calculation for CVRPTW-PD

Phase 0: Foundation
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime, timedelta

# Assumes cvrptw_parser classes are available
# from cvrptw_parser import ProblemInstance, Solution, Service, Task, Order, Location

@dataclass
class ConstraintViolation:
    """Record of a constraint violation"""
    service_id: int
    violation_type: str
    description: str
    severity: str  # 'HARD' or 'SOFT'

class ConstraintValidator:
    """Validates all problem constraints"""
    
    def __init__(self, problem: 'ProblemInstance'):
        self.problem = problem
    
    def validate_solution(self, solution: 'Solution') -> Tuple[bool, List[ConstraintViolation]]:
        """
        Validate complete solution against all constraints
        Returns: (is_feasible, list_of_violations)
        """
        violations = []
        
        for service in solution.services:
            violations.extend(self.validate_service(service))
        
        # Check depot constraint: number of services <= number of vehicles
        violations.extend(self._validate_depot_capacity(solution))
        
        is_feasible = all(v.severity != 'HARD' for v in violations)
        return is_feasible, violations
    
    def validate_service(self, service: 'Service') -> List[ConstraintViolation]:
        """Validate all constraints for a single service"""
        violations = []
        
        # A. Service Constraints
        violations.extend(self._validate_service_constraints(service))
        
        # B. Time Window Constraints
        violations.extend(self._validate_time_windows(service))
        
        # C. Vehicle Constraints
        violations.extend(self._validate_vehicle_constraints(service))
        
        # D. Break Constraints
        violations.extend(self._validate_break_constraints(service))
        
        # E. Precedence Constraints (Pickup before Delivery)
        violations.extend(self._validate_precedence(service))
        
        # F. Capacity Profile (dynamic capacity check)
        violations.extend(self._validate_capacity_profile(service))
        
        return violations
    
    def _validate_service_constraints(self, service: 'Service') -> List[ConstraintViolation]:
        """A. Service Constraints"""
        violations = []
        
        # A1. Must start and end at same location
        if service.start_node != service.end_node:
            violations.append(ConstraintViolation(
                service_id=service.service_id,
                violation_type='SERVICE_RETURN',
                description=f"Service doesn't return to origin (start={service.start_node}, end={service.end_node})",
                severity='HARD'
            ))
        
        # A2. Max duration 8 hours (480 min) including 30-min break
        duration = service.duration_minutes()
        max_duration = self.problem.params.worktime_standard
        if duration > max_duration:
            violations.append(ConstraintViolation(
                service_id=service.service_id,
                violation_type='SERVICE_DURATION',
                description=f"Service duration {duration:.1f} min exceeds max {max_duration} min",
                severity='HARD'
            ))
        
        return violations
    
    def _validate_time_windows(self, service: 'Service') -> List[ConstraintViolation]:
        """B. Time Window Constraints"""
        violations = []
        
        for task in service.tasks:
            if task.task_type == 'PICKUP' and task.order:
                order = task.order
                # Pickup must be within [from_time, to_time]
                if task.start_time:
                    if task.start_time < order.from_time:
                        violations.append(ConstraintViolation(
                            service_id=service.service_id,
                            violation_type='TIME_WINDOW_EARLY',
                            description=f"Order {order.id} pickup at {task.start_time.time()} before window start {order.from_time.time()}",
                            severity='HARD'
                        ))
                    if task.start_time > order.to_time:
                        violations.append(ConstraintViolation(
                            service_id=service.service_id,
                            violation_type='TIME_WINDOW_LATE',
                            description=f"Order {order.id} pickup at {task.start_time.time()} after window end {order.to_time.time()}",
                            severity='HARD'
                        ))
        
        return violations
    
    def _validate_vehicle_constraints(self, service: 'Service') -> List[ConstraintViolation]:
        """C. Vehicle Constraints"""
        violations = []
        
        # C1. Vehicle capacity (basic check - detailed check in capacity_profile)
        total_capacity = sum(order.capacity_required for order in service.orders)
        if total_capacity > service.vehicle.capacity:
            violations.append(ConstraintViolation(
                service_id=service.service_id,
                violation_type='VEHICLE_CAPACITY',
                description=f"Total capacity {total_capacity:.2f} exceeds vehicle capacity {service.vehicle.capacity}",
                severity='HARD'
            ))
        
        # C2. Vehicle-location restrictions
        for task in service.tasks:
            if task.task_type in ['PICKUP', 'DELIVERY'] and task.node:
                if not service.vehicle.can_visit(task.node):
                    violations.append(ConstraintViolation(
                        service_id=service.service_id,
                        violation_type='VEHICLE_RESTRICTION',
                        description=f"Vehicle {service.vehicle.number} cannot visit node {task.node}",
                        severity='HARD'
                    ))
        
        # C3. Check orders are compatible with vehicle
        for order in service.orders:
            if not service.vehicle.can_visit(order.from_node):
                violations.append(ConstraintViolation(
                    service_id=service.service_id,
                    violation_type='VEHICLE_RESTRICTION_PICKUP',
                    description=f"Vehicle {service.vehicle.number} cannot pickup order {order.id} at node {order.from_node}",
                    severity='HARD'
                ))
            if not service.vehicle.can_visit(order.to_node):
                violations.append(ConstraintViolation(
                    service_id=service.service_id,
                    violation_type='VEHICLE_RESTRICTION_DELIVERY',
                    description=f"Vehicle {service.vehicle.number} cannot deliver order {order.id} to node {order.to_node}",
                    severity='HARD'
                ))
        
        return violations
    
    def _validate_break_constraints(self, service: 'Service') -> List[ConstraintViolation]:
        """D. Break Constraints"""
        violations = []
        
        # D1. Must have exactly one 30-min break
        break_tasks = [t for t in service.tasks if t.task_type == 'BREAK']
        if len(break_tasks) == 0:
            violations.append(ConstraintViolation(
                service_id=service.service_id,
                violation_type='BREAK_MISSING',
                description="Service has no break assigned",
                severity='HARD'
            ))
        elif len(break_tasks) > 1:
            violations.append(ConstraintViolation(
                service_id=service.service_id,
                violation_type='BREAK_MULTIPLE',
                description=f"Service has {len(break_tasks)} breaks (should be 1)",
                severity='HARD'
            ))
        
        # D2. Break must be before 4h30 driving or 5h30 working
        if break_tasks and service.tasks:
            break_task = break_tasks[0]
            break_idx = service.tasks.index(break_task)
            
            # Calculate driving and working time before break
            driving_time_before = 0
            working_time_before = 0
            
            for i in range(break_idx):
                task = service.tasks[i]
                if task.task_type in ['PICKUP', 'DELIVERY']:
                    working_time_before += task.duration
                # Note: travel time between tasks not tracked in task object
                # This is a simplified check
            
            max_drive = self.problem.params.max_drive_time_before_break
            max_work = self.problem.params.max_work_time_before_break
            
            if working_time_before > max_work:
                violations.append(ConstraintViolation(
                    service_id=service.service_id,
                    violation_type='BREAK_WORK_TIME',
                    description=f"Break after {working_time_before} min working (max {max_work})",
                    severity='HARD'
                ))
        
        # D3. Break must be at allowed location
        if break_tasks:
            break_task = break_tasks[0]
            if break_task.node:
                location = self.problem.locations.get(break_task.node)
                if location and not location.break_allowed:
                    violations.append(ConstraintViolation(
                        service_id=service.service_id,
                        violation_type='BREAK_LOCATION',
                        description=f"Break at node {break_task.node} where breaks not allowed",
                        severity='HARD'
                    ))
        
        return violations
    
    def _validate_precedence(self, service: 'Service') -> List[ConstraintViolation]:
        """E. Precedence Constraints - Pickup must come before Delivery"""
        violations = []
        
        for order in service.orders:
            pickup_idx = None
            delivery_idx = None
            
            for i, task in enumerate(service.tasks):
                if task.order and task.order.id == order.id:
                    if task.task_type == 'PICKUP':
                        pickup_idx = i
                    elif task.task_type == 'DELIVERY':
                        delivery_idx = i
            
            # Check precedence
            if delivery_idx is not None and pickup_idx is not None:
                if delivery_idx <= pickup_idx:
                    violations.append(ConstraintViolation(
                        service_id=service.service_id,
                        violation_type='PRECEDENCE',
                        description=f"Order {order.id}: Delivery at position {delivery_idx} before/equal pickup at {pickup_idx}",
                        severity='HARD'
                    ))
        
        return violations
    
    def _validate_capacity_profile(self, service: 'Service') -> List[ConstraintViolation]:
        """F. Capacity Profile - Check capacity at EACH point in route"""
        violations = []
        current_load = 0.0
        
        for i, task in enumerate(service.tasks):
            if task.task_type == 'PICKUP' and task.order:
                current_load += task.order.capacity_required
            elif task.task_type == 'DELIVERY' and task.order:
                current_load -= task.order.capacity_required
            
            if current_load > service.vehicle.capacity:
                violations.append(ConstraintViolation(
                    service_id=service.service_id,
                    violation_type='CAPACITY_PROFILE',
                    description=f"Load {current_load:.2f} exceeds capacity {service.vehicle.capacity} at task {i}",
                    severity='HARD'
                ))
            
            if current_load < -0.01:  # Small tolerance for floating point
                violations.append(ConstraintViolation(
                    service_id=service.service_id,
                    violation_type='CAPACITY_NEGATIVE',
                    description=f"Negative load {current_load:.2f} at task {i}",
                    severity='HARD'
                ))
        
        return violations
    
    def _validate_depot_capacity(self, solution: 'Solution') -> List[ConstraintViolation]:
        """Check that services per depot <= vehicles available"""
        violations = []
        
        # Simplified: assume all vehicles can start from any depot
        total_vehicles = len(self.problem.vehicles)
        total_services = len(solution.services)
        
        if total_services > total_vehicles:
            violations.append(ConstraintViolation(
                service_id=-1,
                violation_type='DEPOT_CAPACITY',
                description=f"Number of services ({total_services}) exceeds available vehicles ({total_vehicles})",
                severity='HARD'
            ))
        
        return violations


class ObjectiveCalculator:
    """Calculate objective function value"""
    
    def __init__(self, problem: 'ProblemInstance', weights: Dict[str, float] = None):
        self.problem = problem
        self.validator = ConstraintValidator(problem)  # Create validator instance
        
        # Default weights for multi-objective optimization
        self.weights = weights or {
            'num_services': 100.0,      # Minimize services
            'num_vehicles': 150.0,      # Minimize vehicles used
            'total_distance': 1.0,      # Minimize distance
            'total_idle_time': 0.5,     # Minimize idle time
            'unserved_penalty': 10000.0  # CRITICAL: High penalty to maintain coverage
        }
    
    def calculate(self, solution: 'Solution') -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted objective value with feasibility penalty
        Returns: (total_cost, component_costs)
        """
        components = {
            'num_services': solution.num_services(),
            'num_vehicles': solution.num_vehicles_used(),
            'total_distance': solution.total_distance(),
            'total_idle_time': solution.total_idle_time(),
            'unserved_orders': len(solution.unserved_orders)
        }
        
        # Calculate weighted cost
        total_cost = 0.0
        for key, value in components.items():
            if key in self.weights:
                total_cost += self.weights[key] * value
        
        # CRITICAL: Add massive penalty for constraint violations
        is_feasible, violations = self.validator.validate_solution(solution)
        
        if not is_feasible:
            # Heavy penalty for each violation (5000 per violation)
            violation_penalty = len(violations) * 5000.0
            total_cost += violation_penalty
        
        return total_cost, components
    
    def print_summary(self, solution: 'Solution'):
        """Print detailed solution summary"""
        cost, components = self.calculate(solution)
        
        print("\n" + "=" * 60)
        print("SOLUTION SUMMARY")
        print("=" * 60)
        print(f"Total Cost: {cost:.2f}")
        print(f"\nMetrics:")
        print(f"  Services: {components['num_services']}")
        print(f"  Vehicles Used: {components['num_vehicles']}")
        print(f"  Total Distance: {components['total_distance']:.2f} km")
        print(f"  Total Idle Time: {components['total_idle_time']:.2f} min")
        print(f"  Unserved Orders: {components['unserved_orders']}/{len(self.problem.orders)}")
        print(f"  Coverage: {solution.coverage_rate(len(self.problem.orders))*100:.1f}%")
        
        # Average metrics
        if components['num_services'] > 0:
            print(f"\nAverage per Service:")
            print(f"  Distance: {components['total_distance']/components['num_services']:.2f} km")
            print(f"  Idle Time: {components['total_idle_time']/components['num_services']:.2f} min")
        
        print("=" * 60)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from cvrptw_parser import ProblemInstance, Solution, Service, Order
    
    # Load problem
    file_paths = {
        'receptacles': 'Receptacles.csv',
        'vehicle_types': 'VehicleTypes.csv',
        'vehicle_restrictions': 'VehicleRestrictions.csv',
        'fleet': 'Fleet.csv',
        'break_allowed': 'BreakAllowed.csv',
        'orders': 'Orders.csv',
        'travel_times': 'TravelTimes.csv',
        'service_items': 'ServiceItems.csv'
    }
    
    problem = ProblemInstance()
    problem.load_from_files(file_paths)
    
    # Create validator and objective calculator
    validator = ConstraintValidator(problem)
    objective = ObjectiveCalculator(problem)
    
    # Example: Create a dummy solution for testing
    solution = Solution()
    solution.unserved_orders = problem.orders.copy()
    
    # Validate and calculate objective
    is_feasible, violations = validator.validate_solution(solution)
    
    print(f"Solution Feasible: {is_feasible}")
    print(f"Violations: {len(violations)}")
    
    objective.print_summary(solution)