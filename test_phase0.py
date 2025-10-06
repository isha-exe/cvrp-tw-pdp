"""
PHASE 0: Complete Test Script
Run this to verify all Phase 0 components work correctly

Requirements:
- cvrptw_parser.py in same directory
- All 8 CSV files in same directory

Usage:
    python test_phase0.py
"""

import sys
from datetime import datetime, timedelta
from typing import List, Optional

print("="*70)
print("PHASE 0: FOUNDATION TEST")
print("="*70)

# ============================================================================
# STEP 1: Import Core Components
# ============================================================================

print("\n[Step 1] Importing modules...")

try:
    from cvrptw_parser import (
        ProblemInstance, 
        Solution, 
        Service, 
        Task, 
        Order, 
        Vehicle,
        Location
    )
    print("✓ Successfully imported cvrptw_parser")
except ImportError as e:
    print(f"✗ ERROR: Could not import from cvrptw_parser.py")
    print(f"  {e}")
    print("\n  Make sure cvrptw_parser.py is in the same directory!")
    sys.exit(1)

try:
    from constraint_validator import ConstraintValidator, ObjectiveCalculator
    print("✓ Successfully imported constraint_validator")
except ImportError as e:
    print(f"✗ ERROR: Could not import from constraint_validator.py")
    print(f"  {e}")
    print("\n  Make sure constraint_validator.py is in the same directory!")
    sys.exit(1)


# ============================================================================
# STEP 2: Simple Greedy Constructor
# ============================================================================

class SimpleGreedyConstructor:
    """Simple greedy constructor for Phase 0 baseline"""
    
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
    
    def construct(self, start_depot: int = 1) -> Solution:
        """Build initial solution using greedy nearest-neighbor"""
        solution = Solution()
        unassigned = self.problem.orders.copy()
        service_id = 0
        
        print(f"\n{'='*70}")
        print(f"CONSTRUCTING INITIAL SOLUTION")
        print(f"{'='*70}")
        print(f"Start depot: Node {start_depot}")
        print(f"Orders to assign: {len(unassigned)}")
        
        # Sort orders by earliest time window
        unassigned.sort(key=lambda o: o.from_time)
        
        # Try to assign all orders
        while unassigned and service_id < len(self.problem.vehicles):
            service_id += 1
            
            # Get next available vehicle
            vehicle = self.problem.vehicles[service_id - 1]
            
            print(f"\n┌─ Service {service_id} ───────────────────────")
            print(f"│ Vehicle: {vehicle.number} ({vehicle.vehicle_type.name})")
            print(f"│ Capacity: {vehicle.capacity}")
            
            # Create service
            service = Service(
                service_id=service_id,
                vehicle=vehicle,
                start_node=start_depot,
                end_node=start_depot,
                start_time=self.problem.base_date.replace(hour=6, minute=0)
            )
            
            # Try to insert orders
            inserted = self._fill_service(service, unassigned)
            
            if inserted:
                # Calculate metrics
                self._calculate_metrics(service)
                
                # Add simple break
                self._add_break(service)
                
                solution.services.append(service)
                
                # Remove inserted orders
                for order in inserted:
                    unassigned.remove(order)
                
                print(f"│ Orders assigned: {len(inserted)}")
                print(f"│ Total capacity used: {sum(o.capacity_required for o in inserted):.2f}/{vehicle.capacity}")
                print(f"│ Distance: {service.total_distance:.2f} km")
                print(f"│ Duration: {service.duration_minutes():.1f} min")
                print(f"└{'─'*42}")
            else:
                print(f"│ ✗ No compatible orders for this vehicle")
                print(f"└{'─'*42}")
        
        solution.unserved_orders = unassigned
        
        return solution
    
    def _fill_service(self, service: Service, available: List[Order]) -> List[Order]:
        """Fill service with compatible orders"""
        inserted = []
        current_time = service.start_time
        current_node = service.start_node
        current_capacity = 0.0
        
        for order in available:
            # Skip if already inserted
            if order in inserted:
                continue
            
            # Check vehicle compatibility
            if not service.vehicle.can_visit(order.from_node):
                continue
            if not service.vehicle.can_visit(order.to_node):
                continue
            
            # Check capacity
            if current_capacity + order.capacity_required > service.vehicle.capacity:
                continue
            
            # Check time feasibility
            _, travel_to_pickup = self.problem.get_travel_time(current_node, order.from_node)
            arrival_at_pickup = current_time + timedelta(minutes=travel_to_pickup)
            
            if arrival_at_pickup > order.to_time:
                continue  # Too late
            
            # INSERT THE ORDER
            pickup_start = max(arrival_at_pickup, order.from_time)
            pickup_duration = 10  # Simplified
            pickup_end = pickup_start + timedelta(minutes=pickup_duration)
            
            # Pickup task
            service.tasks.append(Task(
                task_type='PICKUP',
                node=order.from_node,
                order=order,
                start_time=pickup_start,
                end_time=pickup_end,
                duration=pickup_duration
            ))
            
            # Delivery task
            _, travel_to_delivery = self.problem.get_travel_time(order.from_node, order.to_node)
            delivery_start = pickup_end + timedelta(minutes=travel_to_delivery)
            delivery_duration = 10  # Simplified
            delivery_end = delivery_start + timedelta(minutes=delivery_duration)
            
            service.tasks.append(Task(
                task_type='DELIVERY',
                node=order.to_node,
                order=order,
                start_time=delivery_start,
                end_time=delivery_end,
                duration=delivery_duration
            ))
            
            service.orders.append(order)
            inserted.append(order)
            current_capacity += order.capacity_required
            current_time = delivery_end
            current_node = order.to_node
        
        # Return to depot
        if current_node != service.end_node:
            _, travel_home = self.problem.get_travel_time(current_node, service.end_node)
            service.end_time = current_time + timedelta(minutes=travel_home)
        else:
            service.end_time = current_time
        
        return inserted
    
    def _calculate_metrics(self, service: Service):
        """Calculate service metrics"""
        total_distance = 0.0
        total_time = 0.0
        
        prev_node = service.start_node
        
        for task in service.tasks:
            if task.node:
                dist, time = self.problem.get_travel_time(prev_node, task.node)
                total_distance += dist
                total_time += time
                prev_node = task.node
        
        # Return to depot
        dist, time = self.problem.get_travel_time(prev_node, service.end_node)
        total_distance += dist
        total_time += time
        
        service.total_distance = total_distance
        service.total_driving_time = total_time
    
    def _add_break(self, service: Service):
        """Add a simple break"""
        if service.tasks:
            # Insert break in the middle
            mid_idx = len(service.tasks) // 2
            if mid_idx < len(service.tasks):
                prev_task = service.tasks[mid_idx]
                break_start = prev_task.end_time
                break_task = Task(
                    task_type='BREAK',
                    node=service.start_node,  # Break at depot
                    start_time=break_start,
                    end_time=break_start + timedelta(minutes=30),
                    duration=30
                )
                service.tasks.insert(mid_idx, break_task)
                service.break_assigned = True


# ============================================================================
# STEP 3: Load Problem Data
# ============================================================================

print(f"\n[Step 2] Loading problem data...")

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

try:
    problem.load_from_files(file_paths)
    print("✓ All CSV files loaded successfully")
except FileNotFoundError as e:
    print(f"✗ ERROR: Could not find file")
    print(f"  {e}")
    print("\n  Make sure all CSV files are in the same directory:")
    for name, path in file_paths.items():
        print(f"    - {path}")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: Failed to load data")
    print(f"  {e}")
    sys.exit(1)

# Print problem summary
print(f"\n[Step 3] Problem instance summary:")
print(f"  Orders: {len(problem.orders)}")
print(f"  Vehicles: {len(problem.vehicles)}")
print(f"  Locations: {len(problem.locations)}")
print(f"  Total capacity demand: {sum(o.capacity_required for o in problem.orders):.2f}")
print(f"  Total fleet capacity: {sum(v.capacity for v in problem.vehicles):.2f}")


# ============================================================================
# STEP 4: Build Initial Solution
# ============================================================================

print(f"\n[Step 4] Building initial solution...")

constructor = SimpleGreedyConstructor(problem)
solution = constructor.construct(start_depot=1)


# ============================================================================
# STEP 5: Validate Solution
# ============================================================================

print(f"\n[Step 5] Validating solution...")

validator = ConstraintValidator(problem)
is_feasible, violations = validator.validate_solution(solution)

print(f"\n{'='*70}")
print(f"VALIDATION RESULTS")
print(f"{'='*70}")
print(f"Feasible: {'✓ YES' if is_feasible else '✗ NO'}")
print(f"Violations found: {len(violations)}")

if violations:
    print(f"\nConstraint Violations:")
    for i, v in enumerate(violations[:15], 1):  # Show first 15
        print(f"  {i}. [{v.severity}] Service {v.service_id}: {v.violation_type}")
        print(f"     {v.description}")
    
    if len(violations) > 15:
        print(f"\n  ... and {len(violations) - 15} more violations")
else:
    print(f"✓ No constraint violations detected")


# ============================================================================
# STEP 6: Calculate Objective
# ============================================================================

print(f"\n[Step 6] Calculating objective...")

objective = ObjectiveCalculator(problem)
total_cost, components = objective.calculate(solution)

print(f"\n{'='*70}")
print(f"SOLUTION METRICS")
print(f"{'='*70}")
print(f"Total Cost: {total_cost:.2f}")
print(f"\nComponents:")
print(f"  Services: {components['num_services']}")
print(f"  Vehicles Used: {components['num_vehicles']}")
print(f"  Total Distance: {components['total_distance']:.2f} km")
print(f"  Total Driving Time: {solution.total_driving_time():.2f} min")
print(f"  Total Idle Time: {components['total_idle_time']:.2f} min")
print(f"  Unserved Orders: {components['unserved_orders']}/{len(problem.orders)}")
print(f"  Coverage: {solution.coverage_rate(len(problem.orders))*100:.1f}%")


# ============================================================================
# STEP 7: Display Service Details
# ============================================================================

print(f"\n{'='*70}")
print(f"SERVICE DETAILS")
print(f"{'='*70}")

for service in solution.services:
    print(f"\n┌─ Service {service.service_id} ───────────────────────")
    print(f"│ Vehicle: {service.vehicle.number} ({service.vehicle.vehicle_type.name})")
    print(f"│ Capacity: {service.vehicle.capacity}")
    print(f"│ Orders: {len(service.orders)}")
    print(f"│ Distance: {service.total_distance:.2f} km")
    print(f"│ Driving Time: {service.total_driving_time:.2f} min")
    print(f"│ Duration: {service.duration_minutes():.1f} min")
    print(f"│ Start: {service.start_time.time()}")
    print(f"│ End: {service.end_time.time()}")
    
    # Build route string
    route_parts = [str(service.start_node)]
    for task in service.tasks:
        if task.task_type == 'PICKUP':
            route_parts.append(f"P{task.order.id}")
        elif task.task_type == 'DELIVERY':
            route_parts.append(f"D{task.order.id}")
        elif task.task_type == 'BREAK':
            route_parts.append("BREAK")
    route_parts.append(str(service.end_node))
    
    print(f"│ Route: {' → '.join(route_parts)}")
    print(f"└{'─'*50}")

# Unserved orders
if solution.unserved_orders:
    print(f"\n⚠️  UNSERVED ORDERS: {len(solution.unserved_orders)}")
    for order in solution.unserved_orders:
        print(f"  Order {order.id}: {order.from_node}→{order.to_node}, "
              f"cap={order.capacity_required:.2f}, "
              f"window={order.from_time.time()}-{order.to_time.time()}")


# ============================================================================
# STEP 8: Summary
# ============================================================================

print(f"\n{'='*70}")
print(f"PHASE 0 TEST COMPLETE")
print(f"{'='*70}")

print(f"\n✓ All components working correctly!")
print(f"\nPhase 0 Status:")
print(f"  [✓] Data loading")
print(f"  [✓] Data structures")
print(f"  [✓] Greedy constructor")
print(f"  [✓] Constraint validation")
print(f"  [✓] Objective calculation")

print(f"\nSolution Quality:")
if solution.coverage_rate(len(problem.orders)) >= 0.9:
    print(f"  Coverage: {solution.coverage_rate(len(problem.orders))*100:.1f}% - GOOD ✓")
elif solution.coverage_rate(len(problem.orders)) >= 0.7:
    print(f"  Coverage: {solution.coverage_rate(len(problem.orders))*100:.1f}% - ACCEPTABLE")
else:
    print(f"  Coverage: {solution.coverage_rate(len(problem.orders))*100:.1f}% - NEEDS IMPROVEMENT")

if is_feasible:
    print(f"  Feasibility: All constraints satisfied ✓")
else:
    print(f"  Feasibility: {len(violations)} violations found ⚠️")

print(f"\n{'='*70}")
print(f"READY FOR PHASE 1: ALNS Optimization")
print(f"{'='*70}")
print(f"\nNext steps:")
print(f"  1. Review the solution metrics above")
print(f"  2. Check unserved orders (if any)")
print(f"  3. Proceed to Phase 1 to improve the solution")
print(f"\n")