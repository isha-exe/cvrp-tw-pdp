"""
test_phase1.py
Complete test script for Phase 1: ALNS Optimization

Combines Phase 0 foundation with ALNS framework
"""

import sys

print("="*70)
print("PHASE 1: ALNS TEST")
print("="*70)

# ============================================================================
# STEP 1: Import all components
# ============================================================================

print("\n[Step 1] Importing modules...")

try:
    from cvrptw_parser import ProblemInstance, Solution, Service, Task, Order
    from constraint_validator import ConstraintValidator, ObjectiveCalculator
    from test_phase0 import SimpleGreedyConstructor
    from pdp_nlns_operators import (
        RandomRemoval, WorstRemoval, ShawRemoval,
        GreedyInsertion, Regret2Insertion
    )
    from alns_framework import ALNSFramework
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nMake sure these files are in the same directory:")
    print("  - cvrptw_parser.py")
    print("  - constraint_validator.py")
    print("  - test_phase0.py")
    print("  - pdp_nlns_operators.py")
    print("  - alns_framework.py")
    sys.exit(1)


# ============================================================================
# STEP 2: Load problem and build initial solution
# ============================================================================

print("\n[Step 2] Loading problem data and building initial solution...")

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

try:
    problem = ProblemInstance()
    problem.load_from_files(file_paths)
    print("✓ Problem data loaded")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

# Build initial solution
constructor = SimpleGreedyConstructor(problem)
initial_solution = constructor.construct(start_depot=1)

# Validate initial solution
validator = ConstraintValidator(problem)
objective = ObjectiveCalculator(problem)

initial_feasible, initial_violations = validator.validate_solution(initial_solution)
initial_cost, initial_components = objective.calculate(initial_solution)

print(f"\n✓ Initial solution created:")
print(f"  Services: {initial_components['num_services']}")
print(f"  Coverage: {initial_solution.coverage_rate(len(problem.orders))*100:.1f}%")
print(f"  Cost: {initial_cost:.2f}")
print(f"  Violations: {len(initial_violations)}")


# ============================================================================
# STEP 3: Setup ALNS
# ============================================================================

print(f"\n[Step 3] Setting up ALNS operators...")

# Create destroy operators
destroy_ops = [
    RandomRemoval(problem),
    WorstRemoval(problem),
    ShawRemoval(problem)
]

# Create repair operators
repair_ops = [
    GreedyInsertion(problem),
    Regret2Insertion(problem)
]

print(f"✓ Operators created:")
print(f"  Destroy: {[op.name for op in destroy_ops]}")
print(f"  Repair: {[op.name for op in repair_ops]}")

# Create ALNS framework
alns = ALNSFramework(
    problem=problem,
    destroy_operators=destroy_ops,
    repair_operators=repair_ops,
    seed=42
)

print(f"✓ ALNS framework initialized")


# ============================================================================
# STEP 4: Run ALNS Optimization
# ============================================================================

print(f"\n[Step 4] Running ALNS optimization...")

# Run optimization
best_solution, stats = alns.optimize(
    initial_solution=initial_solution,
    max_iterations=500,          # Moderate number for testing
    temperature_start=1000.0,
    temperature_end=0.1,
    cooling_rate=0.995,
    destroy_size_min=1,
    destroy_size_max=3
)


# ============================================================================
# STEP 5: Analyze results
# ============================================================================

print(f"\n{'='*70}")
print(f"FINAL RESULTS COMPARISON")
print(f"{'='*70}")

# Validate final solution
final_feasible, final_violations = validator.validate_solution(best_solution)
final_cost, final_components = objective.calculate(best_solution)

# Print comparison table
print(f"\n{'Metric':<25} {'Initial':<15} {'Final':<15} {'Change':<15}")
print(f"{'-'*70}")

# Coverage
initial_coverage = initial_solution.coverage_rate(len(problem.orders)) * 100
final_coverage = best_solution.coverage_rate(len(problem.orders)) * 100
coverage_change = final_coverage - initial_coverage
print(f"{'Coverage (%)':<25} {initial_coverage:>14.1f} {final_coverage:>14.1f} {coverage_change:>+14.1f}")

# Cost
cost_change = final_cost - initial_cost
cost_change_pct = (cost_change / initial_cost * 100) if initial_cost > 0 else 0
print(f"{'Total Cost':<25} {initial_cost:>14.2f} {final_cost:>14.2f} {cost_change_pct:>+13.1f}%")

# Services
services_change = final_components['num_services'] - initial_components['num_services']
print(f"{'Services':<25} {initial_components['num_services']:>14} {final_components['num_services']:>14} {services_change:>+14}")

# Vehicles
vehicles_change = final_components['num_vehicles'] - initial_components['num_vehicles']
print(f"{'Vehicles Used':<25} {initial_components['num_vehicles']:>14} {final_components['num_vehicles']:>14} {vehicles_change:>+14}")

# Distance
distance_change = final_components['total_distance'] - initial_components['total_distance']
print(f"{'Distance (km)':<25} {initial_components['total_distance']:>14.2f} {final_components['total_distance']:>14.2f} {distance_change:>+14.2f}")

# Violations
violations_change = len(final_violations) - len(initial_violations)
print(f"{'Violations':<25} {len(initial_violations):>14} {len(final_violations):>14} {violations_change:>+14}")

# Feasibility
print(f"\n{'Feasibility':<25} {'✗ NO' if not initial_feasible else '✓ YES':<15} {'✗ NO' if not final_feasible else '✓ YES':<15}")

print(f"\n{'-'*70}")

# Show remaining violations if any
if final_violations:
    print(f"\n⚠️  Remaining violations ({len(final_violations)}):")
    for i, v in enumerate(final_violations[:5], 1):
        print(f"  {i}. [{v.severity}] Service {v.service_id}: {v.violation_type}")
        print(f"     {v.description}")
    if len(final_violations) > 5:
        print(f"  ... and {len(final_violations) - 5} more")
else:
    print(f"\n✓ Solution is fully feasible!")

# Show unserved orders
if best_solution.unserved_orders:
    print(f"\n⚠️  Unserved orders ({len(best_solution.unserved_orders)}):")
    for order in best_solution.unserved_orders:
        print(f"  Order {order.id}: {order.from_node}→{order.to_node}, "
              f"cap={order.capacity_required:.2f}")
else:
    print(f"\n✓ All orders served!")


# ============================================================================
# STEP 6: Display best solution details
# ============================================================================

print(f"\n{'='*70}")
print(f"BEST SOLUTION DETAILS")
print(f"{'='*70}")

for service in best_solution.services:
    print(f"\n┌─ Service {service.service_id} ───────────────────────")
    print(f"│ Vehicle: {service.vehicle.number} ({service.vehicle.vehicle_type.name})")
    print(f"│ Capacity: {service.vehicle.capacity}")
    print(f"│ Orders: {len(service.orders)}")
    print(f"│ Capacity used: {sum(o.capacity_required for o in service.orders):.2f}/{service.vehicle.capacity}")
    print(f"│ Distance: {service.total_distance:.2f} km")
    print(f"│ Duration: {service.duration_minutes():.1f} min / {problem.params.worktime_standard} min")
    
    # Build route
    route_parts = [str(service.start_node)]
    for task in service.tasks:
        if task.task_type == 'PICKUP':
            route_parts.append(f"P{task.order.id}")
        elif task.task_type == 'DELIVERY':
            route_parts.append(f"D{task.order.id}")
        elif task.task_type == 'BREAK':
            route_parts.append("BRK")
    route_parts.append(str(service.end_node))
    
    print(f"│ Route: {' → '.join(route_parts)}")
    print(f"└{'─'*50}")


# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"PHASE 1: ALNS TEST COMPLETE")
print(f"{'='*70}")

print(f"\n✓ ALNS optimization completed successfully!")

if final_coverage >= 95 and len(final_violations) == 0:
    print(f"\n🎉 EXCELLENT SOLUTION!")
    print(f"   - High coverage ({final_coverage:.1f}%)")
    print(f"   - Fully feasible")
elif final_coverage >= 90:
    print(f"\n✓ GOOD SOLUTION!")
    print(f"   - Good coverage ({final_coverage:.1f}%)")
    if len(final_violations) > 0:
        print(f"   - Some violations remain (consider more iterations)")
else:
    print(f"\n⚠️  SOLUTION NEEDS IMPROVEMENT")
    print(f"   - Coverage: {final_coverage:.1f}%")
    print(f"   - Violations: {len(final_violations)}")

print(f"\n{'='*70}")