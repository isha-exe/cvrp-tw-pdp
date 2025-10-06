"""
test_phase2a.py
Complete test script for Phase 2a: RL-based ALNS

Tests simplified RL agent with DQN
"""

import sys

print("="*70)
print("PHASE 2a: RL-BASED ALNS TEST")
print("="*70)

# ============================================================================
# STEP 1: Import modules
# ============================================================================

print("\n[Step 1] Importing modules...")

try:
    from cvrptw_parser import ProblemInstance
    from constraint_validator import ConstraintValidator, ObjectiveCalculator
    from test_phase0 import SimpleGreedyConstructor
    from pdp_nlns_operators import (
        RandomRemoval, WorstRemoval, ShawRemoval,
        GreedyInsertion, Regret2Insertion
    )
    from simple_rl_agent import SimplifiedRLAgent, TORCH_AVAILABLE
    from rl_alns_integration import RLBasedALNS
    print("✓ All modules imported successfully")
    print(f"  PyTorch available: {TORCH_AVAILABLE}")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nMake sure these files exist:")
    print("  - cvrptw_parser.py")
    print("  - constraint_validator.py")
    print("  - test_phase0.py")
    print("  - pdp_nlns_operators.py")
    print("  - simple_rl_agent.py")
    print("  - rl_alns_integration.py")
    sys.exit(1)


# ============================================================================
# STEP 2: Load problem and build initial solution
# ============================================================================

print("\n[Step 2] Loading problem and building initial solution...")

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

# Build initial solution (quietly)
constructor = SimpleGreedyConstructor(problem)
print("  Building initial solution...")
initial_solution = constructor.construct(start_depot=1)

validator = ConstraintValidator(problem)
objective = ObjectiveCalculator(problem)

initial_cost, initial_components = objective.calculate(initial_solution)
_, initial_violations = validator.validate_solution(initial_solution)

print(f"✓ Initial solution created:")
print(f"  Services: {initial_components['num_services']}")
print(f"  Coverage: {initial_solution.coverage_rate(len(problem.orders))*100:.1f}%")
print(f"  Cost: {initial_cost:.2f}")
print(f"  Violations: {len(initial_violations)}")


# ============================================================================
# STEP 3: Setup RL-based ALNS
# ============================================================================

print(f"\n[Step 3] Setting up RL-based ALNS...")

# Create operators
destroy_ops = [
    RandomRemoval(problem),
    WorstRemoval(problem),
    ShawRemoval(problem)
]

repair_ops = [
    GreedyInsertion(problem),
    Regret2Insertion(problem)
]

print(f"✓ Operators created:")
print(f"  Destroy: {[op.name for op in destroy_ops]}")
print(f"  Repair: {[op.name for op in repair_ops]}")
print(f"  Total action space: {len(destroy_ops) * len(repair_ops)} pairs")

# Create RL-based ALNS
rl_alns = RLBasedALNS(
    problem=problem,
    destroy_operators=destroy_ops,
    repair_operators=repair_ops,
    use_rl=True
)

print(f"✓ RL-based ALNS initialized")
print(f"  State dimension: {rl_alns.rl_agent.state_dim}")
print(f"  Action dimension: {rl_alns.rl_agent.action_dim}")


# ============================================================================
# STEP 4: Train RL agent
# ============================================================================

print(f"\n[Step 4] Training RL agent...")

# Training parameters
best_solution, train_stats = rl_alns.train(
    initial_solution=initial_solution,
    num_episodes=5,           # Small number for quick test
    steps_per_episode=50,     # 50 steps per episode
    destroy_size_min=1,
    destroy_size_max=3,
    update_frequency=2        # Update target network every 2 episodes
)


# ============================================================================
# STEP 5: Evaluate final solution
# ============================================================================

print(f"\n[Step 5] Evaluating final solution...")

final_cost, final_components = objective.calculate(best_solution)
final_feasible, final_violations = validator.validate_solution(best_solution)

print(f"\n{'='*70}")
print(f"FINAL RESULTS COMPARISON")
print(f"{'='*70}")

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

# Violations
violations_change = len(final_violations) - len(initial_violations)
print(f"{'Violations':<25} {len(initial_violations):>14} {len(final_violations):>14} {violations_change:>+14}")

# Distance
distance_change = final_components['total_distance'] - initial_components['total_distance']
print(f"{'Distance (km)':<25} {initial_components['total_distance']:>14.2f} {final_components['total_distance']:>14.2f} {distance_change:>+14.2f}")

# Unserved
unserved_change = final_components['unserved_orders'] - initial_components['unserved_orders']
print(f"{'Unserved Orders':<25} {initial_components['unserved_orders']:>14} {final_components['unserved_orders']:>14} {unserved_change:>+14}")

print(f"\n{'-'*70}")


# ============================================================================
# STEP 6: Training statistics
# ============================================================================

print(f"\n{'='*70}")
print(f"TRAINING STATISTICS")
print(f"{'='*70}")

print(f"\nPer-Episode Metrics:")
print(f"{'Episode':<10} {'Cost':<15} {'Coverage':<12} {'Violations':<12} {'Reward':<12}")
print(f"{'-'*70}")

for i in range(len(train_stats['episode_costs'])):
    print(f"{i+1:<10} {train_stats['episode_costs'][i]:<15.2f} "
          f"{train_stats['episode_coverage'][i]:<12.1f} "
          f"{train_stats['episode_violations'][i]:<12} "
          f"{train_stats['episode_rewards'][i]:<12.2f}")

print(f"\nLearning Progress:")
print(f"  Best cost trajectory: {' → '.join([f'{c:.0f}' for c in train_stats['best_costs_per_episode']])}")


# ============================================================================
# STEP 7: Display best solution
# ============================================================================

print(f"\n{'='*70}")
print(f"BEST SOLUTION DETAILS")
print(f"{'='*70}")

for service in best_solution.services:
    if len(service.orders) > 0:  # Only show non-empty services
        print(f"\n┌─ Service {service.service_id} ───────────────────────")
        print(f"│ Vehicle: {service.vehicle.number} ({service.vehicle.vehicle_type.name})")
        print(f"│ Orders: {len(service.orders)}")
        print(f"│ Capacity: {sum(o.capacity_required for o in service.orders):.2f}/{service.vehicle.capacity}")
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

if best_solution.unserved_orders:
    print(f"\n⚠️  Unserved orders ({len(best_solution.unserved_orders)}):")
    for order in best_solution.unserved_orders:
        print(f"  Order {order.id}: {order.from_node}→{order.to_node}, cap={order.capacity_required:.2f}")


# ============================================================================
# STEP 8: Save trained agent (optional)
# ============================================================================

print(f"\n[Step 8] Saving trained agent...")

try:
    rl_alns.save_agent('rl_agent_checkpoint.pt')
    print("✓ Agent saved successfully")
except Exception as e:
    print(f"⚠️  Could not save agent: {e}")


# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"PHASE 2a: RL-BASED ALNS COMPLETE")
print(f"{'='*70}")

print(f"\n✓ Pipeline completed successfully!")

if final_cost < initial_cost:
    improvement_pct = (initial_cost - final_cost) / initial_cost * 100
    print(f"\n✓ Solution improved by {improvement_pct:.2f}%")
else:
    print(f"\n⚠️  Solution did not improve (may need more episodes)")

if len(final_violations) < len(initial_violations):
    print(f"✓ Violations reduced: {len(initial_violations)} → {len(final_violations)}")
elif len(final_violations) == len(initial_violations):
    print(f"= Violations unchanged: {len(final_violations)}")
else:
    print(f"⚠️  Violations increased: {len(initial_violations)} → {len(final_violations)}")

print(f"\nRL Agent Status:")
print(f"  Episodes trained: 5")
print(f"  Total steps: 250")
print(f"  Final epsilon: {rl_alns.rl_agent.epsilon:.3f}")
print(f"  Agent saved: rl_agent_checkpoint.pt")

print(f"\n{'='*70}")
print(f"Next steps:")
print(f"  1. Increase num_episodes for better learning (e.g., 20-50)")
print(f"  2. Tune reward function for your specific objectives")
print(f"  3. Upgrade to full JAMPR implementation (attention networks)")
print(f"{'='*70}\n")