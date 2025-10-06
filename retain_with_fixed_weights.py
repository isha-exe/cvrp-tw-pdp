"""
retrain_with_fixed_weights.py
Retrain RL agent with corrected objective weights

Key change: unserved_penalty 1000 → 10000
Goal: Maintain 91.7% coverage while optimizing cost
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*70)
print("RETRAINING WITH FIXED OBJECTIVE WEIGHTS")
print("="*70)
print("\nKey Change:")
print("  unserved_penalty: 1000 → 10,000 (10x increase)")
print("  Goal: Maintain coverage while reducing violations")
print("="*70)

# ============================================================================
# Import modules
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
    from simple_rl_agent import TORCH_AVAILABLE
    from rl_alns_integration import RLBasedALNS
    print("✓ All modules imported")
    print(f"  PyTorch: {TORCH_AVAILABLE}")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)


# ============================================================================
# Load problem
# ============================================================================

print("\n[Step 2] Loading problem...")

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

print(f"✓ Loaded: {len(problem.orders)} orders, {len(problem.vehicles)} vehicles")


# ============================================================================
# Build initial solution
# ============================================================================

print("\n[Step 3] Building initial solution...")

constructor = SimpleGreedyConstructor(problem)
initial_solution = constructor.construct(start_depot=1)

validator = ConstraintValidator(problem)
objective = ObjectiveCalculator(problem)  # Now using fixed weights

initial_cost, _ = objective.calculate(initial_solution)
_, initial_violations = validator.validate_solution(initial_solution)
initial_coverage = initial_solution.coverage_rate(len(problem.orders)) * 100

print(f"✓ Initial solution:")
print(f"  Cost: {initial_cost:.2f} (with NEW weights)")
print(f"  Coverage: {initial_coverage:.1f}%")
print(f"  Violations: {len(initial_violations)}")
print(f"  Unserved: {len(initial_solution.unserved_orders)}")


# ============================================================================
# Setup RL-ALNS
# ============================================================================

print(f"\n[Step 4] Setting up RL-ALNS...")

destroy_ops = [
    RandomRemoval(problem),
    WorstRemoval(problem),
    ShawRemoval(problem)
]

repair_ops = [
    GreedyInsertion(problem),
    Regret2Insertion(problem)
]

rl_alns = RLBasedALNS(
    problem=problem,
    destroy_operators=destroy_ops,
    repair_operators=repair_ops,
    use_rl=True
)

print(f"✓ RL-ALNS initialized")


# ============================================================================
# Training with fixed weights
# ============================================================================

print(f"\n[Step 5] Training with corrected objective...")
print(f"{'='*70}")
print(f"TRAINING CONFIGURATION")
print(f"{'='*70}")
print(f"Episodes: 20")
print(f"Steps per episode: 100")
print(f"Total iterations: 2000")
print(f"Objective: FIXED (unserved penalty 10x higher)")
print(f"Expected: Coverage should NOT drop below 91.7%")
print(f"{'='*70}")

best_solution, train_stats = rl_alns.train(
    initial_solution=initial_solution,
    num_episodes=20,
    steps_per_episode=100,
    destroy_size_min=1,
    destroy_size_max=3,
    update_frequency=5
)


# ============================================================================
# Results analysis
# ============================================================================

print(f"\n[Step 6] Analyzing results...")

final_cost, final_components = objective.calculate(best_solution)
_, final_violations = validator.validate_solution(best_solution)
final_coverage = best_solution.coverage_rate(len(problem.orders)) * 100

print(f"\n{'='*70}")
print(f"COMPARISON: INITIAL vs FIXED WEIGHTS")
print(f"{'='*70}")

print(f"\n{'Metric':<25} {'Initial':<15} {'Final':<15} {'Change':<15}")
print(f"{'-'*70}")

# Coverage - CRITICAL
coverage_change = final_coverage - initial_coverage
if coverage_change >= 0:
    coverage_status = "✓ MAINTAINED"
elif coverage_change > -5:
    coverage_status = "⚠️ SLIGHT DROP"
else:
    coverage_status = "✗ LOST"
print(f"{'Coverage (%)':<25} {initial_coverage:>14.1f} {final_coverage:>14.1f} {coverage_change:>+13.1f} {coverage_status}")

# Cost
cost_change_pct = ((final_cost - initial_cost) / initial_cost * 100) if initial_cost > 0 else 0
cost_status = "✓" if cost_change_pct < 0 else "="
print(f"{'Cost':<25} {initial_cost:>14.2f} {final_cost:>14.2f} {cost_change_pct:>+13.1f}% {cost_status}")

# Violations
viol_change = len(final_violations) - len(initial_violations)
viol_status = "✓ REDUCED" if viol_change < 0 else ("= SAME" if viol_change == 0 else "✗ INCREASED")
print(f"{'Violations':<25} {len(initial_violations):>14} {len(final_violations):>14} {viol_change:>+14} {viol_status}")

# Unserved
unserved_change = len(best_solution.unserved_orders) - len(initial_solution.unserved_orders)
unserved_status = "✓ LESS" if unserved_change < 0 else ("= SAME" if unserved_change == 0 else "✗ MORE")
print(f"{'Unserved Orders':<25} {len(initial_solution.unserved_orders):>14} {len(best_solution.unserved_orders):>14} {unserved_change:>+14} {unserved_status}")

print(f"\n{'-'*70}")


# ============================================================================
# Coverage analysis
# ============================================================================

print(f"\n{'='*70}")
print(f"COVERAGE ANALYSIS")
print(f"{'='*70}")

print(f"\nPer-Episode Coverage:")
min_coverage = min(train_stats['episode_coverage'])
max_coverage = max(train_stats['episode_coverage'])
avg_coverage = sum(train_stats['episode_coverage']) / len(train_stats['episode_coverage'])

print(f"  Minimum: {min_coverage:.1f}%")
print(f"  Maximum: {max_coverage:.1f}%")
print(f"  Average: {avg_coverage:.1f}%")
print(f"  Final: {final_coverage:.1f}%")

if min_coverage >= 90:
    print(f"\n✓ EXCELLENT: Coverage maintained above 90% throughout training")
elif min_coverage >= 80:
    print(f"\n✓ GOOD: Coverage stayed above 80%")
else:
    print(f"\n⚠️ CONCERN: Coverage dropped below 80% in some episodes")


# ============================================================================
# Best episodes
# ============================================================================

print(f"\n{'='*70}")
print(f"TOP 5 EPISODES")
print(f"{'='*70}")

episode_data = [
    (i+1, train_stats['episode_costs'][i], train_stats['episode_coverage'][i], 
     train_stats['episode_violations'][i], train_stats['episode_rewards'][i])
    for i in range(len(train_stats['episode_costs']))
]

# Sort by: 1) Coverage (desc), 2) Violations (asc), 3) Cost (asc)
episode_data.sort(key=lambda x: (-x[2], x[3], x[1]))

print(f"\nBest by Coverage + Feasibility:")
for rank, (ep, cost, cov, viol, reward) in enumerate(episode_data[:5], 1):
    print(f"  {rank}. Episode {ep:2d}: Coverage={cov:5.1f}%, Violations={viol}, "
          f"Cost={cost:8.2f}, Reward={reward:7.2f}")


# ============================================================================
# Visualizations
# ============================================================================

print(f"\n[Step 7] Creating plots...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Coverage (MOST IMPORTANT)
    axes[0, 0].plot(train_stats['episode_coverage'], 'g-o', linewidth=2)
    axes[0, 0].axhline(y=91.7, color='b', linestyle='--', label='Initial (91.7%)')
    axes[0, 0].axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Coverage (%)')
    axes[0, 0].set_title('Coverage Over Episodes (CRITICAL)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 105])
    
    # Plot 2: Cost
    axes[0, 1].plot(train_stats['episode_costs'], 'b-o', label='Episode Cost')
    axes[0, 1].plot(train_stats['best_costs_per_episode'], 'r--', linewidth=2, label='Best Cost')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Cost (with fixed weights)')
    axes[0, 1].set_title('Cost Progression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Violations
    axes[1, 0].plot(train_stats['episode_violations'], 'm-o')
    axes[1, 0].axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Initial (5)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Violations')
    axes[1, 0].set_title('Constraint Violations')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Rewards
    axes[1, 1].plot(train_stats['episode_rewards'], 'c-o')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].set_title('Episode Rewards')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_training_fixed_weights.png', dpi=150, bbox_inches='tight')
    print("✓ Plots saved: rl_training_fixed_weights.png")
except Exception as e:
    print(f"⚠️ Could not create plots: {e}")


# ============================================================================
# Save agent
# ============================================================================

print(f"\n[Step 8] Saving agent...")
try:
    rl_alns.save_agent('rl_agent_fixed_weights.pt')
    print("✓ Saved: rl_agent_fixed_weights.pt")
except Exception as e:
    print(f"⚠️ Could not save: {e}")


# ============================================================================
# Best solution details
# ============================================================================

print(f"\n{'='*70}")
print(f"BEST SOLUTION")
print(f"{'='*70}")

for service in best_solution.services:
    if len(service.orders) > 0:
        print(f"\nService {service.service_id}: {service.vehicle.number}")
        print(f"  Orders: {[o.id for o in service.orders]}")
        print(f"  Capacity: {sum(o.capacity_required for o in service.orders):.2f}/{service.vehicle.capacity}")
        print(f"  Duration: {service.duration_minutes():.1f}/{problem.params.worktime_standard} min")

if best_solution.unserved_orders:
    print(f"\n⚠️ Unserved: {[o.id for o in best_solution.unserved_orders]}")
else:
    print(f"\n✓ All orders served!")


# ============================================================================
# Final verdict
# ============================================================================

print(f"\n{'='*70}")
print(f"RETRAINING COMPLETE")
print(f"{'='*70}")

total_improvement = (initial_cost - final_cost) / initial_cost * 100

print(f"\nResults:")
print(f"  Cost improvement: {total_improvement:+.2f}%")
print(f"  Coverage: {initial_coverage:.1f}% → {final_coverage:.1f}% ({coverage_change:+.1f}%)")
print(f"  Violations: {len(initial_violations)} → {len(final_violations)} ({viol_change:+d})")

print(f"\nVerdict:")
if final_coverage >= initial_coverage and total_improvement > 0:
    print(f"  ✓ SUCCESS: Maintained coverage AND reduced cost")
elif final_coverage >= initial_coverage * 0.95:
    print(f"  ✓ GOOD: Coverage mostly maintained, some optimization")
elif final_coverage >= 85:
    print(f"  ⚠️ ACCEPTABLE: Some coverage loss, but still good")
else:
    print(f"  ✗ ISSUE: Too much coverage lost")

if len(final_violations) < len(initial_violations):
    print(f"  ✓ BONUS: Reduced violations")

print(f"\n{'='*70}")
print(f"Ready to move to Phase 2b if Phase 2a results are satisfactory")
print(f"{'='*70}\n")