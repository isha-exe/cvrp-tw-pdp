"""
test_phase2a_improved.py
Improved training with longer episodes and better reward shaping

Phase 2a: Extended training for better results
"""

import sys
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

print("="*70)
print("PHASE 2a: IMPROVED RL TRAINING")
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
    sys.exit(1)


# ============================================================================
# STEP 2: Load problem
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

try:
    problem = ProblemInstance()
    problem.load_from_files(file_paths)
    print("✓ Problem data loaded")
    print(f"  Orders: {len(problem.orders)}")
    print(f"  Vehicles: {len(problem.vehicles)}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)


# ============================================================================
# STEP 3: Build initial solution
# ============================================================================

print("\n[Step 3] Building initial solution...")

constructor = SimpleGreedyConstructor(problem)
initial_solution = constructor.construct(start_depot=1)

validator = ConstraintValidator(problem)
objective = ObjectiveCalculator(problem)

initial_cost, _ = objective.calculate(initial_solution)
_, initial_violations = validator.validate_solution(initial_solution)
initial_coverage = initial_solution.coverage_rate(len(problem.orders)) * 100

print(f"✓ Initial solution:")
print(f"  Cost: {initial_cost:.2f}")
print(f"  Coverage: {initial_coverage:.1f}%")
print(f"  Violations: {len(initial_violations)}")


# ============================================================================
# STEP 4: Setup RL-ALNS
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

print(f"✓ RL-ALNS ready")
print(f"  State dim: {rl_alns.rl_agent.state_dim}")
print(f"  Action space: {rl_alns.rl_agent.action_dim}")


# ============================================================================
# STEP 5: Extended training
# ============================================================================

print(f"\n[Step 5] Starting extended training...")
print(f"{'='*70}")
print(f"TRAINING CONFIGURATION")
print(f"{'='*70}")
print(f"Episodes: 20 (was 5)")
print(f"Steps per episode: 100 (was 50)")
print(f"Total iterations: 2000 (was 250)")
print(f"Expected duration: ~5-10 minutes")
print(f"Improved reward: Feasibility-focused")
print(f"{'='*70}")

# Train with improved parameters
best_solution, train_stats = rl_alns.train(
    initial_solution=initial_solution,
    num_episodes=20,              # 4x more episodes
    steps_per_episode=100,        # 2x more steps
    destroy_size_min=1,
    destroy_size_max=3,
    update_frequency=5            # Update target network every 5 episodes
)


# ============================================================================
# STEP 6: Results analysis
# ============================================================================

print(f"\n[Step 6] Analyzing results...")

final_cost, final_components = objective.calculate(best_solution)
_, final_violations = validator.validate_solution(best_solution)
final_coverage = best_solution.coverage_rate(len(problem.orders)) * 100

print(f"\n{'='*70}")
print(f"FINAL RESULTS")
print(f"{'='*70}")

print(f"\n{'Metric':<25} {'Initial':<15} {'Final':<15} {'Change':<15}")
print(f"{'-'*70}")

# Coverage
coverage_change = final_coverage - initial_coverage
coverage_status = "✓" if coverage_change >= 0 else "✗"
print(f"{'Coverage (%)':<25} {initial_coverage:>14.1f} {final_coverage:>14.1f} {coverage_change:>+13.1f} {coverage_status}")

# Cost
cost_change_pct = ((final_cost - initial_cost) / initial_cost * 100) if initial_cost > 0 else 0
cost_status = "✓" if cost_change_pct < 0 else "✗"
print(f"{'Cost':<25} {initial_cost:>14.2f} {final_cost:>14.2f} {cost_change_pct:>+13.1f}% {cost_status}")

# Violations
viol_change = len(final_violations) - len(initial_violations)
viol_status = "✓" if viol_change < 0 else ("=" if viol_change == 0 else "✗")
print(f"{'Violations':<25} {len(initial_violations):>14} {len(final_violations):>14} {viol_change:>+14} {viol_status}")

# Distance
dist_change = final_components['total_distance'] - train_stats['episode_costs'][0]
dist_status = "✓" if dist_change < 0 else "✗"
print(f"{'Unserved Orders':<25} {len(initial_solution.unserved_orders):>14} {len(best_solution.unserved_orders):>14} {len(best_solution.unserved_orders)-len(initial_solution.unserved_orders):>+14}")

print(f"\n{'-'*70}")


# ============================================================================
# STEP 7: Learning analysis
# ============================================================================

print(f"\n{'='*70}")
print(f"LEARNING ANALYSIS")
print(f"{'='*70}")

# Episode statistics
print(f"\nBest 5 Episodes (by cost):")
episode_costs = list(enumerate(train_stats['episode_costs'], 1))
episode_costs.sort(key=lambda x: x[1])
for rank, (ep, cost) in enumerate(episode_costs[:5], 1):
    coverage = train_stats['episode_coverage'][ep-1]
    violations = train_stats['episode_violations'][ep-1]
    reward = train_stats['episode_rewards'][ep-1]
    print(f"  {rank}. Episode {ep:2d}: Cost={cost:8.2f}, Coverage={coverage:5.1f}%, "
          f"Violations={violations:2d}, Reward={reward:7.2f}")

print(f"\nWorst 3 Episodes (by cost):")
for rank, (ep, cost) in enumerate(episode_costs[-3:], 1):
    coverage = train_stats['episode_coverage'][ep-1]
    violations = train_stats['episode_violations'][ep-1]
    reward = train_stats['episode_rewards'][ep-1]
    print(f"  {rank}. Episode {ep:2d}: Cost={cost:8.2f}, Coverage={coverage:5.1f}%, "
          f"Violations={violations:2d}, Reward={reward:7.2f}")

# Learning trends
import numpy as np
costs = np.array(train_stats['episode_costs'])
coverages = np.array(train_stats['episode_coverage'])
violations = np.array(train_stats['episode_violations'])

print(f"\nOverall Trends:")
print(f"  Cost: {costs[0]:.2f} → {costs[-1]:.2f} (trend: {(costs[-1]-costs[0])/costs[0]*100:+.2f}%)")
print(f"  Coverage: {coverages[0]:.1f}% → {coverages[-1]:.1f}% (trend: {coverages[-1]-coverages[0]:+.1f}%)")
print(f"  Violations: {violations[0]} → {violations[-1]} (trend: {int(violations[-1]-violations[0]):+d})")

# Check if learning occurred
if len(set(costs)) > 5:  # More than 5 unique cost values
    print(f"\n✓ Learning detected: Agent explored diverse solutions")
else:
    print(f"\n⚠️  Limited exploration: Consider more episodes or different reward")


# ============================================================================
# STEP 8: Visualize learning
# ============================================================================

print(f"\n[Step 8] Creating learning plots...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cost over episodes
    axes[0, 0].plot(train_stats['episode_costs'], 'b-o', label='Episode Cost')
    axes[0, 0].plot(train_stats['best_costs_per_episode'], 'r--', label='Best Cost')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Cost')
    axes[0, 0].set_title('Cost Progression')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Coverage over episodes
    axes[0, 1].plot(train_stats['episode_coverage'], 'g-o')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Coverage (%)')
    axes[0, 1].set_title('Coverage Progression')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 105])
    
    # Plot 3: Violations over episodes
    axes[1, 0].plot(train_stats['episode_violations'], 'm-o')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Violations')
    axes[1, 0].set_title('Constraint Violations')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Rewards over episodes
    axes[1, 1].plot(train_stats['episode_rewards'], 'c-o')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].set_title('Episode Rewards')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_training_progress.png', dpi=150, bbox_inches='tight')
    print("✓ Plots saved to: rl_training_progress.png")
except Exception as e:
    print(f"⚠️  Could not create plots: {e}")


# ============================================================================
# STEP 9: Save agent
# ============================================================================

print(f"\n[Step 9] Saving trained agent...")

try:
    rl_alns.save_agent('rl_agent_improved.pt')
    print("✓ Agent saved: rl_agent_improved.pt")
except Exception as e:
    print(f"⚠️  Could not save: {e}")


# ============================================================================
# STEP 10: Best solution details
# ============================================================================

print(f"\n{'='*70}")
print(f"BEST SOLUTION ROUTES")
print(f"{'='*70}")

for service in best_solution.services:
    if len(service.orders) > 0:
        print(f"\nService {service.service_id}: {service.vehicle.number} ({service.vehicle.vehicle_type.name})")
        print(f"  Orders: {len(service.orders)}")
        print(f"  Capacity: {sum(o.capacity_required for o in service.orders):.2f}/{service.vehicle.capacity}")
        print(f"  Duration: {service.duration_minutes():.1f}/{problem.params.worktime_standard} min")
        
        route = [str(service.start_node)]
        for task in service.tasks:
            if task.task_type == 'PICKUP':
                route.append(f"P{task.order.id}")
            elif task.task_type == 'DELIVERY':
                route.append(f"D{task.order.id}")
            elif task.task_type == 'BREAK':
                route.append("BRK")
        route.append(str(service.end_node))
        print(f"  Route: {' → '.join(route)}")

if best_solution.unserved_orders:
    print(f"\n⚠️  Unserved: {[o.id for o in best_solution.unserved_orders]}")


# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"PHASE 2a: IMPROVED TRAINING COMPLETE")
print(f"{'='*70}")

total_improvement = (initial_cost - final_cost) / initial_cost * 100

print(f"\nTotal Improvement: {total_improvement:+.2f}%")
print(f"Training iterations: 2000 (vs 250 initial)")
print(f"Final epsilon: {rl_alns.rl_agent.epsilon:.3f}")

if total_improvement > 5:
    print(f"\n✓ EXCELLENT: Significant improvement achieved!")
elif total_improvement > 1:
    print(f"\n✓ GOOD: Moderate improvement, consider more episodes")
elif total_improvement > 0:
    print(f"\n⚠️  MODEST: Small improvement, problem is challenging")
else:
    print(f"\n⚠️  LIMITED: Consider Phase 2b (full JAMPR) for better results")

print(f"\n{'='*70}")
print(f"Ready for Phase 2b: Full JAMPR Implementation")
print(f"{'='*70}\n")