"""
rl_alns_integration.py
Integrates RL agent with ALNS framework

Phase 2a: Complete RL pipeline (simplified version)
"""

import copy
import random
from typing import List, Tuple, Dict

from cvrptw_parser import ProblemInstance, Solution
from constraint_validator import ConstraintValidator, ObjectiveCalculator
from simple_rl_agent import SimplifiedRLAgent


class RLBasedALNS:
    """ALNS with RL-based operator selection"""
    
    def __init__(self, problem: ProblemInstance,
                 destroy_operators: List,
                 repair_operators: List,
                 use_rl: bool = True):
        
        self.problem = problem
        self.destroy_ops = destroy_operators
        self.repair_ops = repair_operators
        
        # Validators
        self.validator = ConstraintValidator(problem)
        self.objective = ObjectiveCalculator(problem)
        
        # RL Agent
        self.use_rl = use_rl
        if use_rl:
            self.rl_agent = SimplifiedRLAgent(
                problem=problem,
                destroy_operators=destroy_operators,
                repair_operators=repair_operators
            )
        
        # Statistics
        self.episode_rewards = []
        self.episode_costs = []
    
    def train(self, initial_solution: Solution,
              num_episodes: int = 10,
              steps_per_episode: int = 100,
              destroy_size_min: int = 1,
              destroy_size_max: int = 3,
              update_frequency: int = 10) -> Tuple[Solution, Dict]:
        """
        Train RL agent while optimizing
        
        Args:
            initial_solution: Starting solution
            num_episodes: Number of training episodes
            steps_per_episode: ALNS iterations per episode
            destroy_size_min: Min orders to remove
            destroy_size_max: Max orders to remove
            update_frequency: Update target network every N episodes
            
        Returns:
            (best_solution, statistics)
        """
        
        print(f"\n{'='*70}")
        print(f"PHASE 2a: RL-BASED ALNS TRAINING")
        print(f"{'='*70}")
        print(f"Episodes: {num_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Using RL: {self.use_rl}")
        print(f"Total iterations: {num_episodes * steps_per_episode}")
        
        global_best_solution = copy.deepcopy(initial_solution)
        global_best_cost, _ = self.objective.calculate(global_best_solution)
        
        all_stats = {
            'episode_rewards': [],
            'episode_costs': [],
            'episode_coverage': [],
            'episode_violations': [],
            'best_costs_per_episode': []
        }
        
        for episode in range(num_episodes):
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*70}")
            
            # Run one episode
            episode_solution, episode_stats = self._run_episode(
                copy.deepcopy(initial_solution),
                steps_per_episode,
                destroy_size_min,
                destroy_size_max
            )
            
            # Update global best
            episode_cost, _ = self.objective.calculate(episode_solution)
            if episode_cost < global_best_cost:
                global_best_solution = copy.deepcopy(episode_solution)
                global_best_cost = episode_cost
                print(f"\n🎉 New global best! Cost: {global_best_cost:.2f}")
            
            # Record statistics
            all_stats['episode_rewards'].append(episode_stats['total_reward'])
            all_stats['episode_costs'].append(episode_cost)
            all_stats['episode_coverage'].append(
                episode_solution.coverage_rate(len(self.problem.orders)) * 100
            )
            _, violations = self.validator.validate_solution(episode_solution)
            all_stats['episode_violations'].append(len(violations))
            all_stats['best_costs_per_episode'].append(global_best_cost)
            
            # Update target network periodically
            if self.use_rl and (episode + 1) % update_frequency == 0:
                self.rl_agent.update_target_network()
                print(f"\n🔄 Target network updated")
            
            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Cost: {episode_cost:.2f}")
            print(f"  Coverage: {all_stats['episode_coverage'][-1]:.1f}%")
            print(f"  Violations: {all_stats['episode_violations'][-1]}")
            print(f"  Total Reward: {episode_stats['total_reward']:.2f}")
            print(f"  Epsilon: {self.rl_agent.epsilon:.3f}" if self.use_rl else "")
        
        # Final report
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"\nGlobal Best Solution:")
        print(f"  Cost: {global_best_cost:.2f}")
        print(f"  Coverage: {global_best_solution.coverage_rate(len(self.problem.orders))*100:.1f}%")
        _, violations = self.validator.validate_solution(global_best_solution)
        print(f"  Violations: {len(violations)}")
        
        initial_cost, _ = self.objective.calculate(initial_solution)
        improvement = (initial_cost - global_best_cost) / initial_cost * 100
        print(f"\nImprovement:")
        print(f"  Initial: {initial_cost:.2f}")
        print(f"  Final: {global_best_cost:.2f}")
        print(f"  Improvement: {improvement:.2f}%")
        
        return global_best_solution, all_stats
    
    def _run_episode(self, solution: Solution,
                    steps: int,
                    destroy_size_min: int,
                    destroy_size_max: int) -> Tuple[Solution, Dict]:
        """Run one training episode"""
        
        current_solution = solution
        best_solution = copy.deepcopy(solution)
        
        current_cost, _ = self.objective.calculate(current_solution)
        best_cost = current_cost
        
        episode_stats = {
            'total_reward': 0.0,
            'improvements': 0,
            'accepted': 0,
            'rejected': 0
        }
        
        for step in range(steps):
            # Get current state
            if self.use_rl:
                destroy_op, repair_op, action_idx, state = self.rl_agent.select_operators(current_solution)
            else:
                # Random selection (baseline)
                destroy_op = random.choice(self.destroy_ops)
                repair_op = random.choice(self.repair_ops)
                state = None
                action_idx = None
            
            # Determine destroy size
            k = random.randint(destroy_size_min, 
                             min(destroy_size_max, len(current_solution.services) * 2))
            
            # Apply destroy-repair
            try:
                new_solution = self._destroy_repair(
                    copy.deepcopy(current_solution),
                    destroy_op,
                    repair_op,
                    k
                )
                
                new_cost, _ = self.objective.calculate(new_solution)
                
                # Calculate reward
                reward = self._calculate_reward(current_cost, new_cost, best_cost)
                episode_stats['total_reward'] += reward
                
                # Accept/reject (simple: always accept better)
                if new_cost < current_cost:
                    current_solution = new_solution
                    current_cost = new_cost
                    episode_stats['accepted'] += 1
                    
                    if new_cost < best_cost:
                        best_solution = copy.deepcopy(new_solution)
                        best_cost = new_cost
                        episode_stats['improvements'] += 1
                else:
                    episode_stats['rejected'] += 1
                
                # Store experience and update RL agent
                if self.use_rl and state is not None:
                    next_state = self.rl_agent.encoder.encode(
                        new_solution, self.validator, self.objective
                    )
                    done = (step == steps - 1)
                    self.rl_agent.store_experience(state, action_idx, reward, next_state, done)
                    self.rl_agent.update()
                
            except Exception as e:
                # If operation fails, skip
                episode_stats['rejected'] += 1
                continue
            
            # Progress reporting
            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{steps} | Best: {best_cost:.2f} | "
                      f"Current: {current_cost:.2f} | "
                      f"Reward: {episode_stats['total_reward']:.2f}")
        
        return best_solution, episode_stats
    
    def _destroy_repair(self, solution: Solution,
                       destroy_op, repair_op, k: int) -> Solution:
        """Apply destroy and repair operators"""
        removed_orders = destroy_op.destroy(solution, k)
        solution = repair_op.repair(solution, removed_orders)
        return solution
    
    def _calculate_reward(self, current_cost: float, 
                         new_cost: float, best_cost: float,
                         current_solution: 'Solution' = None,
                         new_solution: 'Solution' = None) -> float:
        """
        Calculate reward for RL agent with feasibility focus
        
        Reward structure:
        - Prioritize feasibility improvements
        - Then optimize cost
        """
        reward = 0.0
        
        # Check feasibility improvements
        if current_solution and new_solution:
            _, curr_violations = self.validator.validate_solution(current_solution)
            _, new_violations = self.validator.validate_solution(new_solution)
            
            # Big reward for reducing violations
            violation_delta = len(curr_violations) - len(new_violations)
            if violation_delta > 0:
                reward += 50.0 * violation_delta  # +50 per violation removed
            elif violation_delta < 0:
                reward -= 30.0 * abs(violation_delta)  # -30 per violation added
            
            # Reward coverage improvements
            curr_coverage = current_solution.coverage_rate(len(self.problem.orders))
            new_coverage = new_solution.coverage_rate(len(self.problem.orders))
            coverage_delta = new_coverage - curr_coverage
            if coverage_delta > 0:
                reward += 100.0 * coverage_delta  # Big reward for serving more orders
            elif coverage_delta < 0:
                reward -= 50.0 * abs(coverage_delta)  # Penalty for losing coverage
        
        # Cost-based rewards
        if new_cost < best_cost:
            # New global best!
            improvement = (current_cost - new_cost) / current_cost if current_cost > 0 else 0
            reward += 100.0 + improvement * 100.0
        elif new_cost < current_cost:
            # Better than current
            improvement = (current_cost - new_cost) / current_cost if current_cost > 0 else 0
            reward += 10.0 + improvement * 50.0
        elif new_cost < current_cost * 1.05:
            # Slightly worse (within 5%) - small positive to encourage exploration
            reward += 1.0
        else:
            # Much worse - but not too harsh to allow exploration
            reward -= 5.0
        
        return reward
    
    def save_agent(self, filepath: str):
        """Save trained RL agent"""
        if self.use_rl:
            self.rl_agent.save(filepath)
            print(f"RL agent saved to {filepath}")
    
    def load_agent(self, filepath: str):
        """Load trained RL agent"""
        if self.use_rl:
            self.rl_agent.load(filepath)
            print(f"RL agent loaded from {filepath}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("RL-based ALNS Integration Ready!")
    print("\nTo use:")
    print("  1. Create RLBasedALNS instance")
    print("  2. Call train() method")
    print("  3. Get optimized solution with learned operator selection")