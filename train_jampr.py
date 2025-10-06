"""
train_jampr.py
Complete training script for JAMPR on CVRPTW

Integrates Modules 1, 2, and 3
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import copy  # Add this import

from cvrptw_parser import ProblemInstance
from test_phase0 import SimpleGreedyConstructor
from graph_state_encoder import GraphStateEncoder, GraphAttentionEncoder
from policy_network import JamprPolicy
from ppo_trainer import PPOTrainer, ExperienceBuffer, PPOLogger
from constraint_validator import ObjectiveCalculator  # Add this import

# Use existing ALNS operators
from pdp_nlns_operators import (
    RandomRemoval, WorstRemoval, ShawRemoval,
    GreedyInsertion, Regret2Insertion
)


class JAMPRTrainer:
    """
    Complete training pipeline for JAMPR
    """
    
    def __init__(self, problem: ProblemInstance, config: dict):
        self.problem = problem
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Initialize objective calculator
        self.objective_calc = ObjectiveCalculator(problem)
        
        # Initialize components
        self._init_models()
        self._init_operators()
        self._init_trainer()
        self._init_logger()
    
    def _init_models(self):
        """Initialize neural networks"""
        print("Initializing models...")
        
        # Graph encoder
        self.graph_encoder_module = GraphStateEncoder(
            self.problem, 
            embedding_dim=self.config.get('embedding_dim', 64)
        )
        
        # Graph attention network
        self.gat = GraphAttentionEncoder(
            node_feature_dim=10,
            edge_feature_dim=10,
            hidden_dim=self.config.get('hidden_dim', 64),
            num_heads=self.config.get('num_heads', 4),
            num_layers=self.config.get('num_layers', 3)
        )
        
        # Policy network
        self.policy = JamprPolicy(
            graph_encoder=self.gat,
            state_dim=self.config.get('state_dim', 128),
            num_destroy_ops=4,
            num_repair_ops=3,
            hidden_dim=self.config.get('policy_hidden_dim', 256)
        )
        
        total_params = sum(p.numel() for p in self.policy.parameters())
        print(f"✓ Models initialized: {total_params:,} parameters")
    
    def _init_operators(self):
        """Initialize ALNS operators using existing implementations"""
        # Destroy operators
        self.destroy_ops_objects = [
            RandomRemoval(self.problem),
            WorstRemoval(self.problem),
            ShawRemoval(self.problem),
        ]
        
        # Repair operators  
        self.repair_ops_objects = [
            GreedyInsertion(self.problem),
            Regret2Insertion(self.problem),
        ]
        
        # Wrapper functions for consistent interface
        self.destroy_ops = [
            self._wrap_destroy(self.destroy_ops_objects[0], "random"),
            self._wrap_destroy(self.destroy_ops_objects[1], "worst"),
            self._wrap_destroy(self.destroy_ops_objects[2], "shaw"),
            self._route_removal_wrapper
        ]
        
        self.repair_ops = [
            self._wrap_repair(self.repair_ops_objects[0]),
            self._wrap_repair(self.repair_ops_objects[1]),
            self._wrap_repair(self.repair_ops_objects[0]),  # Best = Greedy for now
        ]
        
        print(f"✓ Operators initialized: {len(self.destroy_ops)} destroy, {len(self.repair_ops)} repair")
    
    def _wrap_destroy(self, operator, name):
        """Wrap existing destroy operator for RL interface"""
        def wrapped(solution, problem, removal_rate=0.3):
            served_orders = [o for o in problem.orders if o not in solution.unserved_orders]
            if not served_orders:
                return solution, []
            
            num_to_remove = max(1, int(len(served_orders) * removal_rate))
            
            # Make a copy (your operators modify in place)
            solution_copy = copy.deepcopy(solution)
            
            # Call destroy - returns list of removed orders
            removed_orders = operator.destroy(solution_copy, num_to_remove)
            
            return solution_copy, removed_orders
        return wrapped
    
    def _route_removal_wrapper(self, solution, problem, removal_rate=0.3):
        """Remove complete route"""
        if not solution.services:
            return solution, []
        
        import random
        service = random.choice(solution.services)
        destroyed = copy.deepcopy(solution)
        removed_orders = service.orders.copy()
        
        # Remove the entire service
        destroyed.services = [s for s in destroyed.services if s.service_id != service.service_id]
        
        # Add orders to unserved (repair will handle them)
        for order in removed_orders:
            if order not in destroyed.unserved_orders:
                destroyed.unserved_orders.append(order)
        
        return destroyed, removed_orders
    
    def _wrap_repair(self, operator):
        """Wrap existing repair operator for RL interface"""
        def wrapped(partial_solution, removed_orders, problem):
            if not removed_orders:
                return partial_solution
            
            # Make a deep copy to avoid modifying original
            solution_copy = copy.deepcopy(partial_solution)
            
            # Your repair operators expect removed_orders to be passed explicitly
            # They will try to insert these orders into the solution
            repaired = operator.repair(solution_copy, removed_orders)
            
            return repaired
        return wrapped
    
    def _init_trainer(self):
        """Initialize PPO trainer"""
        self.trainer = PPOTrainer(
            policy=self.policy,
            learning_rate=self.config.get('learning_rate', 3e-4),
            gamma=self.config.get('gamma', 0.99),
            lambda_gae=self.config.get('lambda_gae', 0.95),
            clip_epsilon=self.config.get('clip_epsilon', 0.2),
            value_loss_coef=self.config.get('value_loss_coef', 0.5),
            entropy_coef=self.config.get('entropy_coef', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            device=self.device
        )
        
        print("✓ PPO trainer initialized")
    
    def _init_logger(self):
        """Initialize logger"""
        self.logger = PPOLogger(
            log_dir=self.config.get('log_dir', 'runs'),
            experiment_name=self.config.get('experiment_name', 'jampr_cvrptw')
        )
        
        print("✓ Logger initialized")
    
    def encode_solution(self, solution):
        """Encode solution as graph"""
        graph_data = self.graph_encoder_module.encode_solution(solution)
        
        return {
            'node_features': torch.FloatTensor(graph_data['node_features']),
            'edge_index': torch.LongTensor(graph_data['edge_index']),
            'edge_features': torch.FloatTensor(graph_data['edge_features']),
            'global_features': torch.FloatTensor(graph_data['global_features'])
        }
    
    def select_operators(self, solution, deterministic=False):
        """Select operators using policy"""
        state_data = self.encode_solution(solution)
        
        with torch.no_grad():
            action_output = self.policy.select_action(
                state_data['node_features'].to(self.device),
                state_data['edge_index'].to(self.device),
                state_data['edge_features'].to(self.device),
                state_data['global_features'].to(self.device),
                deterministic=deterministic
            )
        
        return {
            'destroy_idx': action_output['destroy_action'].item(),
            'repair_idx': action_output['repair_action'].item(),
            'log_prob': action_output['log_prob'].item(),
            'value': action_output['value'].item()
        }
    
    def run_episode(self, max_iterations=100, temperature=10.0):
        """
        Run one episode of RL-guided ALNS
        
        Returns:
            best_solution: Best solution found
            buffer: Experience buffer for training
            episode_metrics: Episode statistics
        """
        # Create initial solution
        constructor = SimpleGreedyConstructor(self.problem)
        current_solution = constructor.construct(start_depot=1)
        best_solution = copy.deepcopy(current_solution)
        
        # Calculate initial cost using objective calculator
        initial_cost, _ = self.objective_calc.calculate(current_solution)
        best_cost = initial_cost
        
        buffer = ExperienceBuffer()
        
        num_improvements = 0
        num_accepts = 0
        
        for iteration in range(max_iterations):
            # Encode current state
            state_data = self.encode_solution(current_solution)
            
            # Select operators
            action_info = self.select_operators(current_solution, deterministic=False)
            
            # Apply destroy operator
            destroyed_solution, removed_orders = self.destroy_ops[action_info['destroy_idx']](
                current_solution, self.problem, 
                removal_rate=self.config.get('removal_rate', 0.3)
            )
            
            # Apply repair operator
            new_solution = self.repair_ops[action_info['repair_idx']](
                destroyed_solution, removed_orders, self.problem
            )
            
            # Evaluate costs
            new_cost, _ = self.objective_calc.calculate(new_solution)
            current_cost, _ = self.objective_calc.calculate(current_solution)
            
            # Calculate coverage
            new_coverage = new_solution.coverage_rate(len(self.problem.orders))
            current_coverage = current_solution.coverage_rate(len(self.problem.orders))
            coverage_change = new_coverage - current_coverage
            
            # DEBUG: Check order counts
            new_served = len([o for s in new_solution.services for o in s.orders])
            new_unserved = len(new_solution.unserved_orders)
            if iteration % 10 == 0:
                print(f"  Iter {iteration}: Served={new_served}, Unserved={new_unserved}, Cost={new_cost:.2f}")
            
            # Acceptance criterion with coverage protection
            if coverage_change < -0.01:  # Lost coverage - REJECT
                accept = False
                reward = -100.0 * abs(coverage_change)
            elif new_cost < current_cost and coverage_change >= 0:  # Better and maintained coverage
                accept = True
                reward = (current_cost - new_cost) / initial_cost
            else:  # Worse cost, use simulated annealing
                delta = new_cost - current_cost
                accept_prob = np.exp(-delta / (temperature * initial_cost)) if temperature > 0 else 0
                accept = np.random.random() < accept_prob
                reward = 0.0 if not accept else -(delta / initial_cost)
            
            # Store experience
            buffer.push(
                state_data=state_data,
                action=(action_info['destroy_idx'], action_info['repair_idx']),
                log_prob=action_info['log_prob'],
                reward=reward,
                value=action_info['value'],
                done=(iteration == max_iterations - 1)
            )
            
            # Update solution
            if accept:
                current_solution = copy.deepcopy(new_solution)
                num_accepts += 1
                
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(new_solution)
                    best_cost = new_cost
                    num_improvements += 1
            
            # Decay temperature
            temperature *= self.config.get('temp_decay', 0.995)
        
        # Episode metrics
        episode_metrics = {
            'initial_cost': initial_cost,
            'final_cost': best_cost,
            'improvement': (initial_cost - best_cost) / initial_cost * 100,
            'num_improvements': num_improvements,
            'num_accepts': num_accepts,
            'acceptance_rate': num_accepts / max_iterations,
            'coverage': best_solution.coverage_rate(len(self.problem.orders)) * 100
        }
        
        return best_solution, buffer, episode_metrics
    
    def train(self, num_episodes=100):
        """
        Main training loop
        
        Args:
            num_episodes: Number of training episodes
        """
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING: {num_episodes} episodes")
        print(f"{'='*60}\n")
        
        best_overall_cost = float('inf')
        best_overall_solution = None
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # Run episode
            best_solution, buffer, episode_metrics = self.run_episode(
                max_iterations=self.config.get('episode_length', 100),
                temperature=self.config.get('initial_temperature', 10.0)
            )
            
            # Update policy
            print("Updating policy...")
            training_metrics = self.trainer.update(
                buffer,
                num_epochs=self.config.get('ppo_epochs', 4),
                batch_size=self.config.get('batch_size', None)
            )
            
            # Log metrics
            all_metrics = {**episode_metrics, **training_metrics}
            self.logger.log_episode(episode + 1, all_metrics)
            
            # Track best solution
            if episode_metrics['final_cost'] < best_overall_cost:
                best_overall_cost = episode_metrics['final_cost']
                best_overall_solution = best_solution
                
                # Save best model
                self.save_checkpoint(f"best_model.pt", episode + 1, best_overall_cost)
            
            # Periodic checkpoint
            if (episode + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f"checkpoint_ep{episode+1}.pt", episode + 1)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best cost achieved: {best_overall_cost:.2f}")
        
        self.logger.close()
        
        return best_overall_solution
    
    def save_checkpoint(self, filename: str, episode: int, cost: float = None):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'config': self.config
        }
        
        if cost is not None:
            checkpoint['best_cost'] = cost
        
        torch.save(checkpoint, checkpoint_dir / filename)
        print(f"✓ Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Checkpoint loaded: {filename}")
        print(f"  Episode: {checkpoint['episode']}")
        if 'best_cost' in checkpoint:
            print(f"  Best cost: {checkpoint['best_cost']:.2f}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train JAMPR on CVRPTW')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=100, help='Length of each episode')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load problem
    print("Loading problem...")
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
    print(f"✓ Problem loaded: {len(problem.orders)} orders")
    
    # Configuration
    config = {
        'device': args.device,
        'learning_rate': args.lr,
        'episode_length': args.episode_length,
        'embedding_dim': 64,
        'hidden_dim': 64,
        'state_dim': 128,
        'policy_hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 3,
        'gamma': 0.99,
        'lambda_gae': 0.95,
        'clip_epsilon': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'ppo_epochs': 4,
        'batch_size': None,  # Full batch
        'removal_rate': 0.3,
        'initial_temperature': 10.0,
        'temp_decay': 0.995,
        'save_interval': 10,
        'log_dir': 'runs',
        'checkpoint_dir': 'checkpoints',
        'experiment_name': 'jampr_cvrptw'
    }
    
    # Create trainer
    trainer = JAMPRTrainer(problem, config)
    
    # Train
    best_solution = trainer.train(num_episodes=args.episodes)
    
    # Print final solution summary
    if best_solution:
        print("\n" + "="*70)
        print("FINAL BEST SOLUTION")
        print("="*70)
        # Get the final cost from the best solution
        final_cost, _ = trainer.objective_calc.calculate(best_solution)
        print(f"Cost: {final_cost:.2f}")
        #print(f"Cost: {best_overall_cost:.2f}")
        print(f"Services: {best_solution.num_services()}")
        print(f"Vehicles used: {best_solution.num_vehicles_used()}")
        print(f"Coverage: {best_solution.coverage_rate(len(problem.orders))*100:.1f}%")
        print(f"Total distance: {best_solution.total_distance():.2f} km")
        print(f"Unserved orders: {len(best_solution.unserved_orders)}")
        print("="*70)


if __name__ == "__main__":
    main()