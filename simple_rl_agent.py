"""
simple_rl_agent.py
Simplified RL agent using DQN for operator selection

Phase 2a: Lightweight RL (not full JAMPR yet)
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict
import pickle

# Try to import PyTorch, fall back to numpy if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using numpy-based Q-learning fallback.")


class StateEncoder:
    """Encode solution state into feature vector"""
    
    def __init__(self, problem: 'ProblemInstance'):
        self.problem = problem
    
    def encode(self, solution: 'Solution', validator, objective) -> np.ndarray:
        """
        Convert solution to feature vector
        Returns: numpy array of shape (feature_dim,)
        """
        features = []
        
        # Basic solution metrics (normalized)
        total_orders = len(self.problem.orders)
        features.append(solution.num_services() / 10.0)  # Normalized by max vehicles
        features.append(solution.num_vehicles_used() / 10.0)
        features.append(len(solution.unserved_orders) / total_orders)
        features.append(solution.coverage_rate(total_orders))
        
        # Distance and time metrics (normalized)
        max_distance = 1000.0  # Assumed max
        features.append(solution.total_distance() / max_distance)
        features.append(solution.total_driving_time() / 1000.0)
        features.append(solution.total_idle_time() / 500.0)
        
        # Constraint violations
        is_feasible, violations = validator.validate_solution(solution)
        features.append(len(violations) / 20.0)  # Normalized
        
        # Violation types breakdown
        violation_types = {
            'duration': 0, 'capacity': 0, 'time_window': 0, 
            'break': 0, 'precedence': 0, 'other': 0
        }
        for v in violations:
            if 'DURATION' in v.violation_type:
                violation_types['duration'] += 1
            elif 'CAPACITY' in v.violation_type:
                violation_types['capacity'] += 1
            elif 'TIME_WINDOW' in v.violation_type:
                violation_types['time_window'] += 1
            elif 'BREAK' in v.violation_type:
                violation_types['break'] += 1
            elif 'PRECEDENCE' in v.violation_type:
                violation_types['precedence'] += 1
            else:
                violation_types['other'] += 1
        
        for count in violation_types.values():
            features.append(count / 10.0)
        
        # Service utilization statistics
        if solution.services:
            capacities = [
                sum(o.capacity_required for o in s.orders) / s.vehicle.capacity 
                for s in solution.services if s.orders
            ]
            if capacities:
                features.append(np.mean(capacities))
                features.append(np.max(capacities))
                features.append(np.min(capacities))
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Duration utilization
            durations = [
                s.duration_minutes() / self.problem.params.worktime_standard
                for s in solution.services if s.duration_minutes() > 0
            ]
            if durations:
                features.append(np.mean(durations))
                features.append(np.max(durations))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)


class DQNNetwork(nn.Module):
    """Simple feedforward network for Q-learning"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class SimplifiedRLAgent:
    """
    Simplified RL agent using DQN for operator selection
    Learns which (destroy, repair) operator pairs work best
    """
    
    def __init__(self, problem: 'ProblemInstance',
                 destroy_operators: List,
                 repair_operators: List,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995):
        
        self.problem = problem
        self.destroy_ops = destroy_operators
        self.repair_ops = repair_operators
        
        # Action space: all (destroy, repair) pairs
        self.actions = [
            (d, r) for d in destroy_operators for r in repair_operators
        ]
        self.action_dim = len(self.actions)
        
        # State encoder
        self.encoder = StateEncoder(problem)
        
        # Determine state dimension
        dummy_solution = self._create_dummy_solution()
        from constraint_validator import ConstraintValidator, ObjectiveCalculator
        self.validator = ConstraintValidator(problem)
        self.objective = ObjectiveCalculator(problem)
        
        dummy_state = self.encoder.encode(dummy_solution, self.validator, self.objective)
        self.state_dim = len(dummy_state)
        
        # Q-learning parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        if TORCH_AVAILABLE:
            # PyTorch-based DQN
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
        else:
            # Numpy-based Q-table fallback
            self.q_table = np.zeros((100, self.action_dim))  # Discretized states
        
        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Statistics
        self.stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'q_values': []
        }
    
    def _create_dummy_solution(self):
        """Create dummy solution for state dimension calculation"""
        from cvrptw_parser import Solution
        return Solution()
    
    def select_operators(self, solution: 'Solution') -> Tuple:
        """
        Select (destroy_op, repair_op) using epsilon-greedy policy
        
        Returns: (destroy_operator, repair_operator)
        """
        # Encode current state
        state = self.encoder.encode(solution, self.validator, self.objective)
        
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action from Q-network
            action_idx = self._get_best_action(state)
        
        destroy_op, repair_op = self.actions[action_idx]
        return destroy_op, repair_op, action_idx, state
    
    def _get_best_action(self, state: np.ndarray) -> int:
        """Get action with highest Q-value"""
        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            # Numpy fallback: discretize state
            state_idx = self._discretize_state(state)
            return np.argmax(self.q_table[state_idx])
    
    def _discretize_state(self, state: np.ndarray) -> int:
        """Discretize continuous state for Q-table (fallback)"""
        # Simple hash-based discretization
        hash_val = hash(tuple(np.round(state, 2)))
        return abs(hash_val) % 100
    
    def store_experience(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """Update Q-network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        
        if TORCH_AVAILABLE:
            self._update_pytorch(batch)
        else:
            self._update_numpy(batch)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _update_pytorch(self, batch):
        """Update using PyTorch"""
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _update_numpy(self, batch):
        """Update using numpy Q-table (fallback)"""
        for state, action, reward, next_state, done in batch:
            state_idx = self._discretize_state(state)
            next_state_idx = self._discretize_state(next_state)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state_idx])
            
            self.q_table[state_idx, action] += 0.1 * (target - self.q_table[state_idx, action])
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        if TORCH_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """Save agent"""
        if TORCH_AVAILABLE:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'stats': self.stats
            }, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon,
                    'stats': self.stats
                }, f)
    
    def load(self, filepath: str):
        """Load agent"""
        if TORCH_AVAILABLE:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.stats = checkpoint['stats']
        else:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
                self.q_table = checkpoint['q_table']
                self.epsilon = checkpoint['epsilon']
                self.stats = checkpoint['stats']


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Simplified RL Agent for ALNS")
    print("=" * 60)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print("\nFeatures:")
    print("  - DQN-based operator selection")
    print("  - Experience replay")
    print("  - Epsilon-greedy exploration")
    print("  - 21-dimensional state representation")
    print("\nReady to integrate with ALNS!")