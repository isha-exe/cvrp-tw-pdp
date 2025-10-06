"""
ppo_trainer.py
PPO (Proximal Policy Optimization) Training Module

Phase 2b - Module 3: Training infrastructure for JAMPR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available for logging")


class ExperienceBuffer:
    """
    Buffer for storing and processing rollout experiences
    """
    
    def __init__(self):
        self.states = []
        self.actions = []  # (destroy_action, repair_action)
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # For graph data
        self.node_features = []
        self.edge_indices = []
        self.edge_features = []
        self.global_features = []
    
    def push(self, state_data: Dict, action: Tuple[int, int], 
             log_prob: float, reward: float, value: float, done: bool):
        """Add experience to buffer"""
        self.node_features.append(state_data['node_features'])
        self.edge_indices.append(state_data['edge_index'])
        self.edge_features.append(state_data['edge_features'])
        self.global_features.append(state_data['global_features'])
        
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get_batch(self):
        """Get all experiences as batch"""
        return {
            'node_features': self.node_features,
            'edge_indices': self.edge_indices,
            'edge_features': self.edge_features,
            'global_features': self.global_features,
            'actions': self.actions,
            'log_probs': torch.FloatTensor(self.log_probs),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'dones': torch.FloatTensor(self.dones)
        }
    
    def clear(self):
        """Clear buffer"""
        self.node_features = []
        self.edge_indices = []
        self.edge_features = []
        self.global_features = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def __len__(self):
        return len(self.rewards)


class GAE:
    """
    Generalized Advantage Estimation
    
    Computes advantages using TD(λ) returns
    """
    
    @staticmethod
    def compute_gae(rewards: torch.Tensor, 
                    values: torch.Tensor,
                    dones: torch.Tensor,
                    next_value: float,
                    gamma: float = 0.99,
                    lambda_: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns
        
        Args:
            rewards: [T] - rewards at each timestep
            values: [T] - value estimates at each timestep
            dones: [T] - done flags
            next_value: Value estimate for next state after trajectory
            gamma: Discount factor
            lambda_: GAE lambda parameter
        
        Returns:
            advantages: [T] - advantage estimates
            returns: [T] - discounted returns
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value_t = next_value
        
        # Compute advantages backwards
        for t in reversed(range(T)):
            if t == T - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value_t = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # TD error
            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
            
            # GAE
            gae = delta + gamma * lambda_ * next_non_terminal * gae
            advantages[t] = gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns


class PPOTrainer:
    """
    PPO Trainer for JAMPR Policy
    
    Implements PPO with clipped objective and entropy regularization
    """
    
    def __init__(self, policy: nn.Module,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        
        self.policy = policy.to(device)
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=learning_rate
        )
        
        # Metrics
        self.train_metrics = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'approx_kl': deque(maxlen=100),
            'clip_fraction': deque(maxlen=100)
        }
    
    def compute_advantages(self, buffer: ExperienceBuffer, 
                          next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using GAE"""
        batch = buffer.get_batch()
        
        advantages, returns = GAE.compute_gae(
            batch['rewards'],
            batch['values'],
            batch['dones'],
            next_value,
            self.gamma,
            self.lambda_gae
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_policy_loss(self, 
                           new_log_probs: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO clipped policy loss
        
        Args:
            new_log_probs: [batch_size] - log probs from current policy
            old_log_probs: [batch_size] - log probs from old policy
            advantages: [batch_size] - advantage estimates
        
        Returns:
            loss: PPO clipped loss
            info: Dict with metrics
        """
        # Probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL divergence (approximation)
        approx_kl = (old_log_probs - new_log_probs).mean().item()
        
        # Clip fraction (for monitoring)
        clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
        
        info = {
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }
        
        return policy_loss, info
    
    def compute_value_loss(self, 
                          new_values: torch.Tensor,
                          returns: torch.Tensor) -> torch.Tensor:
        """Compute value function loss (MSE)"""
        # Squeeze to match dimensions
        return F.mse_loss(new_values.squeeze(), returns)
    
    def compute_entropy(self,
                       destroy_logits: torch.Tensor,
                       repair_logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of policy distribution"""
        destroy_probs = F.softmax(destroy_logits, dim=-1)
        repair_probs = F.softmax(repair_logits, dim=-1)
        
        destroy_entropy = -(destroy_probs * torch.log(destroy_probs + 1e-8)).sum(dim=-1).mean()
        repair_entropy = -(repair_probs * torch.log(repair_probs + 1e-8)).sum(dim=-1).mean()
        
        return destroy_entropy + repair_entropy
    
    def update(self, buffer: ExperienceBuffer, 
               num_epochs: int = 4,
               batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Update policy using PPO
        
        Args:
            buffer: Experience buffer with rollout data
            num_epochs: Number of epochs to train on the data
            batch_size: Mini-batch size (None = full batch)
        
        Returns:
            metrics: Dict with training metrics
        """
        if len(buffer) == 0:
            return {}
        
        # Compute advantages
        advantages, returns = self.compute_advantages(buffer)
        
        # Get batch data
        batch_data = buffer.get_batch()
        old_log_probs = batch_data['log_probs'].to(self.device)
        
        # Train for multiple epochs
        epoch_metrics = []
        
        for epoch in range(num_epochs):
            # Use full batch if batch_size not specified
            indices = list(range(len(buffer)))
            
            if batch_size is not None and batch_size < len(buffer):
                # Mini-batch training
                np.random.shuffle(indices)
                num_batches = len(indices) // batch_size
            else:
                # Full batch
                num_batches = 1
                batch_size = len(buffer)
            
            for batch_idx in range(num_batches):
                # Get mini-batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(buffer))
                mb_indices = indices[start_idx:end_idx]
                
                # Forward pass through policy
                new_log_probs_list = []
                new_values_list = []
                destroy_logits_list = []
                repair_logits_list = []
                
                for idx in mb_indices:
                    node_feat = batch_data['node_features'][idx].to(self.device)
                    edge_idx = batch_data['edge_indices'][idx].to(self.device)
                    edge_feat = batch_data['edge_features'][idx].to(self.device)
                    global_feat = batch_data['global_features'][idx].to(self.device)
                    
                    # Forward pass
                    output = self.policy(node_feat, edge_idx, edge_feat, global_feat)
                    
                    # Get log prob for taken action
                    destroy_action, repair_action = batch_data['actions'][idx]
                    
                    destroy_log_prob = F.log_softmax(output['destroy_logits'], dim=-1)[0, destroy_action]
                    repair_log_prob = F.log_softmax(output['repair_logits'], dim=-1)[0, repair_action]
                    new_log_prob = destroy_log_prob + repair_log_prob
                    
                    new_log_probs_list.append(new_log_prob)
                    new_values_list.append(output['value'][0])
                    destroy_logits_list.append(output['destroy_logits'])
                    repair_logits_list.append(output['repair_logits'])
                
                new_log_probs = torch.stack(new_log_probs_list)
                new_values = torch.stack(new_values_list)
                destroy_logits = torch.cat(destroy_logits_list, dim=0)
                repair_logits = torch.cat(repair_logits_list, dim=0)
                
                # Get corresponding advantages and returns
                mb_advantages = advantages[mb_indices].to(self.device)
                mb_returns = returns[mb_indices].to(self.device)
                mb_old_log_probs = old_log_probs[mb_indices]
                
                # Compute losses
                policy_loss, policy_info = self.compute_policy_loss(
                    new_log_probs, mb_old_log_probs, mb_advantages
                )
                
                value_loss = self.compute_value_loss(new_values, mb_returns)
                
                entropy = self.compute_entropy(destroy_logits, repair_logits)
                
                # Total loss
                total_loss = (policy_loss + 
                            self.value_loss_coef * value_loss - 
                            self.entropy_coef * entropy)
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Record metrics
                epoch_metrics.append({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item(),
                    'total_loss': total_loss.item(),
                    'approx_kl': policy_info['approx_kl'],
                    'clip_fraction': policy_info['clip_fraction']
                })
        
        # Aggregate metrics
        avg_metrics = {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
        
        # Update running metrics
        for key, value in avg_metrics.items():
            self.train_metrics[key].append(value)
        
        return avg_metrics
    
    def get_metrics(self) -> Dict[str, float]:
        """Get recent training metrics"""
        return {
            key: np.mean(values) if len(values) > 0 else 0.0
            for key, values in self.train_metrics.items()
        }


class PPOLogger:
    """
    Logger for PPO training with tensorboard support
    """
    
    def __init__(self, log_dir: str = 'runs', experiment_name: str = 'jampr_ppo'):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=f'{log_dir}/{experiment_name}_{int(time.time())}')
        else:
            self.writer = None
        
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_coverages = []
    
    def log_episode(self, episode: int, metrics: Dict):
        """Log episode metrics"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Episode/{key}', value, episode)
        
        # Print to console
        print(f"Episode {episode}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    def log_training(self, step: int, metrics: Dict):
        """Log training metrics"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Training/{key}', value, step)
    
    def close(self):
        """Close logger"""
        if self.writer:
            self.writer.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("PPO Trainer - Module 3")
    print("="*60)
    print("\nComponents:")
    print("  ✓ ExperienceBuffer")
    print("  ✓ GAE (Generalized Advantage Estimation)")
    print("  ✓ PPOTrainer")
    print("  ✓ PPOLogger")
    print("\nFeatures:")
    print("  ✓ Clipped surrogate objective")
    print("  ✓ Value function learning")
    print("  ✓ Entropy regularization")
    print("  ✓ Gradient clipping")
    print("  ✓ Mini-batch training")
    print("  ✓ Tensorboard logging")
    print("\nReady for training!")