"""
policy_network.py
Attention-based Policy Network for JAMPR

Phase 2b - Module 2: Actor-Critic architecture for operator selection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch required for Module 2")


class OperatorEncoder(nn.Module):
    """
    Encodes ALNS operators as embeddings
    
    Destroy Operators:
    - Random removal (id=0)
    - Worst removal (id=1) 
    - Shaw removal (id=2)
    - Route removal (id=3)
    
    Repair Operators:
    - Greedy insertion (id=0)
    - Regret-2 insertion (id=1)
    - Best insertion (id=2)
    """
    
    def __init__(self, num_destroy_ops: int = 4, 
                 num_repair_ops: int = 3,
                 embedding_dim: int = 64):
        super().__init__()
        
        self.num_destroy_ops = num_destroy_ops
        self.num_repair_ops = num_repair_ops
        self.num_total_ops = num_destroy_ops + num_repair_ops
        
        # Operator embeddings (learned)
        self.operator_embeddings = nn.Embedding(self.num_total_ops, embedding_dim)
        
        # Operator type embeddings
        self.type_embeddings = nn.Embedding(2, embedding_dim)  # destroy/repair
        
    def forward(self, operator_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode operators
        
        Args:
            operator_ids: [batch_size] or [batch_size, 2] for (destroy, repair)
        
        Returns:
            embeddings: [batch_size, embedding_dim] or [batch_size, 2, embedding_dim]
        """
        return self.operator_embeddings(operator_ids)
    
    def get_destroy_repair_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all destroy and repair operator embeddings"""
        destroy_ids = torch.arange(self.num_destroy_ops)
        repair_ids = torch.arange(self.num_repair_ops) + self.num_destroy_ops
        
        destroy_emb = self.operator_embeddings(destroy_ids)
        repair_emb = self.operator_embeddings(repair_ids)
        
        return destroy_emb, repair_emb


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for global context
    
    Computes weighted average of node embeddings using attention
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        self.scale = np.sqrt(output_dim)
    
    def forward(self, node_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool node embeddings to global context
        
        Args:
            node_embeddings: [batch_size, num_nodes, input_dim]
            mask: [batch_size, num_nodes] - True for valid nodes
        
        Returns:
            global_context: [batch_size, output_dim]
        """
        batch_size, num_nodes, _ = node_embeddings.shape
        
        # Compute attention scores
        Q = self.query(node_embeddings.mean(dim=1, keepdim=True))  # [B, 1, D]
        K = self.key(node_embeddings)  # [B, N, D]
        V = self.value(node_embeddings)  # [B, N, D]
        
        # Attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, 1, N]
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        weights = F.softmax(scores, dim=-1)  # [B, 1, N]
        
        # Weighted sum
        global_context = torch.matmul(weights, V).squeeze(1)  # [B, D]
        
        return global_context


class StateEmbedding(nn.Module):
    """
    Processes graph encoding into state representation
    
    Takes output from GraphAttentionEncoder (Module 1) and produces
    a global state embedding for policy/value networks
    """
    
    def __init__(self, node_dim: int = 64, 
                 global_dim: int = 128,
                 num_global_features: int = 8):
        super().__init__()
        
        self.node_dim = node_dim
        self.global_dim = global_dim
        
        # Attention pooling for node-level info
        self.attention_pooling = AttentionPooling(node_dim, global_dim)
        
        # Process global features
        self.global_processor = nn.Sequential(
            nn.Linear(num_global_features, global_dim),
            nn.ReLU(),
            nn.LayerNorm(global_dim)
        )
        
        # Combine pooled nodes + global features
        self.combiner = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim),
            nn.ReLU(),
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, global_dim)
        )
    
    def forward(self, node_embeddings: torch.Tensor, 
                global_features: torch.Tensor,
                node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Produce state embedding
        
        Args:
            node_embeddings: [batch_size, num_nodes, node_dim]
            global_features: [batch_size, num_global_features]
            node_mask: [batch_size, num_nodes] - True for valid nodes
        
        Returns:
            state_embedding: [batch_size, global_dim]
        """
        # Pool node information
        pooled_nodes = self.attention_pooling(node_embeddings, node_mask)
        
        # Process global features
        processed_global = self.global_processor(global_features)
        
        # Combine
        combined = torch.cat([pooled_nodes, processed_global], dim=-1)
        state_embedding = self.combiner(combined)
        
        return state_embedding


class ActorNetwork(nn.Module):
    """
    Actor network for operator selection (Policy π_θ)
    
    Outputs probability distribution over (destroy_op, repair_op) pairs
    """
    
    def __init__(self, state_dim: int = 128,
                 num_destroy_ops: int = 4,
                 num_repair_ops: int = 3,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_destroy_ops = num_destroy_ops
        self.num_repair_ops = num_repair_ops
        
        # Separate heads for destroy and repair operators
        self.destroy_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_destroy_ops)
        )
        
        self.repair_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_repair_ops)
        )
    
    def forward(self, state_embedding: torch.Tensor,
                destroy_mask: Optional[torch.Tensor] = None,
                repair_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action logits
        
        Args:
            state_embedding: [batch_size, state_dim]
            destroy_mask: [batch_size, num_destroy_ops] - True for valid ops
            repair_mask: [batch_size, num_repair_ops] - True for valid ops
        
        Returns:
            destroy_logits: [batch_size, num_destroy_ops]
            repair_logits: [batch_size, num_repair_ops]
        """
        destroy_logits = self.destroy_head(state_embedding)
        repair_logits = self.repair_head(state_embedding)
        
        # Apply masks (set invalid actions to -inf)
        if destroy_mask is not None:
            destroy_logits = destroy_logits.masked_fill(~destroy_mask, float('-inf'))
        if repair_mask is not None:
            repair_logits = repair_logits.masked_fill(~repair_mask, float('-inf'))
        
        return destroy_logits, repair_logits
    
    def get_action_probs(self, state_embedding: torch.Tensor,
                        destroy_mask: Optional[torch.Tensor] = None,
                        repair_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get probability distributions"""
        destroy_logits, repair_logits = self.forward(state_embedding, destroy_mask, repair_mask)
        
        destroy_probs = F.softmax(destroy_logits, dim=-1)
        repair_probs = F.softmax(repair_logits, dim=-1)
        
        return destroy_probs, repair_probs
    
    def sample_action(self, state_embedding: torch.Tensor,
                     destroy_mask: Optional[torch.Tensor] = None,
                     repair_mask: Optional[torch.Tensor] = None,
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Returns:
            destroy_action: [batch_size]
            repair_action: [batch_size]
            log_prob: [batch_size] - log π(a|s)
        """
        destroy_logits, repair_logits = self.forward(state_embedding, destroy_mask, repair_mask)
        
        if deterministic:
            # Greedy selection
            destroy_action = destroy_logits.argmax(dim=-1)
            repair_action = repair_logits.argmax(dim=-1)
            
            # Get log probs for the selected actions
            destroy_probs = F.softmax(destroy_logits, dim=-1)
            repair_probs = F.softmax(repair_logits, dim=-1)
            
            destroy_log_prob = torch.log(destroy_probs.gather(1, destroy_action.unsqueeze(1)) + 1e-8)
            repair_log_prob = torch.log(repair_probs.gather(1, repair_action.unsqueeze(1)) + 1e-8)
        else:
            # Stochastic sampling
            destroy_dist = Categorical(logits=destroy_logits)
            repair_dist = Categorical(logits=repair_logits)
            
            destroy_action = destroy_dist.sample()
            repair_action = repair_dist.sample()
            
            destroy_log_prob = destroy_dist.log_prob(destroy_action)
            repair_log_prob = repair_dist.log_prob(repair_action)
        
        log_prob = destroy_log_prob + repair_log_prob
        
        return destroy_action, repair_action, log_prob


class CriticNetwork(nn.Module):
    """
    Critic network for value estimation (Value function V_φ)
    
    Estimates expected return V(s) from current state
    """
    
    def __init__(self, state_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value
        
        Args:
            state_embedding: [batch_size, state_dim]
        
        Returns:
            value: [batch_size, 1]
        """
        return self.value_head(state_embedding)


class JamprPolicy(nn.Module):
    """
    Complete JAMPR Policy Network (Actor-Critic)
    
    Combines:
    - Graph encoder (Module 1)
    - State embedding
    - Actor (operator selection)
    - Critic (value estimation)
    """
    
    def __init__(self, 
                 graph_encoder: nn.Module,
                 state_dim: int = 128,
                 num_destroy_ops: int = 4,
                 num_repair_ops: int = 3,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.graph_encoder = graph_encoder
        self.state_embedding = StateEmbedding(
            node_dim=graph_encoder.hidden_dim,
            global_dim=state_dim
        )
        
        self.actor = ActorNetwork(state_dim, num_destroy_ops, num_repair_ops, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)
        
        self.operator_encoder = OperatorEncoder(num_destroy_ops, num_repair_ops, state_dim)
    
    def forward(self, node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: torch.Tensor,
                global_features: torch.Tensor,
                node_mask: Optional[torch.Tensor] = None,
                destroy_mask: Optional[torch.Tensor] = None,
                repair_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass
        
        Returns dict with:
        - 'state_embedding': State representation
        - 'destroy_logits': Destroy operator logits
        - 'repair_logits': Repair operator logits
        - 'value': State value estimate
        """
        # Encode graph (Module 1)
        node_embeddings = self.graph_encoder(node_features, edge_index, edge_features)
        
        # Add batch dimension if needed
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
            global_features = global_features.unsqueeze(0)
        
        # Create state embedding
        state_emb = self.state_embedding(node_embeddings, global_features, node_mask)
        
        # Get action logits
        destroy_logits, repair_logits = self.actor(state_emb, destroy_mask, repair_mask)
        
        # Get value estimate
        value = self.critic(state_emb)
        
        return {
            'state_embedding': state_emb,
            'destroy_logits': destroy_logits,
            'repair_logits': repair_logits,
            'value': value
        }
    
    def select_action(self, node_features: torch.Tensor,
                     edge_index: torch.Tensor,
                     edge_features: torch.Tensor,
                     global_features: torch.Tensor,
                     deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Select action using current policy
        
        Returns:
        - 'destroy_action': Selected destroy operator
        - 'repair_action': Selected repair operator
        - 'log_prob': Log probability of action
        - 'value': Value estimate
        """
        output = self.forward(node_features, edge_index, edge_features, global_features)
        
        destroy_action, repair_action, log_prob = self.actor.sample_action(
            output['state_embedding'],
            deterministic=deterministic
        )
        
        return {
            'destroy_action': destroy_action,
            'repair_action': repair_action,
            'log_prob': log_prob,
            'value': output['value']
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("JAMPR Policy Network - Module 2")
    print("="*60)
    print("\nComponents:")
    print("  ✓ OperatorEncoder")
    print("  ✓ AttentionPooling")
    print("  ✓ StateEmbedding")
    print("  ✓ ActorNetwork (Policy)")
    print("  ✓ CriticNetwork (Value)")
    print("  ✓ JamprPolicy (Complete)")
    print("\nReady for integration with Module 1!")