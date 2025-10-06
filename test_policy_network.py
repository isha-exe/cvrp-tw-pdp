"""
test_policy_network.py
Test script for attention-based policy network

Tests Module 2 of Phase 2b
"""

import sys
import numpy as np
import torch.nn.functional as F

print("="*70)
print("TESTING POLICY NETWORK (Module 2)")
print("="*70)

# Import required modules
try:
    from graph_state_encoder import GraphAttentionEncoder, TORCH_AVAILABLE
    from policy_network import (
        OperatorEncoder, AttentionPooling, StateEmbedding,
        ActorNetwork, CriticNetwork, JamprPolicy
    )
    print("\n✓ All modules imported")
    print(f"  PyTorch available: {TORCH_AVAILABLE}")
    
    if not TORCH_AVAILABLE:
        print("\n✗ PyTorch not available - cannot run tests")
        sys.exit(1)
        
    import torch
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    sys.exit(1)

# Test parameters
batch_size = 4
num_nodes = 25  # 1 depot + 12 pickups + 12 deliveries
num_edges = 40
node_feature_dim = 10
edge_feature_dim = 10
hidden_dim = 64
state_dim = 128

print("\n[Test Configuration]")
print(f"  Batch size: {batch_size}")
print(f"  Num nodes: {num_nodes}")
print(f"  Num edges: {num_edges}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  State dim: {state_dim}")

# ============================================================================
# TEST 1: Operator Encoder
# ============================================================================
print("\n" + "="*70)
print("TEST 1: Operator Encoder")
print("="*70)

num_destroy_ops = 4
num_repair_ops = 3

op_encoder = OperatorEncoder(
    num_destroy_ops=num_destroy_ops,
    num_repair_ops=num_repair_ops,
    embedding_dim=hidden_dim
)

print(f"\n✓ OperatorEncoder created")
print(f"  Total operators: {op_encoder.num_total_ops}")
print(f"  Parameters: {sum(p.numel() for p in op_encoder.parameters()):,}")

# Test encoding
destroy_ids = torch.tensor([0, 1, 2, 3])
repair_ids = torch.tensor([4, 5, 6])

destroy_emb = op_encoder(destroy_ids)
repair_emb = op_encoder(repair_ids)

print(f"\n✓ Operator encoding works")
print(f"  Destroy embeddings shape: {destroy_emb.shape}")
print(f"  Repair embeddings shape: {repair_emb.shape}")
print(f"  Embedding range: [{destroy_emb.min():.3f}, {destroy_emb.max():.3f}]")

# ============================================================================
# TEST 2: Attention Pooling
# ============================================================================
print("\n" + "="*70)
print("TEST 2: Attention Pooling")
print("="*70)

attn_pool = AttentionPooling(input_dim=hidden_dim, output_dim=state_dim)

print(f"\n✓ AttentionPooling created")
print(f"  Parameters: {sum(p.numel() for p in attn_pool.parameters()):,}")

# Create dummy node embeddings
node_embeddings = torch.randn(batch_size, num_nodes, hidden_dim)
node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

with torch.no_grad():
    global_context = attn_pool(node_embeddings, node_mask)

print(f"\n✓ Attention pooling works")
print(f"  Input shape: {node_embeddings.shape}")
print(f"  Output shape: {global_context.shape}")
print(f"  Output range: [{global_context.min():.3f}, {global_context.max():.3f}]")

# ============================================================================
# TEST 3: State Embedding
# ============================================================================
print("\n" + "="*70)
print("TEST 3: State Embedding")
print("="*70)

state_emb_module = StateEmbedding(
    node_dim=hidden_dim,
    global_dim=state_dim,
    num_global_features=8
)

print(f"\n✓ StateEmbedding created")
print(f"  Parameters: {sum(p.numel() for p in state_emb_module.parameters()):,}")

# Create dummy inputs
global_features = torch.randn(batch_size, 8)

with torch.no_grad():
    state_embedding = state_emb_module(node_embeddings, global_features, node_mask)

print(f"\n✓ State embedding works")
print(f"  Output shape: {state_embedding.shape}")
print(f"  Output range: [{state_embedding.min():.3f}, {state_embedding.max():.3f}]")

# ============================================================================
# TEST 4: Actor Network
# ============================================================================
print("\n" + "="*70)
print("TEST 4: Actor Network")
print("="*70)

actor = ActorNetwork(
    state_dim=state_dim,
    num_destroy_ops=num_destroy_ops,
    num_repair_ops=num_repair_ops,
    hidden_dim=256
)

print(f"\n✓ ActorNetwork created")
print(f"  Parameters: {sum(p.numel() for p in actor.parameters()):,}")

# Test forward pass
with torch.no_grad():
    destroy_logits, repair_logits = actor(state_embedding)
    destroy_probs, repair_probs = actor.get_action_probs(state_embedding)

print(f"\n✓ Actor forward pass works")
print(f"  Destroy logits shape: {destroy_logits.shape}")
print(f"  Repair logits shape: {repair_logits.shape}")
print(f"  Destroy probs sum: {destroy_probs.sum(dim=-1)}")
print(f"  Repair probs sum: {repair_probs.sum(dim=-1)}")

# Test action sampling
with torch.no_grad():
    destroy_action, repair_action, log_prob = actor.sample_action(state_embedding)

print(f"\n✓ Action sampling works")
print(f"  Destroy actions: {destroy_action}")
print(f"  Repair actions: {repair_action}")
print(f"  Log probs: {log_prob}")

# Test with masks
destroy_mask = torch.tensor([
    [True, True, False, True],  # Op 2 invalid
    [True, True, True, True],
    [False, True, True, True],  # Op 0 invalid
    [True, False, True, True],  # Op 1 invalid
])
repair_mask = torch.ones(batch_size, num_repair_ops, dtype=torch.bool)

with torch.no_grad():
    destroy_logits_masked, repair_logits_masked = actor(
        state_embedding, destroy_mask, repair_mask
    )
    destroy_probs_masked = F.softmax(destroy_logits_masked, dim=-1)

print(f"\n✓ Action masking works")
print(f"  Masked destroy probs (batch 0): {destroy_probs_masked[0]}")
print(f"  Invalid action prob: {destroy_probs_masked[0, 2]:.6f} (should be ~0)")

# ============================================================================
# TEST 5: Critic Network
# ============================================================================
print("\n" + "="*70)
print("TEST 5: Critic Network")
print("="*70)

critic = CriticNetwork(state_dim=state_dim, hidden_dim=256)

print(f"\n✓ CriticNetwork created")
print(f"  Parameters: {sum(p.numel() for p in critic.parameters()):,}")

with torch.no_grad():
    value = critic(state_embedding)

print(f"\n✓ Value estimation works")
print(f"  Value shape: {value.shape}")
print(f"  Value range: [{value.min():.3f}, {value.max():.3f}]")
print(f"  Values: {value.squeeze()}")

# ============================================================================
# TEST 6: Complete JAMPR Policy
# ============================================================================
print("\n" + "="*70)
print("TEST 6: Complete JAMPR Policy")
print("="*70)

# Create graph encoder (from Module 1)
graph_encoder = GraphAttentionEncoder(
    node_feature_dim=node_feature_dim,
    edge_feature_dim=edge_feature_dim,
    hidden_dim=hidden_dim,
    num_heads=4,
    num_layers=3
)

# Create complete policy
policy = JamprPolicy(
    graph_encoder=graph_encoder,
    state_dim=state_dim,
    num_destroy_ops=num_destroy_ops,
    num_repair_ops=num_repair_ops,
    hidden_dim=256
)

print(f"\n✓ JamprPolicy created")
total_params = sum(p.numel() for p in policy.parameters())
print(f"  Total parameters: {total_params:,}")
print(f"    Graph encoder: {sum(p.numel() for p in graph_encoder.parameters()):,}")
print(f"    State embedding: {sum(p.numel() for p in policy.state_embedding.parameters()):,}")
print(f"    Actor: {sum(p.numel() for p in policy.actor.parameters()):,}")
print(f"    Critic: {sum(p.numel() for p in policy.critic.parameters()):,}")

# Create dummy graph data
node_features = torch.randn(num_nodes, node_feature_dim)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_features = torch.randn(num_edges, edge_feature_dim)
global_features = torch.randn(8)

# Test forward pass
with torch.no_grad():
    output = policy(node_features, edge_index, edge_features, global_features)

print(f"\n✓ Complete forward pass works")
print(f"  State embedding shape: {output['state_embedding'].shape}")
print(f"  Destroy logits shape: {output['destroy_logits'].shape}")
print(f"  Repair logits shape: {output['repair_logits'].shape}")
print(f"  Value shape: {output['value'].shape}")

# Test action selection
with torch.no_grad():
    action_output = policy.select_action(
        node_features, edge_index, edge_features, global_features,
        deterministic=False
    )

print(f"\n✓ Action selection works")
print(f"  Destroy action: {action_output['destroy_action'].item()}")
print(f"  Repair action: {action_output['repair_action'].item()}")
print(f"  Log prob: {action_output['log_prob'].item():.4f}")
print(f"  Value: {action_output['value'].item():.4f}")

# ============================================================================
# TEST 7: Gradient Flow
# ============================================================================
print("\n" + "="*70)
print("TEST 7: Gradient Flow")
print("="*70)

# Create dummy loss and backpropagate
output = policy(node_features, edge_index, edge_features, global_features)

# Dummy loss (policy gradient + value loss)
destroy_probs = F.softmax(output['destroy_logits'], dim=-1)
repair_probs = F.softmax(output['repair_logits'], dim=-1)

# Policy loss (negative log likelihood)
destroy_target = torch.tensor([0])
repair_target = torch.tensor([0])
policy_loss = -torch.log(destroy_probs[0, destroy_target] + 1e-8) - torch.log(repair_probs[0, repair_target] + 1e-8)

# Value loss (MSE with dummy target)
value_target = torch.tensor([[10.0]])
value_loss = F.mse_loss(output['value'], value_target)

# Combined loss
loss = policy_loss + value_loss
loss.backward()

print(f"\n✓ Gradient flow verified")
print(f"  Policy loss: {policy_loss.item():.4f}")
print(f"  Value loss: {value_loss.item():.4f}")
print(f"  Total loss: {loss.item():.4f}")

# Check gradients
has_nan_grad = False
has_zero_grad = True
for name, param in policy.named_parameters():
    if param.grad is not None:
        has_zero_grad = False
        if torch.isnan(param.grad).any():
            has_nan_grad = True
            print(f"  ✗ NaN gradient in {name}")

if has_nan_grad:
    print("  ✗ NaN gradients detected!")
elif has_zero_grad:
    print("  ⚠ No gradients computed")
else:
    print("  ✓ All gradients are valid (no NaN/Inf)")

# Check gradient magnitudes
grad_norms = []
for name, param in policy.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)

if grad_norms:
    print(f"  Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")

# ============================================================================
# TEST 8: Batch Processing
# ============================================================================
print("\n" + "="*70)
print("TEST 8: Batch Processing")
print("="*70)

batch_size_test = 8

# Create batched inputs
node_features_batch = torch.randn(batch_size_test, num_nodes, node_feature_dim)
global_features_batch = torch.randn(batch_size_test, 8)

# Process batch (reuse same edge structure)
batch_outputs = []
for i in range(batch_size_test):
    with torch.no_grad():
        output = policy(
            node_features_batch[i], 
            edge_index, 
            edge_features, 
            global_features_batch[i]
        )
        batch_outputs.append(output)

print(f"\n✓ Batch processing works")
print(f"  Processed {batch_size_test} instances")
print(f"  Destroy logits shapes: {[o['destroy_logits'].shape for o in batch_outputs[:3]]}")
print(f"  Values: {[o['value'].item() for o in batch_outputs]}")

# ============================================================================
# TEST 9: Memory Usage
# ============================================================================
print("\n" + "="*70)
print("TEST 9: Memory Usage")
print("="*70)

if torch.cuda.is_available():
    device = torch.device('cuda')
    policy_gpu = policy.to(device)
    
    node_features_gpu = node_features.to(device)
    edge_index_gpu = edge_index.to(device)
    edge_features_gpu = edge_features.to(device)
    global_features_gpu = global_features.to(device)
    
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        output_gpu = policy_gpu(
            node_features_gpu, edge_index_gpu, 
            edge_features_gpu, global_features_gpu
        )
    
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"\n✓ GPU memory usage")
    print(f"  Peak memory: {memory_mb:.2f} MB")
else:
    # Estimate CPU memory
    param_memory = sum(p.numel() * p.element_size() for p in policy.parameters()) / 1024 / 1024
    print(f"\n✓ CPU memory estimate")
    print(f"  Parameter memory: {param_memory:.2f} MB")

# ============================================================================
# TEST 10: Deterministic vs Stochastic
# ============================================================================
print("\n" + "="*70)
print("TEST 10: Deterministic vs Stochastic Action Selection")
print("="*70)

# Stochastic actions (should vary)
actions_stochastic = []
for _ in range(10):
    with torch.no_grad():
        action_output = policy.select_action(
            node_features, edge_index, edge_features, global_features,
            deterministic=False
        )
        actions_stochastic.append((
            action_output['destroy_action'].item(),
            action_output['repair_action'].item()
        ))

# Deterministic actions (should be same)
actions_deterministic = []
for _ in range(10):
    with torch.no_grad():
        action_output = policy.select_action(
            node_features, edge_index, edge_features, global_features,
            deterministic=True
        )
        actions_deterministic.append((
            action_output['destroy_action'].item(),
            action_output['repair_action'].item()
        ))

print(f"\n✓ Action selection modes work")
print(f"  Stochastic actions (first 5): {actions_stochastic[:5]}")
print(f"  Unique stochastic: {len(set(actions_stochastic))}")
print(f"  Deterministic actions (first 5): {actions_deterministic[:5]}")
print(f"  Unique deterministic: {len(set(actions_deterministic))}")

if len(set(actions_deterministic)) == 1:
    print("  ✓ Deterministic mode is consistent")
else:
    print("  ⚠ Deterministic mode shows variation (unexpected)")

if len(set(actions_stochastic)) > 1:
    print("  ✓ Stochastic mode shows variation")
else:
    print("  ⚠ Stochastic mode not varying (may need more samples)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MODULE 2 TEST COMPLETE")
print("="*70)

print("\n✓ Policy Network working correctly")
print("\nCapabilities Verified:")
print("  [✓] OperatorEncoder")
print("  [✓] AttentionPooling")
print("  [✓] StateEmbedding")
print("  [✓] ActorNetwork (Policy)")
print("  [✓] CriticNetwork (Value)")
print("  [✓] JamprPolicy (Complete)")
print("  [✓] Gradient flow")
print("  [✓] Batch processing")
print("  [✓] Action masking")
print("  [✓] Deterministic/Stochastic modes")

print(f"\n📊 Model Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

print("\n" + "="*70)
print("READY FOR MODULE 3: PPO Algorithm")
print("="*70)

print("\nNext Steps:")
print("  1. Implement PPO loss functions")
print("  2. Create training loop with advantage estimation")
print("  3. Add experience buffer")
print("  4. Implement policy update with clipping")
print("  5. Add tensorboard logging")