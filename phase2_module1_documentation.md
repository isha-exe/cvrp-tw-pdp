# Phase 2b Module 1: Analysis & Optimization Guide

## 📊 Current Implementation Analysis

### Strengths ✓
1. **Well-structured graph representation**
   - Proper node types (depot, pickups, deliveries)
   - Two edge types (route, precedence)
   - Comprehensive feature engineering

2. **Good neural architecture foundation**
   - Multi-head attention mechanism
   - Proper Q, K, V projections
   - Modular design

### Critical Issues 🔴

#### 1. **Performance Bottlenecks**

| Issue | Impact | Location |
|-------|--------|----------|
| Repeated dictionary lookups | O(n) per call | `_get_node_location()` |
| No feature caching | Recomputes static features | `_build_static_features()` |
| Inefficient edge construction | O(n²) worst case | `encode_solution()` |
| Manual scatter operations | 10-50x slower | `GraphAttentionLayer.forward()` |

#### 2. **Memory Issues**

```python
# PROBLEM: Creates new tensors for every operation
Q[dst] * (K[src] + E)  # Creates 3 intermediate tensors
```

**Impact**: For 100 nodes, 500 edges → ~50MB per forward pass

#### 3. **Numerical Stability**

```python
# PROBLEM: No gradient clipping or normalization
attention_scores = (q_dst * (k_src + E)).sum(dim=-1) / np.sqrt(self.head_dim)
```

**Risk**: Exploding/vanishing gradients during training

---

## 🚀 Optimization Strategies

### Priority 1: Performance (Must Have)

#### A. Cache Static Features
```python
# BEFORE: Recomputes every time
for order in self.problem.orders:
    pickup_features = {...}  # Computed every encode_solution() call

# AFTER: Compute once in __init__
self.static_pickup_features = np.zeros((num_orders, 8))
# ... populate once
```

**Expected Speedup**: 3-5x for encoding

#### B. Use torch_scatter for Aggregation
```python
# BEFORE: Manual loop (slow)
for i in range(num_edges):
    output[dst[i]] += messages[i]

# AFTER: Vectorized scatter_add
output = scatter_add(messages, dst, dim=0, dim_size=num_nodes)
```

**Expected Speedup**: 10-50x for attention layers

#### C. Vectorize Feature Construction
```python
# BEFORE: Loop over orders
for order in self.problem.orders:
    features = self._encode_order_node(order, ...)
    node_features.append(features)

# AFTER: Batch operation
node_features[1:num_orders+1] = self._batch_encode_pickups(order_served_mask)
```

**Expected Speedup**: 5-10x for encoding

### Priority 2: Memory Efficiency

#### A. In-place Operations
```python
# Use .add_() instead of +
x = x + residual  # Creates new tensor
x.add_(residual)  # In-place, saves memory
```

#### B. Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for layer in self.attention_layers:
        x = checkpoint(layer, x)  # Saves memory during backward
```

### Priority 3: Numerical Stability

#### A. Layer Normalization
```python
self.node_norm = nn.LayerNorm(hidden_dim)
x = self.node_norm(self.node_embedding(node_features))
```

#### B. Gradient Clipping
```python
# During training
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 📋 Module 2: Attention-based Policy Network

### Architecture Overview

```
Input State (Graph)
      ↓
[Graph Encoder] ← Module 1
      ↓
[State Embedding] (64D per node)
      ↓
[Attention Pooling] → Global context
      ↓
┌─────────────────────┬─────────────────────┐
│   Actor Network     │   Critic Network    │
│  (Policy π_θ)       │   (Value V_φ)       │
├─────────────────────┼─────────────────────┤
│ Operator Selection  │ State Value         │
│ Action Logits       │ V(s) ∈ ℝ            │
│ π(a|s) ∈ ℝ^|A|      │                     │
└─────────────────────┴─────────────────────┘
```

### Key Components

#### 1. **State Embedding Layer**
```python
class StateEmbedding(nn.Module):
    """
    Processes graph encoding into state representation
    """
    def __init__(self, node_dim=64, global_dim=128):
        self.node_encoder = GraphAttentionEncoder(...)
        self.global_pooling = AttentionPooling(node_dim, global_dim)
```

#### 2. **Operator Encoder**
```python
class OperatorEncoder(nn.Module):
    """
    Encodes destruction/repair operators
    
    Operators for CVRPTW:
    - Worst removal
    - Random removal  
    - Shaw removal (similarity-based)
    - Greedy insertion
    - Regret insertion
    """
    def forward(self, operator_id):
        return self.operator_embeddings[operator_id]
```

#### 3. **Actor Network** (Policy)
```python
class ActorNetwork(nn.Module):
    """
    Selects which operator to apply
    
    Output: Probability distribution over operators
    π(a|s) where a ∈ {destroy_ops} × {repair_ops}
    """
    def forward(self, state_embedding):
        logits = self.policy_head(state_embedding)
        return F.softmax(logits, dim=-1)
```

#### 4. **Critic Network** (Value Function)
```python
class CriticNetwork(nn.Module):
    """
    Estimates expected return from current state
    
    Output: Scalar value V(s)
    """
    def forward(self, state_embedding):
        return self.value_head(state_embedding)
```

### Module 2 Implementation Plan

#### **File 1: `policy_network.py`** (Core)
```python
# Components:
1. StateEmbedding (uses Module 1 encoder)
2. OperatorEncoder
3. ActorNetwork
4. CriticNetwork
5. JamprPolicy (combines actor + critic)
```

#### **File 2: `attention_layers.py`** (Utilities)
```python
# Components:
1. AttentionPooling (global context)
2. CrossAttention (operator-state interaction)
3. PositionalEncoding (sequence position)
```

#### **File 3: `test_policy.py`** (Testing)
```python
# Tests:
1. Forward pass with dummy data
2. Action sampling
3. Value estimation
4. Gradient flow
```

---

## 🎯 Immediate Action Items

### Week 1: Optimize Module 1
- [ ] Implement feature caching
- [ ] Add torch_scatter integration
- [ ] Vectorize encoding operations
- [ ] Add layer normalization
- [ ] Benchmark: Target 5x speedup

### Week 2: Build Module 2 Foundation
- [ ] Implement StateEmbedding
- [ ] Implement OperatorEncoder  
- [ ] Create basic Actor/Critic networks
- [ ] Test forward pass

### Week 3: Complete Module 2
- [ ] Add attention pooling
- [ ] Implement operator compatibility scoring
- [ ] Add masking for invalid operators
- [ ] Integration testing with Module 1

---

## 📈 Performance Targets

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Encoding time | ~100ms | <20ms | 🔴 |
| Forward pass | ~50ms | <10ms | 🔴 |
| Memory usage | ~200MB | <50MB | 🔴 |
| Batch size | 1 | 32 | 🔴 |

---

## 🔧 Quick Fixes for Immediate Use

### 1. Add to `GraphStateEncoder.__init__`:
```python
self._feature_cache = {}
self._travel_time_cache = {}
```

### 2. Replace in `_encode_order_node`:
```python
# Cache key
cache_key = (order.id, is_pickup, is_served)
if cache_key in self._feature_cache:
    return self._feature_cache[cache_key].copy()
```

### 3. Add to `GraphAttentionEncoder`:
```python
self.layer_norms = nn.ModuleList([
    nn.LayerNorm(hidden_dim) for _ in range(num_layers)
])
```

---

## 📚 References for Module 2

1. **JAMPR Paper**: Focus on Section 3.2 (Policy Network Architecture)
2. **Attention Is All You Need**: Multi-head attention mechanism
3. **PPO Paper**: Actor-critic architecture details
4. **Graph Neural Networks**: Message passing variants

---

## ✅ Validation Checklist

Before moving to Module 3 (PPO):

- [ ] Module 1 encoding <20ms for 50 orders
- [ ] Module 2 forward pass <10ms
- [ ] Memory usage <50MB per episode
- [ ] Batch size ≥32 supported
- [ ] Gradient flow verified (no NaN/Inf)
- [ ] Action sampling works correctly
- [ ] Value estimation reasonable range

---

**Next Steps**: I can help you implement any of these optimizations or start building Module 2. Which would you like to tackle first?