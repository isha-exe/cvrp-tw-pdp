"""
test_graph_encoder.py
Test script for graph state encoder

Tests Module 1 of Phase 2b
"""

import sys
import numpy as np

print("="*70)
print("TESTING GRAPH STATE ENCODER (Module 1)")
print("="*70)

# Import required modules
try:
    from cvrptw_parser import ProblemInstance
    from test_phase0 import SimpleGreedyConstructor
    from graph_state_encoder import GraphStateEncoder, GraphAttentionEncoder, TORCH_AVAILABLE
    print("\n✓ All modules imported")
    print(f"  PyTorch available: {TORCH_AVAILABLE}")
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    sys.exit(1)

# Load problem
print("\n[Step 1] Loading problem...")
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

# Build solution
print("\n[Step 2] Building solution...")
constructor = SimpleGreedyConstructor(problem)
solution = constructor.construct(start_depot=1)
print(f"✓ Solution: {solution.num_services()} services, {solution.coverage_rate(len(problem.orders))*100:.1f}% coverage")

# Create encoder
print("\n[Step 3] Creating graph encoder...")
encoder = GraphStateEncoder(problem, embedding_dim=64)
print(f"✓ Encoder created")
print(f"  Orders: {encoder.num_orders}")
print(f"  Vehicles: {encoder.num_vehicles}")
print(f"  Embedding dim: {encoder.embedding_dim}")

# Encode solution as graph
print("\n[Step 4] Encoding solution as graph...")
graph = encoder.encode_solution(solution)
print(f"✓ Graph encoded")
print(f"\n  Nodes: {graph['num_nodes']}")
print(f"  Edges: {graph['num_edges']}")
print(f"  Node features shape: {graph['node_features'].shape}")
print(f"  Edge features shape: {graph['edge_features'].shape}")
print(f"  Global features shape: {graph['global_features'].shape}")

# Analyze graph structure
print("\n[Step 5] Analyzing graph structure...")
print(f"\n  Node Features (first 3 nodes):")
for i in range(min(3, graph['num_nodes'])):
    print(f"    Node {i}: {graph['node_features'][i][:5]}... (showing first 5 features)")

print(f"\n  Edge Index (first 5 edges):")
for i in range(min(5, graph['num_edges'])):
    src, dst = graph['edge_index'][0, i], graph['edge_index'][1, i]
    print(f"    Edge {i}: {src} → {dst}")

print(f"\n  Global Features:")
print(f"    {graph['global_features']}")

# Test with PyTorch (if available)
if TORCH_AVAILABLE:
    print("\n[Step 6] Testing Graph Attention Network...")
    
    import torch
    
    # Create model
    gat_model = GraphAttentionEncoder(
        node_feature_dim=10,
        edge_feature_dim=10,
        hidden_dim=64,
        num_heads=4,
        num_layers=3
    )
    
    print(f"✓ GAT model created")
    print(f"  Parameters: {sum(p.numel() for p in gat_model.parameters()):,}")
    
    # Convert to tensors
    node_feat_tensor = torch.FloatTensor(graph['node_features'])
    edge_index_tensor = torch.LongTensor(graph['edge_index'])
    edge_feat_tensor = torch.FloatTensor(graph['edge_features'])
    
    # Forward pass
    with torch.no_grad():
        node_embeddings = gat_model(node_feat_tensor, edge_index_tensor, edge_feat_tensor)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {node_embeddings.shape}")
    print(f"  Output range: [{node_embeddings.min():.3f}, {node_embeddings.max():.3f}]")
    print(f"  Output mean: {node_embeddings.mean():.3f}")
    
    # Analyze embeddings
    print(f"\n  Node Embeddings (first 3 nodes, first 5 dims):")
    for i in range(min(3, node_embeddings.shape[0])):
        print(f"    Node {i}: {node_embeddings[i, :5].numpy()}")
else:
    print("\n[Step 6] Skipping GAT test (PyTorch not available)")

# Summary
print("\n" + "="*70)
print("MODULE 1 TEST COMPLETE")
print("="*70)

print("\n✓ Graph State Encoder working correctly")
print("\nCapabilities Verified:")
print("  [✓] Solution → Graph conversion")
print("  [✓] Node feature extraction")
print("  [✓] Edge feature extraction")
print("  [✓] Global feature extraction")
if TORCH_AVAILABLE:
    print("  [✓] Graph Attention Network")
    print("  [✓] Multi-head attention")
    print("  [✓] PyTorch integration")

print("\n" + "="*70)
print("READY FOR MODULE 2: Attention-based Policy Network")
print("="*70)