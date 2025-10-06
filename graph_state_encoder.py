"""
graph_state_encoder.py
Graph-based state encoder with attention mechanism for JAMPR

Phase 2b - Module 1: Encodes VRP solutions as graphs
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch required for Phase 2b")


class GraphStateEncoder:
    """
    Encodes solution state as a graph with node embeddings and attention
    
    Graph Structure:
    - Nodes: Depot, Orders (pickup/delivery pairs), Vehicles
    - Edges: Routes, temporal sequence, spatial proximity
    - Features: Location, time windows, capacity, current assignment
    """
    
    def __init__(self, problem: 'ProblemInstance', embedding_dim: int = 64):
        self.problem = problem
        self.embedding_dim = embedding_dim
        
        # Problem dimensions
        self.num_orders = len(problem.orders)
        self.num_vehicles = len(problem.vehicles)
        self.num_locations = len(problem.locations)
        
        # Build node and edge features
        self._build_static_features()
    
    def _build_static_features(self):
        """Build static node and edge features from problem instance"""
        
        # Node features: orders (pickup + delivery as separate nodes)
        self.order_features = {}
        for order in self.problem.orders:
            # Pickup node features
            pickup_features = {
                'type': 'pickup',
                'order_id': order.id,
                'location': order.from_node,
                'time_window_start': order.from_time.hour * 60 + order.from_time.minute,
                'time_window_end': order.to_time.hour * 60 + order.to_time.minute,
                'capacity': order.capacity_required,
                'receptacle_type': 1.0 if order.receptacle_type == 'TypeA' else 2.0,
            }
            
            # Delivery node features
            delivery_features = {
                'type': 'delivery',
                'order_id': order.id,
                'location': order.to_node,
                'time_window_start': 0,  # No explicit delivery window
                'time_window_end': 1440,  # End of day
                'capacity': -order.capacity_required,  # Negative (unloading)
                'receptacle_type': 1.0 if order.receptacle_type == 'TypeA' else 2.0,
            }
            
            self.order_features[f'P{order.id}'] = pickup_features
            self.order_features[f'D{order.id}'] = delivery_features
        
        # Vehicle features
        self.vehicle_features = {}
        for vehicle in self.problem.vehicles:
            self.vehicle_features[vehicle.number] = {
                'capacity': vehicle.capacity,
                'vehicle_type': vehicle.vehicle_type.name,
                'restricted_nodes': vehicle.restricted_nodes,
                'fixed_cost': vehicle.vehicle_type.fixed_cost,
                'variable_cost': vehicle.vehicle_type.variable_cost_per_km,
            }
        
        # Location features (spatial)
        self.location_features = {}
        for node_id, location in self.problem.locations.items():
            self.location_features[node_id] = {
                'break_allowed': float(location.break_allowed),
                'num_restrictions': len(location.disallowed_vehicle_types),
            }
    
    def encode_solution(self, solution: 'Solution') -> Dict[str, np.ndarray]:
        """
        Encode solution state as graph representation
        
        Returns dictionary with:
        - 'node_features': [num_nodes, feature_dim]
        - 'edge_index': [2, num_edges]
        - 'edge_features': [num_edges, edge_feature_dim]
        - 'global_features': [global_feature_dim]
        """
        
        # Build node list
        nodes = []
        node_features = []
        node_to_idx = {}
        
        # Add depot node
        node_to_idx['depot'] = len(nodes)
        nodes.append('depot')
        depot_features = self._encode_depot_node()
        node_features.append(depot_features)
        
        # Add order nodes (pickup + delivery)
        for order in self.problem.orders:
            # Pickup node
            pickup_id = f'P{order.id}'
            node_to_idx[pickup_id] = len(nodes)
            nodes.append(pickup_id)
            pickup_features = self._encode_order_node(order, solution, is_pickup=True)
            node_features.append(pickup_features)
            
            # Delivery node
            delivery_id = f'D{order.id}'
            node_to_idx[delivery_id] = len(nodes)
            nodes.append(delivery_id)
            delivery_features = self._encode_order_node(order, solution, is_pickup=False)
            node_features.append(delivery_features)
        
        # Stack node features
        node_features = np.array(node_features, dtype=np.float32)
        
        # Build edge list and features
        edge_index = []
        edge_features = []
        
        # Add route edges (sequential tasks in services)
        for service in solution.services:
            prev_node_id = 'depot'
            for task in service.tasks:
                if task.task_type == 'PICKUP':
                    curr_node_id = f'P{task.order.id}'
                elif task.task_type == 'DELIVERY':
                    curr_node_id = f'D{task.order.id}'
                else:
                    continue  # Skip breaks
                
                # Add directed edge
                if prev_node_id in node_to_idx and curr_node_id in node_to_idx:
                    edge_index.append([node_to_idx[prev_node_id], node_to_idx[curr_node_id]])
                    edge_feat = self._encode_route_edge(prev_node_id, curr_node_id, service)
                    edge_features.append(edge_feat)
                    prev_node_id = curr_node_id
            
            # Return to depot
            if prev_node_id != 'depot' and prev_node_id in node_to_idx:
                edge_index.append([node_to_idx[prev_node_id], node_to_idx['depot']])
                edge_feat = self._encode_route_edge(prev_node_id, 'depot', service)
                edge_features.append(edge_feat)
        
        # Add precedence edges (pickup -> delivery)
        for order in self.problem.orders:
            pickup_id = f'P{order.id}'
            delivery_id = f'D{order.id}'
            edge_index.append([node_to_idx[pickup_id], node_to_idx[delivery_id]])
            edge_feat = self._encode_precedence_edge(order)
            edge_features.append(edge_feat)
        
        # Convert to arrays
        edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 10), dtype=np.float32)
        
        # Global features
        global_features = self._encode_global_features(solution)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'global_features': global_features,
            'num_nodes': len(nodes),
            'num_edges': edge_index.shape[1]
        }
    
    def _encode_depot_node(self) -> np.ndarray:
        """Encode depot node features"""
        features = [
            1.0,  # is_depot
            0.0,  # is_pickup
            0.0,  # is_delivery
            0.0,  # capacity_change
            0.5,  # normalized_location (depot is 1)
            0.0,  # time_window_start_norm
            1.0,  # time_window_end_norm
            1.0,  # break_allowed
            0.0,  # is_served
            0.0,  # service_id_norm
        ]
        return np.array(features, dtype=np.float32)
    
    def _encode_order_node(self, order: 'Order', solution: 'Solution', 
                          is_pickup: bool) -> np.ndarray:
        """Encode order node (pickup or delivery) features"""
        
        # Check if order is served
        is_served = order not in solution.unserved_orders
        service_id = 0
        if is_served:
            for service in solution.services:
                if order in service.orders:
                    service_id = service.service_id
                    break
        
        features = [
            0.0,  # is_depot
            1.0 if is_pickup else 0.0,  # is_pickup
            0.0 if is_pickup else 1.0,  # is_delivery
            order.capacity_required if is_pickup else -order.capacity_required,  # capacity_change
            (order.from_node if is_pickup else order.to_node) / 12.0,  # normalized_location
            order.from_time.hour / 24.0,  # time_window_start_norm
            order.to_time.hour / 24.0,  # time_window_end_norm
            1.0 if self.problem.locations.get(order.from_node if is_pickup else order.to_node, 
                                             None) and self.problem.locations[order.from_node if is_pickup else order.to_node].break_allowed else 0.0,
            1.0 if is_served else 0.0,  # is_served
            service_id / 10.0,  # service_id_norm
        ]
        return np.array(features, dtype=np.float32)
    
    def _encode_route_edge(self, from_node: str, to_node: str, service: 'Service') -> np.ndarray:
        """Encode edge features for route connections"""
        
        # Get locations
        from_loc = self._get_node_location(from_node)
        to_loc = self._get_node_location(to_node)
        
        # Get travel distance and time
        if from_loc and to_loc:
            distance, travel_time = self.problem.get_travel_time(from_loc, to_loc)
        else:
            distance, travel_time = 0.0, 0.0
        
        features = [
            1.0,  # is_route_edge
            0.0,  # is_precedence_edge
            distance / 200.0,  # normalized_distance
            travel_time / 500.0,  # normalized_travel_time
            service.service_id / 10.0,  # service_id
            service.vehicle.capacity / 12.0,  # vehicle_capacity_norm
            len(service.orders) / 10.0,  # service_load_norm
            service.duration_minutes() / 456.0,  # service_duration_norm
            1.0 if service.break_assigned else 0.0,  # has_break
            service.total_distance / 200.0,  # total_route_distance_norm
        ]
        return np.array(features, dtype=np.float32)
    
    def _encode_precedence_edge(self, order: 'Order') -> np.ndarray:
        """Encode precedence constraint edge (pickup must precede delivery)"""
        distance, travel_time = self.problem.get_travel_time(order.from_node, order.to_node)
        
        features = [
            0.0,  # is_route_edge
            1.0,  # is_precedence_edge
            distance / 200.0,  # normalized_distance
            travel_time / 500.0,  # normalized_travel_time
            order.id / 12.0,  # order_id_norm
            order.capacity_required / 10.0,  # capacity_norm
            0.0,  # unused
            0.0,  # unused
            0.0,  # unused
            0.0,  # unused
        ]
        return np.array(features, dtype=np.float32)
    
    def _encode_global_features(self, solution: 'Solution') -> np.ndarray:
        """Encode global solution-level features"""
        from constraint_validator import ConstraintValidator, ObjectiveCalculator
        
        validator = ConstraintValidator(self.problem)
        objective = ObjectiveCalculator(self.problem)
        
        is_feasible, violations = validator.validate_solution(solution)
        cost, components = objective.calculate(solution)
        
        features = [
            solution.num_services() / 10.0,
            solution.num_vehicles_used() / 10.0,
            solution.total_distance() / 1000.0,
            solution.total_driving_time() / 2000.0,
            solution.coverage_rate(len(self.problem.orders)),
            len(solution.unserved_orders) / 12.0,
            len(violations) / 20.0,
            1.0 if is_feasible else 0.0,
        ]
        return np.array(features, dtype=np.float32)
    
    def _get_node_location(self, node_id: str) -> Optional[int]:
        """Get physical location for a node"""
        if node_id == 'depot':
            return 1
        elif node_id.startswith('P'):
            order_id = int(node_id[1:])
            order = next(o for o in self.problem.orders if o.id == order_id)
            return order.from_node
        elif node_id.startswith('D'):
            order_id = int(node_id[1:])
            order = next(o for o in self.problem.orders if o.id == order_id)
            return order.to_node
        return None


class GraphAttentionEncoder(nn.Module):
    """
    Graph Attention Network for processing graph state
    Uses multi-head attention to aggregate node features
    """
    
    def __init__(self, node_feature_dim: int = 10, 
                 edge_feature_dim: int = 10,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3):
        super(GraphAttentionEncoder, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GraphAttentionEncoder")
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node feature embedding
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # Edge feature embedding
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Graph attention layers
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, node_features: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph attention network
        
        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_feature_dim]
        
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        # Embed features
        x = self.node_embedding(node_features)
        edge_attr = self.edge_embedding(edge_features)
        
        # Apply attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer with multi-head attention"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # Attention parameters
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(in_dim, out_dim)
        
        # Output
        self.out_proj = nn.Linear(out_dim, out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Apply graph attention
        
        Args:
            x: [num_nodes, in_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, in_dim]
        
        Returns:
            output: [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        
        # Compute query, key, value
        Q = self.query(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.key(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.value(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Edge features
        E = self.edge_proj(edge_attr).view(-1, self.num_heads, self.head_dim)
        
        # Attention scores
        src, dst = edge_index[0], edge_index[1]
        
        # Q[dst] * (K[src] + E)
        attention_scores = (Q[dst] * (K[src] + E)).sum(dim=-1) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Aggregate
        messages = (attention_weights.unsqueeze(-1) * V[src]).view(-1, self.out_dim)
        
        # Scatter to destination nodes
        output = torch.zeros(num_nodes, self.out_dim, device=x.device)
        output.index_add_(0, dst, messages)
        
        # Add residual and project
        output = self.out_proj(output) + x if x.size(-1) == self.out_dim else self.out_proj(output)
        
        return output


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Graph State Encoder - Module 1")
    print("="*60)
    print("\nCapabilities:")
    print("  - Encodes VRP solutions as graphs")
    print("  - Node features: depot, pickups, deliveries")
    print("  - Edge features: routes, precedence constraints")
    print("  - Attention mechanism for aggregation")
    print("\nReady for integration with Phase 2b!")