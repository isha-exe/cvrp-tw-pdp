"""
cvrptw_parser.py
Complete data structures and CSV parser for CVRPTW-PD problem

Phase 0: Foundation
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ReceptacleType:
    """Container/Receptacle type definition"""
    name: str
    capacity: float  # Capacity units (0.75 for TypeA, 1.5 for TypeB)
    
@dataclass
class Order:
    """Transport order (pickup-delivery pair)"""
    id: int
    from_node: int
    to_node: int
    from_time: datetime  # Pickup time window start
    to_time: datetime    # Pickup time window end
    receptacle_type: str
    quantity: float      # Number of receptacles
    capacity_required: float  # Total capacity = quantity * receptacle_capacity
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class VehicleType:
    """Vehicle type specification"""
    name: str
    capacity: float
    fixed_cost: float
    variable_cost_per_km: float
    change_dock_time: int  # minutes
    folded_load_time: int
    folded_unload_time: int
    unfolded_load_time: int
    unfolded_unload_time: int

@dataclass
class Vehicle:
    """Individual vehicle instance"""
    number: str
    vehicle_type: VehicleType
    restricted_nodes: Set[int] = field(default_factory=set)
    
    @property
    def capacity(self) -> float:
        return self.vehicle_type.capacity
    
    def can_visit(self, node: int) -> bool:
        """Check if vehicle can visit a node"""
        return node not in self.restricted_nodes

@dataclass
class Location:
    """Node/Location information"""
    node_id: int
    break_allowed: bool
    disallowed_vehicle_types: Set[str] = field(default_factory=set)

@dataclass
class ServiceParameters:
    """Global service configuration"""
    worktime_standard: int  # minutes
    worktime_nb: int
    setup_time: int
    windup_time: int
    idle_time_duration: int
    service_cost: float
    break_time: int
    max_drive_time_before_break: int
    max_work_time_before_break: int
    trip_buffer_time: int

@dataclass
class Task:
    """Individual task in a service (pickup, delivery, break)"""
    task_type: str  # 'PICKUP', 'DELIVERY', 'BREAK', 'START', 'END'
    node: int
    order: Optional[Order] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: int = 0  # minutes
    
@dataclass
class Service:
    """A service (shift) for a vehicle"""
    service_id: int
    vehicle: Vehicle
    start_node: int
    end_node: int
    tasks: List[Task] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_distance: float = 0.0
    total_driving_time: float = 0.0
    total_idle_time: float = 0.0
    break_assigned: bool = False
    
    def duration_minutes(self) -> float:
        """Calculate service duration in minutes"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return 0.0
    
    def is_feasible(self, params: ServiceParameters) -> Tuple[bool, List[str]]:
        """Check service feasibility"""
        violations = []
        
        # Check duration constraint
        duration = self.duration_minutes()
        if duration > params.worktime_standard:
            violations.append(f"Service duration {duration:.1f} exceeds max {params.worktime_standard}")
        
        # Check capacity
        total_capacity = sum(order.capacity_required for order in self.orders)
        if total_capacity > self.vehicle.capacity:
            violations.append(f"Capacity {total_capacity:.2f} exceeds vehicle capacity {self.vehicle.capacity}")
        
        # Check break requirement
        if not self.break_assigned and duration > 0:
            violations.append("Break not assigned")
        
        # Check start/end at same location
        if self.start_node != self.end_node:
            violations.append(f"Service doesn't return to origin (start={self.start_node}, end={self.end_node})")
        
        return len(violations) == 0, violations

@dataclass
class Solution:
    """Complete solution"""
    services: List[Service] = field(default_factory=list)
    unserved_orders: List[Order] = field(default_factory=list)
    
    def num_services(self) -> int:
        return len(self.services)
    
    def num_vehicles_used(self) -> int:
        """Count unique vehicles used"""
        return len(set(s.vehicle.number for s in self.services))
    
    def total_distance(self) -> float:
        return sum(s.total_distance for s in self.services)
    
    def total_driving_time(self) -> float:
        return sum(s.total_driving_time for s in self.services)
    
    def total_idle_time(self) -> float:
        return sum(s.total_idle_time for s in self.services)
    
    def coverage_rate(self, total_orders: int) -> float:
        """Calculate percentage of orders served"""
        served = total_orders - len(self.unserved_orders)
        return served / total_orders if total_orders > 0 else 0.0
    def copy(self):
        """Create a deep copy of the solution"""
        import copy
        return copy.deepcopy(self)

# ============================================================================
# DATA LOADER
# ============================================================================

class ProblemInstance:
    """Complete problem instance with all data"""
    
    def __init__(self):
        self.orders: List[Order] = []
        self.vehicles: List[Vehicle] = []
        self.vehicle_types: Dict[str, VehicleType] = {}
        self.receptacle_types: Dict[str, ReceptacleType] = {}
        self.locations: Dict[int, Location] = {}
        self.travel_time_matrix: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.params: Optional[ServiceParameters] = None
        self.base_date = datetime(2024, 1, 1)  # Base date for time windows
        
    def load_from_files(self, file_paths: Dict[str, str]):
        """Load all data from CSV files"""
        
        # Load receptacle types
        receptacles_df = pd.read_csv(file_paths['receptacles'])
        for _, row in receptacles_df.iterrows():
            self.receptacle_types[row['ReceptacleType']] = ReceptacleType(
                name=row['ReceptacleType'],
                capacity=row['Capacity_decimal']
            )
        
        # Load vehicle types
        vehicle_types_df = pd.read_csv(file_paths['vehicle_types'])
        for _, row in vehicle_types_df.iterrows():
            self.vehicle_types[row['VehicleType']] = VehicleType(
                name=row['VehicleType'],
                capacity=row['Capacity_decimal'],
                fixed_cost=row['FixedCost'],
                variable_cost_per_km=row['VariableCost_per_km'],
                change_dock_time=row['changeDock_time_min'],
                folded_load_time=row['folded_load_time_min'],
                folded_unload_time=row['folded_unload_time_min'],
                unfolded_load_time=row['unfolded_load_time_min'],
                unfolded_unload_time=row['unfolded_unload_time_min']
            )
        
        # Load vehicle restrictions
        restrictions_df = pd.read_csv(file_paths['vehicle_restrictions'])
        node_restrictions = {}
        for _, row in restrictions_df.iterrows():
            node = row['Node']
            disallowed = row['DisallowedVehicleTypes']
            if pd.notna(disallowed) and disallowed.strip():
                node_restrictions[node] = set(v.strip() for v in disallowed.split(','))
            else:
                node_restrictions[node] = set()
        
        # Load fleet
        fleet_df = pd.read_csv(file_paths['fleet'])
        for _, row in fleet_df.iterrows():
            vehicle_type_name = row['VehicleType']
            vehicle_type = self.vehicle_types[vehicle_type_name]
            
            # Find restricted nodes for this vehicle type
            restricted_nodes = set()
            for node, disallowed_types in node_restrictions.items():
                if vehicle_type_name in disallowed_types:
                    restricted_nodes.add(node)
            
            vehicle = Vehicle(
                number=row['VehicleNumber'],
                vehicle_type=vehicle_type,
                restricted_nodes=restricted_nodes
            )
            self.vehicles.append(vehicle)
        
        # Load break allowed locations
        break_df = pd.read_csv(file_paths['break_allowed'])
        for _, row in break_df.iterrows():
            node_id = row['Node']
            self.locations[node_id] = Location(
                node_id=node_id,
                break_allowed=row['BreakAllowed'] == 'Y',
                disallowed_vehicle_types=node_restrictions.get(node_id, set())
            )
        
        # Load orders
        orders_df = pd.read_csv(file_paths['orders'])
        for _, row in orders_df.iterrows():
            receptacle_type = self.receptacle_types[row['ReceptacleType']]
            from_time = self._parse_time(row['FromTime'])
            to_time = self._parse_time(row['ToTime'])
            
            order = Order(
                id=row['ID'],
                from_node=row['FromNode'],
                to_node=row['ToNode'],
                from_time=from_time,
                to_time=to_time,
                receptacle_type=row['ReceptacleType'],
                quantity=row['Quantity'],
                capacity_required=row['Quantity'] * receptacle_type.capacity
            )
            self.orders.append(order)
        
        # Load travel times
        travel_df = pd.read_csv(file_paths['travel_times'])
        for _, row in travel_df.iterrows():
            key = (row['SourceNode'], row['DestinationNode'])
            self.travel_time_matrix[key] = (row['Distance_km'], row['MeanDuration_min'])
        
        # Load service parameters
        service_df = pd.read_csv(file_paths['service_items'])
        params_dict = dict(zip(service_df['Item'], service_df['Value']))
        self.params = ServiceParameters(
            worktime_standard=params_dict['worktimestandardservice'],
            worktime_nb=params_dict['worktimeNBService'],
            setup_time=params_dict['service setup time'],
            windup_time=params_dict['service windup time'],
            idle_time_duration=params_dict['idle time duration'],
            service_cost=params_dict['service cost'],
            break_time=params_dict['breaktime'],
            max_drive_time_before_break=params_dict['MaxDriveTimeBeforeBreak'],
            max_work_time_before_break=params_dict['MaxWorkTimeBeforeBreak'],
            trip_buffer_time=params_dict['Trip buffertime']
        )
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse HH:MM time string to datetime"""
        time_obj = datetime.strptime(time_str, '%H:%M').time()
        return datetime.combine(self.base_date, time_obj)
    
    def get_travel_time(self, from_node: int, to_node: int) -> Tuple[float, float]:
        """Get distance and travel time between nodes"""
        return self.travel_time_matrix.get((from_node, to_node), (0.0, 0.0))
    
    def print_summary(self):
        """Print problem instance summary"""
        print("=" * 60)
        print("PROBLEM INSTANCE SUMMARY")
        print("=" * 60)
        print(f"Orders: {len(self.orders)}")
        print(f"Vehicles: {len(self.vehicles)}")
        print(f"Vehicle Types: {len(self.vehicle_types)}")
        print(f"Locations: {len(self.locations)}")
        print(f"\nService Parameters:")
        print(f"  Max worktime: {self.params.worktime_standard} min ({self.params.worktime_standard/60:.1f} hours)")
        print(f"  Break time: {self.params.break_time} min")
        print(f"  Max drive before break: {self.params.max_drive_time_before_break} min")
        print(f"  Max work before break: {self.params.max_work_time_before_break} min")
        
        print(f"\nVehicle Fleet:")
        for vtype, count in pd.Series([v.vehicle_type.name for v in self.vehicles]).value_counts().items():
            capacity = self.vehicle_types[vtype].capacity
            print(f"  {vtype}: {count} vehicles (capacity: {capacity})")
        
        print(f"\nOrder Statistics:")
        total_capacity = sum(o.capacity_required for o in self.orders)
        print(f"  Total capacity required: {total_capacity:.2f}")
        print(f"  Avg capacity per order: {total_capacity/len(self.orders):.2f}")
        print(f"  TypeA orders: {sum(1 for o in self.orders if o.receptacle_type == 'TypeA')}")
        print(f"  TypeB orders: {sum(1 for o in self.orders if o.receptacle_type == 'TypeB')}")
        print("=" * 60)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # File paths
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
    
    # Load problem instance
    problem = ProblemInstance()
    problem.load_from_files(file_paths)
    problem.print_summary()
    
    # Example: Check which vehicles can serve order 1
    order1 = problem.orders[0]
    print(f"\nOrder {order1.id}: {order1.from_node} -> {order1.to_node}")
    print(f"  Capacity required: {order1.capacity_required:.2f}")
    print(f"  Compatible vehicles:")
    for vehicle in problem.vehicles:
        can_pickup = vehicle.can_visit(order1.from_node)
        can_deliver = vehicle.can_visit(order1.to_node)
        has_capacity = vehicle.capacity >= order1.capacity_required
        if can_pickup and can_deliver and has_capacity:
            print(f"    ✓ {vehicle.number} ({vehicle.vehicle_type.name}, cap: {vehicle.capacity})")