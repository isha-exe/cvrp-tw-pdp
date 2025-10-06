# Phase 0: Foundation - Complete Documentation

## 📋 Overview

Phase 0 establishes the foundational infrastructure for solving the Capacitated Vehicle Routing Problem with Time Windows and Pickup-Delivery constraints (CVRPTW-PD).

---

## 🎯 What Phase 0 Accomplishes

### 1. **Data Management**
- Load and parse 8 CSV files
- Validate data integrity
- Create structured problem instances

### 2. **Problem Representation**
- Define all entities (Orders, Vehicles, Services, etc.)
- Handle pickup-delivery pairs
- Track capacity, time windows, restrictions

### 3. **Constraint Validation**
- Check all hard constraints (capacity, time windows, precedence, breaks, etc.)
- Identify violations
- Return feasibility status

### 4. **Objective Calculation**
- Multi-objective cost function
- Service metrics (distance, time, coverage)
- Weighted scoring

### 5. **Baseline Solution**
- Greedy constructor for initial solutions
- Provides starting point for optimization

---

## 📁 File Structure

```
your_project/
├── Receptacles.csv          # Container types
├── VehicleTypes.csv         # Vehicle specifications
├── VehicleRestrictions.csv  # Location access rules
├── Fleet.csv                # Available vehicles
├── BreakAllowed.csv         # Break locations
├── Orders.csv               # Transport orders
├── TravelTimes.csv          # Distance matrix
├── ServiceItems.csv         # Service parameters
│
├── cvrptw_parser.py         # Core data structures & loader
├── constraint_validator.py  # Feasibility checker
├── test_phase0.py           # Complete test script
└── README_phase0.md         # This documentation
```

---

## 🔧 Components Breakdown

### **Component 1: Data Structures** (`cvrptw_parser.py`)

#### Key Classes:

**Order** - A transport request
```python
Order(
    id=1,                          # Order ID
    from_node=1,                   # Pickup location
    to_node=9,                     # Delivery location
    from_time=datetime(2024,1,1,12,0),  # Pickup time window start
    to_time=datetime(2024,1,1,15,0),    # Pickup time window end
    receptacle_type='TypeB',       # Container type
    quantity=0.45,                 # Number of containers
    capacity_required=0.675        # Total capacity (quantity × type capacity)
)
```

**Vehicle** - A delivery vehicle
```python
Vehicle(
    number='VA001',                # Vehicle ID
    vehicle_type=VehicleType(...), # Specifications
    restricted_nodes={1, 3}        # Cannot visit these nodes
)
```

**Service** - A work shift/route for one vehicle
```python
Service(
    service_id=1,
    vehicle=Vehicle(...),
    start_node=1,                  # Depot
    end_node=1,                    # Must return to depot
    tasks=[...],                   # Pickup/Delivery/Break tasks
    orders=[...],                  # Assigned orders
    start_time=datetime(...),
    end_time=datetime(...),
    total_distance=45.2,
    total_driving_time=67.5,
    break_assigned=True
)
```

**Task** - Individual operation in a service
```python
Task(
    task_type='PICKUP',            # PICKUP, DELIVERY, BREAK
    node=7,                        # Location
    order=Order(...),              # Associated order
    start_time=datetime(...),
    end_time=datetime(...),
    duration=10                    # Minutes
)
```

**Solution** - Complete routing plan
```python
Solution(
    services=[Service1, Service2, ...],  # All routes
    unserved_orders=[Order10, ...]       # Orders not assigned
)
```

---

### **Component 2: Problem Instance** (`cvrptw_parser.py`)

**ProblemInstance** - Loads and stores all problem data

```python
problem = ProblemInstance()
problem.load_from_files(file_paths)

# Access data:
problem.orders           # List of 12 orders
problem.vehicles         # List of 10 vehicles
problem.vehicle_types    # Dict of vehicle specs
problem.locations        # Dict of location info
problem.travel_time_matrix  # (from, to) → (distance, time)
problem.params           # Service parameters
```

---

### **Component 3: Constraints** (`constraint_validator.py`)

**ConstraintValidator** - Checks solution feasibility

Validates:
- ✅ **A. Service Constraints**
  - Service returns to origin
  - Max duration ≤ 456 min (7.6 hours)
  
- ✅ **B. Time Windows**
  - Pickup within [from_time, to_time]
  
- ✅ **C. Vehicle Constraints**
  - Capacity not exceeded at any point
  - Vehicle can visit all locations
  - Precedence: pickup before delivery
  
- ✅ **D. Break Constraints**
  - Exactly one 30-min break
  - Before 270 min driving or 330 min working
  - At break-allowed location

```python
validator = ConstraintValidator(problem)
is_feasible, violations = validator.validate_solution(solution)

# violations is a list of ConstraintViolation objects:
# - service_id
# - violation_type
# - description
# - severity ('HARD' or 'SOFT')
```

---

### **Component 4: Objective Function** (`constraint_validator.py`)

**ObjectiveCalculator** - Computes solution quality

```python
objective = ObjectiveCalculator(problem, weights={
    'num_services': 100.0,
    'num_vehicles': 150.0,
    'total_distance': 1.0,
    'total_idle_time': 0.5,
    'unserved_penalty': 1000.0
})

total_cost, components = objective.calculate(solution)
```

**Objective Components:**
- Minimize: Number of services (routes)
- Minimize: Number of vehicles used
- Minimize: Total distance traveled
- Minimize: Total idle time
- Maximize: Order coverage (minimize unserved)

---

### **Component 5: Greedy Constructor** (`test_phase0.py`)

**SimpleGreedyConstructor** - Builds initial solution

Algorithm:
1. Sort orders by earliest time window
2. For each vehicle:
   - Create new service
   - Greedily insert compatible orders
   - Add break
3. Return solution with assigned orders

```python
constructor = SimpleGreedyConstructor(problem)
solution = constructor.construct(start_depot=1)
```

---

## 🚀 How to Run Phase 0

### **Step 1: Verify Files**

Make sure all CSV files are in your working directory:
```bash
ls *.csv
# Should show: Receptacles.csv, VehicleTypes.csv, etc.
```

### **Step 2: Run the Test Script**

```bash
python test_phase0.py
```

### **Expected Output:**

```
Loading problem data...
✓ Data loaded successfully

🚀 Building Initial Solution from depot 1
============================================================

📦 Service 1 - Vehicle: VA001 (Van)
   ✓ Assigned 2 orders
   ✓ Duration: 185.3 min

📦 Service 2 - Vehicle: VA002 (Van)
   ✓ Assigned 1 orders
   ✓ Duration: 142.7 min

...

============================================================
CONSTRUCTION COMPLETE
  Services: 5
  Served: 10/12
  Coverage: 83.3%
============================================================

============================================================
SOLUTION METRICS
============================================================
Services Created: 5
Vehicles Used: 5
Total Distance: 234.56 km
Total Driving Time: 378.2 min
Unserved Orders: 2
Coverage: 83.3%

============================================================
SERVICE DETAILS
============================================================

Service 1:
  Vehicle: VA001 (Van)
  Orders: 2
  Distance: 45.23 km
  Duration: 185.3 min
  Route: 1 → P1 → D1 → P3 → D3 → 1

Service 2:
  Vehicle: VA002 (Van)
  Orders: 1
  Distance: 38.12 km
  Duration: 142.7 min
  Route: 1 → P5 → D5 → 1

...

⚠️  UNSERVED ORDERS: 2
  Order 4: 11→5, cap=3.29
  Order 8: 11→8, cap=1.48

============================================================
```

---

## 📊 Understanding the Output

### **Metrics Explained:**

**Services Created** - Number of routes/shifts needed
- Lower is better (efficiency)
- Each service has fixed cost + break time

**Vehicles Used** - Number of unique vehicles
- Should be ≤ Services
- Constraint: Can't exceed available fleet

**Total Distance** - Sum of all travel distances
- In kilometers
- Affects fuel cost and time

**Coverage** - Percentage of orders served
- Target: 100%
- Unserved orders indicate infeasibility or poor routing

**Route Notation:**
- `1` = Depot node
- `P1` = Pickup for Order 1
- `D1` = Delivery for Order 1

---

## 🔍 Key Data Insights (From Your Dataset)

**Orders:**
- Total: 12 orders
- Capacity demand: ~32.14 units
- Time windows: Average 180 min width
- Types: 8 TypeA (0.75 cap), 4 TypeB (1.5 cap)

**Fleet:**
- 10 vehicles total
- Types: Van (2.5 cap), TruckSmall (6.0), TruckLarge (12.0), Bike (0.5), MiniTruck (4.0)
- Total fleet capacity: ~44 units (sufficient)

**Constraints:**
- Multiple vehicle-location restrictions
- Break required every ~5.5 hours
- Only certain nodes allow breaks

**Challenges:**
- Node 2: Popular delivery (5 orders)
- Node 11: Popular pickup (4 orders)  
- Tight time windows require sequential routing
- Vehicle restrictions limit flexibility

---

## ✅ Phase 0 Checklist

Before moving to Phase 1, verify:

- [ ] All CSV files load without errors
- [ ] `test_phase0.py` runs successfully
- [ ] Solution is created (even if not 100% coverage)
- [ ] Metrics are displayed
- [ ] No Python import errors
- [ ] Understand the data structures
- [ ] Understand constraint validation
- [ ] Understand objective function

---

## 🐛 Common Issues & Fixes

### **Issue 1: File Not Found**
```
FileNotFoundError: [Errno 2] No such file or directory: 'Orders.csv'
```
**Fix:** Make sure all CSV files are in the same directory as the Python script.

### **Issue 2: Import Error**
```
ImportError: cannot import name 'ProblemInstance' from 'cvrptw_parser'
```
**Fix:** Make sure `cvrptw_parser.py` is in the same directory.

### **Issue 3: Low Coverage**
```
Coverage: 41.7% (only 5/12 orders served)
```
**Fix:** This is expected with simple greedy! Phase 1 (ALNS) will improve this.

### **Issue 4: All Orders Unserved**
```
Unserved Orders: 12/12
```
**Fix:** Check vehicle restrictions - some orders may be unreachable by available vehicles.

---

## 📈 What's Next: Phase 1 Preview

Once Phase 0 works, Phase 1 will:

1. **Improve solutions** using destroy/repair operators
2. **Escape local optima** with simulated annealing
3. **Adaptive learning** which operators work best
4. **Target: 90-100% coverage** with optimized routes

Phase 1 will use the same data structures, just add iterative improvement!

---

## 🎓 Key Concepts Review

### **Pickup-Delivery Problem (PDP)**
- Each order has TWO locations (pickup + delivery)
- Must pickup before delivery (precedence)
- Same vehicle handles both

### **Time Windows**
- Each order has [earliest, latest] pickup time
- Service must arrive within window
- Waiting is allowed (creates idle time)

### **Capacity Tracking**
- Load increases at pickup
- Load decreases at delivery
- Must check capacity at EVERY point in route

### **Service Constraints**
- Max 456 min (7.6 hours) including break
- Must start and end at depot
- One 30-min break before 4.5h driving

---

## 💡 Phase 0 Architecture Summary

```
┌─────────────────────────────────────────────────┐
│           PHASE 0: FOUNDATION                   │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐      ┌──────────────┐        │
│  │   CSV Files │─────▶│ ProblemInstance│        │
│  └─────────────┘      └──────┬───────┘        │
│                              │                  │
│                              ▼                  │
│                    ┌──────────────────┐        │
│                    │ SimpleGreedy     │        │
│                    │ Constructor      │        │
│                    └─────────┬────────┘        │
│                              │                  │
│                              ▼                  │
│                         ┌─────────┐            │
│                         │Solution │            │
│                         └────┬────┘            │
│                              │                  │
│              ┌───────────────┴────────────┐    │
│              ▼                            ▼    │
│     ┌─────────────────┐        ┌──────────────┐│
│     │ Constraint      │        │ Objective    ││
│     │ Validator       │        │ Calculator   ││
│     └─────────────────┘        └──────────────┘│
│              │                            │    │
│              ▼                            ▼    │
│         Feasibility?               Total Cost  │
│         Violations                  Metrics    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 📝 Next Steps

1. **Run** `test_phase0.py` and verify output
2. **Experiment** with different parameters
3. **Analyze** which orders are hard to serve
4. **Review** this documentation
5. **Ready for Phase 1!** 🚀