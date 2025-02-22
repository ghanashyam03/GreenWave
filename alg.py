import os
import traci
import sumolib
import numpy as np
import networkx as nx
from scipy.stats import entropy
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import sys
import itertools

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('traffic_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VehicleState:
    id: str
    type: str
    position: Tuple[float, float]
    speed: float
    route: List[str]
    current_edge: str
    destination: str
    reroute_attempts: int
    priority: float
    last_reroute_time: float
    waiting_time: float
    lane_position: float  # Added to track vehicle's position in lane
    acceleration: float   # Added to track vehicle dynamics

class TrafficMetrics:
    def __init__(self):
        self.volume: float = 0.0
        self.speed_variance: float = 0.0
        self.speed_entropy: float = 0.0
        self.density: float = 0.0
        self.avg_speed: float = 0.0
        self.congestion_index: float = 0.0
        self.queue_length: int = 0    # Number of vehicles queued
        self.flow_rate: float = 0.0   # Vehicles per hour
        self.occupancy: float = 0.0   # Percentage of time detector is occupied

class AdvancedTrafficManager:
    def __init__(self):
        self.sumo_config = {
            'gui': True,
            'config_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\test.sumocfg',
            'net_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\network.net.xml',
            'route_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\kochi.rou.xml',
        }
        
        # Concrete system parameters based on traffic engineering standards
        self.OPTIMIZATION_INTERVAL = 60  # Optimize every 60 simulation steps (1 minute at 1Hz)
        self.CONGESTION_THRESHOLDS = {
            'free_flow': 0.2,      # Less than 20% capacity utilized
            'moderate': 0.4,       # 20-40% capacity utilized
            'heavy': 0.7,          # 40-70% capacity utilized
            'severe': 0.85,        # 70-85% capacity utilized
            'gridlock': 0.95       # More than 85% capacity utilized
        }
        
        # Vehicle dynamics (all speeds in m/s)
        self.SPEED_LIMITS = {
            'urban': 13.89,        # 50 km/h
            'arterial': 16.67,     # 60 km/h
            'highway': 27.78,      # 100 km/h
            'residential': 8.33,   # 30 km/h
            'bus_lane': 13.89      # 50 km/h
        }
        
        # Passenger Car Unit (PCU) values based on standard traffic engineering
        self.PCU_VALUES = {
            'passenger': 1.0,
            'truck': 2.5,          # Standard truck
            'trailer': 3.5,        # Multi-unit truck
            'bus': 2.0,           # Standard bus
            'motorcycle': 0.5,
            'emergency': 1.0,
            'bicycle': 0.2
        }
        
        # Vehicle priority weights for routing
        self.PRIORITY_WEIGHTS = {
            'emergency': 10.0,     # Highest priority
            'bus': 3.0,           # Public transit priority
            'truck': 2.0,         # Commercial traffic
            'passenger': 1.0,     # Standard priority
            'motorcycle': 1.0,
            'bicycle': 1.0
        }
        
        # Traffic management parameters
        self.MAX_REROUTE_ATTEMPTS = 3
        self.MIN_REROUTE_INTERVAL = 300  # Minimum 5 minutes between reroutes
        self.CONGESTION_HISTORY_SIZE = 30  # 30 steps of history (30 seconds)
        self.ADAPTIVE_ROUTING_THRESHOLD = 0.7  # Start rerouting at 70% congestion
        
        # Signal timing parameters
        self.MIN_GREEN_TIME = 15   # Minimum green phase duration (seconds)
        self.MAX_GREEN_TIME = 90   # Maximum green phase duration (seconds)
        self.YELLOW_TIME = 3       # Yellow phase duration (seconds)
        self.ALL_RED_TIME = 2      # All-red clearance interval (seconds)
        
        # Initialize data structures
        self.network_graph = nx.DiGraph()
        self.traffic_metrics = defaultdict(TrafficMetrics)
        self.vehicle_states: Dict[str, VehicleState] = {}
        self.edge_congestion_history: Dict[str, List[float]] = defaultdict(list)
        self.emergency_routes: Set[str] = set()
        
        # Traffic signal states
        self.signal_states: Dict[str, Dict] = {}
        
        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize SUMO and create network graph with error handling."""
        try:
            # Verify file existence
            for key, path in self.sumo_config.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Required file not found: {path}")
            
            # Initialize SUMO with specific parameters
            sumo_binary = sumolib.checkBinary('sumo-gui' if self.sumo_config['gui'] else 'sumo')
            sumo_cmd = [
                sumo_binary,
                '-c', self.sumo_config['config_file'],
                '--net-file', self.sumo_config['net_file'],
                '--route-files', self.sumo_config['route_file'],
                '--time-to-teleport', '-1',
                '--waiting-time-memory', '10000',
                '--device.emissions.probability', '1.0',  # Enable emissions tracking
                '--device.rerouting.probability', '1.0',  # Enable rerouting devices
                '--device.rerouting.period', '30',        # Check for rerouting every 30s
                '--step-length', '1.0',                   # 1-second simulation steps
                '--collision.action', 'warn',             # Warn on collisions
                '--lateral-resolution', '0.1'             # Precise lateral movement
            ]
            
            traci.start(sumo_cmd)
            self.net = sumolib.net.readNet(self.sumo_config['net_file'])
            self._build_network_graph()
            self._initialize_traffic_signals()
            
            logger.info("Traffic Management System initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise

    def _build_network_graph(self):
        """Build NetworkX graph with detailed edge attributes."""
        try:
            for edge in self.net.getEdges():
                edge_id = edge.getID()
                from_node = edge.getFromNode().getID()
                to_node = edge.getToNode().getID()
                
                # Calculate realistic edge capacity using HCM formula
                num_lanes = len(edge.getLanes())
                lane_width = edge.getLanes()[0].getWidth()
                speed_limit = edge.getSpeed()
                
                # Highway Capacity Manual (HCM) based capacity calculation
                base_capacity = min(2200, 1900 + 20 * speed_limit) # vehicles per hour per lane
                capacity_adjustment = min(1.0, (lane_width - 3.0) * 0.1 + 1.0) # width adjustment
                
                theoretical_capacity = (
                    base_capacity * 
                    num_lanes * 
                    capacity_adjustment * 
                    0.95  # Peak hour factor
                )
                
                attrs = {
                    'length': edge.getLength(),
                    'speed_limit': speed_limit,
                    'lanes': num_lanes,
                    'lane_width': lane_width,
                    'capacity': theoretical_capacity,
                    'priority': edge.getPriority(),
                    'type': edge.getFunction(),
                    'grade': edge.getGrade() if hasattr(edge, 'getGrade') else 0.0
                }
                
                self.network_graph.add_edge(from_node, to_node, 
                                         edge_id=edge_id, **attrs)
            
            logger.info(f"Network graph built with {self.network_graph.number_of_nodes()} nodes and {self.network_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Network graph building failed: {str(e)}")
            raise

    def _initialize_traffic_signals(self):
        """Initialize traffic signal states and timing plans."""
        try:
            for tls_id in traci.trafficlight.getIDList():
                # Get signal programs
                programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
                
                # Store initial signal state
                self.signal_states[tls_id] = {
                    'current_phase': 0,
                    'phase_duration': 0,
                    'last_change': 0,
                    'programs': programs,
                    'controlled_lanes': traci.trafficlight.getControlledLanes(tls_id),
                    'controlled_links': traci.trafficlight.getControlledLinks(tls_id)
                }
                
            logger.info(f"Initialized {len(self.signal_states)} traffic signals")
            
        except Exception as e:
            logger.error(f"Traffic signal initialization failed: {str(e)}")
            raise

    def _update_vehicle_states(self):
        """Update vehicle states with comprehensive metrics."""
        try:
            current_time = traci.simulation.getTime()
            new_states = {}
            
            for vehicle_id in traci.vehicle.getIDList():
                try:
                    vehicle_type = traci.vehicle.getVehicleClass(vehicle_id)
                    current_route = traci.vehicle.getRoute(vehicle_id)
                    
                    # Get detailed vehicle metrics
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    acceleration = traci.vehicle.getAcceleration(vehicle_id)
                    lane_position = traci.vehicle.getLanePosition(vehicle_id)
                    
                    # Get or create vehicle state
                    state = self.vehicle_states.get(vehicle_id, None)
                    reroute_attempts = state.reroute_attempts if state else 0
                    last_reroute_time = state.last_reroute_time if state else 0
                    
                    new_states[vehicle_id] = VehicleState(
                        id=vehicle_id,
                        type=vehicle_type,
                        position=traci.vehicle.getPosition(vehicle_id),
                        speed=speed,
                        route=current_route,
                        current_edge=traci.vehicle.getRoadID(vehicle_id),
                        destination=current_route[-1],
                        reroute_attempts=reroute_attempts,
                        priority=self.PRIORITY_WEIGHTS.get(vehicle_type, 1.0),
                        last_reroute_time=last_reroute_time,
                        waiting_time=traci.vehicle.getWaitingTime(vehicle_id),
                        lane_position=lane_position,
                        acceleration=acceleration
                    )
                    
                    # Log significant changes in vehicle state
                    if state and (
                        abs(state.speed - speed) > 5.0 or  # Speed change > 5 m/s
                        abs(state.acceleration - acceleration) > 2.0  # Acceleration change > 2 m/s²
                    ):
                        logger.debug(f"Vehicle {vehicle_id} state change - Speed: {speed:.2f}, Acc: {acceleration:.2f}")
                    
                except traci.exceptions.TraCIException as e:
                    logger.warning(f"Failed to update state for vehicle {vehicle_id}: {str(e)}")
                    continue
            
            self.vehicle_states = new_states
            
        except Exception as e:
            logger.error(f"Vehicle state update failed: {str(e)}")
            raise

    def _compute_edge_metrics(self) -> Dict[str, TrafficMetrics]:
        try:
            edge_metrics = defaultdict(TrafficMetrics)

            # Collect raw data per edge
            edge_data = defaultdict(lambda: {
                'speeds': [],
                'volumes': 0.0,
                'queue': 0,
                'occupancy': 0.0
            })

            # First pass: collect raw data
            for vehicle in self.vehicle_states.values():
                edge = vehicle.current_edge
                if edge.startswith(':'): 
                    continue

                # Calculate PCU-adjusted volume
                pcu = self.PCU_VALUES.get(vehicle.type, 1.0)
                edge_data[edge]['speeds'].append(vehicle.speed)
                edge_data[edge]['volumes'] += pcu

                # Count queued vehicles (speed < 1 m/s)
                if vehicle.speed < 1.0:
                    edge_data[edge]['queue'] += 1

            # Second pass: compute metrics for each edge
            for edge in self.net.getEdges():
                edge_id = edge.getID()
                data = edge_data[edge_id]
                speeds = data['speeds']
                volume = data['volumes']

                metrics = TrafficMetrics()
                metrics.volume = volume

                if speeds:
                    metrics.avg_speed = np.mean(speeds)
                    metrics.speed_variance = np.var(speeds) if len(speeds) > 1 else 0
                    metrics.speed_entropy = entropy(speeds) if len(speeds) > 1 else 0

                # Compute HCM-based metrics
                edge_length = edge.getLength()
                num_lanes = len(edge.getLanes())

                # Density (vehicles/km/lane)
                metrics.density = (volume / (edge_length / 1000) / num_lanes) if edge_length > 0 else 0

                # Flow rate (vehicles/hour)
                metrics.flow_rate = volume * 3600  # Convert from vehicles/step to vehicles/hour

                # Queue length
                metrics.queue_length = edge_data[edge_id]['queue']

                # Occupancy (percentage of time detector is occupied)
                metrics.occupancy = min(100.0, (metrics.density / 120) * 100)  # Assuming jam density of 120 veh/km

                # Congestion index using HCM level of service thresholds
                capacity = self.network_graph[edge.getFromNode().getID()][edge.getToNode().getID()]['capacity']
                metrics.congestion_index = min(1.0, metrics.flow_rate / capacity if capacity > 0 else 1.0)

                edge_metrics[edge_id] = metrics

                # Update historical congestion data
                self.edge_congestion_history[edge_id].append(metrics.congestion_index)
                if len(self.edge_congestion_history[edge_id]) > self.CONGESTION_HISTORY_SIZE:
                    self.edge_congestion_history[edge_id].pop(0)

            return edge_metrics
        except Exception as e:
            logger.error(f"Error computing edge metrics: {e}")
            return {}
            
    def _calculate_route_cost(self, route: List[str], vehicle_type: str) -> float:
        """Calculate route cost using HCM-based impedance function."""
        try:
            total_cost = 0.0
            
            for i in range(len(route) - 1):
                edge_id = route[i]
                if edge_id.startswith(':'):
                    continue
                
                metrics = self.traffic_metrics[edge_id]
                edge = self.net.getEdge(edge_id)
                
                # Base travel time (length/speed)
                free_flow_time = edge.getLength() / edge.getSpeed()
                
                # BPR (Bureau of Public Roads) function parameters
                alpha = 0.15  # Standard BPR parameter
                beta = 4.0    # Standard BPR parameter
                
                # Calculate congestion factor using BPR function
                congestion_factor = 1.0 + alpha * (metrics.congestion_index ** beta)
                
                # Calculate impedance components
                travel_time_cost = free_flow_time * congestion_factor
                queue_delay = (metrics.queue_length * 2.0) if metrics.queue_length > 0 else 0
                
                # Vehicle type specific adjustments
                type_factor = 1.0
                if vehicle_type == 'emergency':
                    type_factor = 0.5  # Emergency vehicles prioritized
                elif vehicle_type == 'truck':
                    type_factor = 1.2  # Trucks avoid congested routes more
                
                # Historical congestion penalty
                hist_congestion = np.mean(self.edge_congestion_history[edge_id][-5:]) if self.edge_congestion_history[edge_id] else 0
                congestion_penalty = hist_congestion * free_flow_time * 0.5
                
                # Combined edge cost
                edge_cost = (
                    travel_time_cost * 0.4 +
                    queue_delay * 0.3 +
                    congestion_penalty * 0.3
                ) * type_factor
                
                total_cost += edge_cost
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Route cost calculation failed: {str(e)}")
            raise

    def _find_optimal_route(self, vehicle_id: str) -> Optional[List[str]]:
        """Find optimal route using modified K-shortest paths with traffic conditions."""
        try:
            vehicle = self.vehicle_states.get(vehicle_id)
            if not vehicle:
                return None
            
            origin_edge = vehicle.current_edge
            destination_edge = vehicle.destination
            
            if origin_edge.startswith(':') or destination_edge.startswith(':'):
                return None
            
            # Get origin and destination nodes
            origin_node = self.net.getEdge(origin_edge).getToNode().getID()
            dest_node = self.net.getEdge(destination_edge).getFromNode().getID()
            
            # Calculate k based on network size and congestion
            network_size = self.network_graph.number_of_nodes()
            avg_congestion = np.mean([m.congestion_index for m in self.traffic_metrics.values()])
            k = min(5, max(2, int(network_size * avg_congestion * 0.01)))
            
            try:
                paths = list(itertools.islice(
                    nx.shortest_simple_paths(
                        self.network_graph,
                        origin_node,
                        dest_node,
                        weight='length'
                    ),
                    k
                ))
            except nx.NetworkXNoPath:
                logger.warning(f"No path found for vehicle {vehicle_id}")
                return None
            
            # Evaluate each path considering real-time conditions
            best_route = None
            best_cost = float('inf')
            
            for path in paths:
                edge_path = []
                valid_path = True
                
                for i in range(len(path) - 1):
                    edges = self.network_graph.get_edge_data(path[i], path[i+1])
                    if edges:
                        edge_id = edges['edge_id']
                        # Check if edge is severely congested
                        if self.traffic_metrics[edge_id].congestion_index > self.CONGESTION_THRESHOLDS['gridlock']:
                            valid_path = False
                            break
                        edge_path.append(edge_id)
                    else:
                        valid_path = False
                        break
                
                if valid_path and edge_path:
                    route_cost = self._calculate_route_cost(edge_path, vehicle.type)
                    if route_cost < best_cost:
                        best_cost = route_cost
                        best_route = edge_path
            
            return best_route
            
        except Exception as e:
            logger.error(f"Optimal route finding failed for vehicle {vehicle_id}: {str(e)}")
            return None

    def _apply_speed_control(self, edge_id: str, congestion_level: float):
        """Apply sophisticated speed control using car-following models."""
        try:
            edge = self.net.getEdge(edge_id)
            metrics = self.traffic_metrics[edge_id]
            
            # Base parameters
            max_speed = edge.getSpeed()
            min_speed = max_speed * 0.3  # Minimum 30% of speed limit
            
            # Calculate optimal speed based on traffic conditions
            density_factor = max(0.0, 1.0 - (metrics.density / 120.0))  # Assuming jam density of 120 veh/km
            queue_factor = max(0.0, 1.0 - (metrics.queue_length / (edge.getLength() / 7.5)))  # Assuming 7.5m per vehicle
            
            # Intelligent Speed Adaptation (ISA) calculation
            base_speed_factor = min(
                1.0,
                density_factor * 0.6 +
                queue_factor * 0.4
            )
            
            # Apply speed controls to vehicles
            for vehicle_id, state in self.vehicle_states.items():
                if state.current_edge != edge_id:
                    continue
                
                # Get leading vehicle data
                leader_data = traci.vehicle.getLeader(vehicle_id, 100)  # Look ahead 100m
                
                if leader_data:
                    leader_id, gap = leader_data
                    # Intelligent Driver Model (IDM) parameters
                    desired_gap = max(2.0, state.speed * 1.5)  # Dynamic safe gap
                    
                    # Adjust speed based on gap
                    gap_factor = min(1.0, gap / desired_gap)
                else:
                    gap_factor = 1.0
                
                # Calculate final speed
                if state.type == 'emergency':
                    target_speed = max_speed  # Emergency vehicles maintain max speed
                else:
                    target_speed = max(
                        min_speed,
                        max_speed * base_speed_factor * gap_factor
                    )
                
                # Apply smooth speed transition
                current_speed = state.speed
                max_acceleration = 2.5  # m/s²
                max_deceleration = -4.5  # m/s²
                
                speed_diff = target_speed - current_speed
                if abs(speed_diff) > 0.1:  # Only adjust if difference is significant
                    if speed_diff > 0:
                        new_speed = current_speed + min(speed_diff, max_acceleration)
                    else:
                        new_speed = current_speed + max(speed_diff, max_deceleration)
                    
                    traci.vehicle.setSpeed(vehicle_id, new_speed)
            
        except Exception as e:
            logger.error(f"Speed control application failed for edge {edge_id}: {str(e)}")

    def _manage_intersection_timing(self):
        """Adaptive signal control using real-time traffic demand."""
        try:
            current_time = traci.simulation.getTime()
            
            for tls_id in traci.trafficlight.getIDList():
                signal_data = self.signal_states[tls_id]
                controlled_lanes = signal_data['controlled_lanes']
                
                # Calculate demand for each approach
                approach_demand = defaultdict(float)
                queue_lengths = defaultdict(int)
                
                for lane_id in controlled_lanes:
                    edge_id = lane_id.split('_')[0]
                    if edge_id in self.traffic_metrics:
                        metrics = self.traffic_metrics[edge_id]
                        # Combined demand metric
                        demand = (
                            metrics.flow_rate * 0.4 +
                            metrics.queue_length * 20 * 0.4 +  # Weight queue length
                            metrics.waiting_time * 0.2
                        )
                        approach_demand[edge_id] = demand
                        queue_lengths[edge_id] = metrics.queue_length
                
                if not approach_demand:
                    continue
                
                current_phase = traci.trafficlight.getPhase(tls_id)
                current_program = signal_data['programs'][0]  # Using first program
                phase_duration = current_program.phases[current_phase].duration
                
                # Calculate optimal green time
                max_demand = max(approach_demand.values())
                max_queue = max(queue_lengths.values())
                
                if max_demand > 0:
                    # Webster's method for optimal green time
                    cycle_time = min(
                        120,  # Maximum cycle length
                        max(
                            60,  # Minimum cycle length
                            1.5 * (self.YELLOW_TIME + self.ALL_RED_TIME) / 
                            (1 - sum(demand/max_demand for demand in approach_demand.values()))
                        )
                    )
                    
                    # Adjust green time based on demand and queues
                    if max_queue > 10 or max_demand > 0.8:
                        new_duration = min(
                            self.MAX_GREEN_TIME,
                            max(
                                self.MIN_GREEN_TIME,
                                phase_duration + 5
                            )
                        )
                        traci.trafficlight.setPhaseDuration(tls_id, new_duration)
                    
                    logger.debug(f"Signal {tls_id} adjusted - Phase: {current_phase}, Duration: {new_duration}")
            
        except Exception as e:
            logger.error(f"Intersection timing management failed: {str(e)}")

    def _manage_emergency_vehicles(self):
        """Priority management for emergency vehicles with preemption."""
        try:
            emergency_vehicles = {
                vid: state for vid, state in self.vehicle_states.items()
                if state.type == 'emergency'
            }
            
            if not emergency_vehicles:
                self.emergency_routes.clear()
                return
            
            for vehicle_id, state in emergency_vehicles.items():
                current_edge = state.current_edge
                if current_edge.startswith(':'):
                    continue
                
                # Clear path ahead
                route = state.route
                next_edges = route[route.index(current_edge):]
                self.emergency_routes.update(next_edges)
                
                # Find upcoming intersections
                for edge_id in next_edges[:3]:  # Look at next 3 edges
                    edge = self.net.getEdge(edge_id)
                    to_node = edge.getToNode()
                    
                    if to_node.getType() == 'traffic_light':
                        tls_id = to_node.getID()
                        
                        # Calculate time to intersection
                        distance_to_intersection = (
                            edge.getLength() - state.lane_position
                            if edge_id == current_edge
                            else edge.getLength()
                        )
                        time_to_intersection = distance_to_intersection / max(state.speed, 5.0)
                        
                        if time_to_intersection < 20.0:  # Within 20 seconds
                            # Determine optimal signal phase for emergency vehicle
                            approach_lane = edge.getLanes()[0].getID()
                            for phase_index, phase in enumerate(self.signal_states[tls_id]['programs'][0].phases):
                                if approach_lane in traci.trafficlight.getControlledLanes(tls_id):
                                    # Set green phase for emergency vehicle
                                    traci.trafficlight.setPhase(tls_id, phase_index)
                                    traci.trafficlight.setPhaseDuration(tls_id, 20)  # Extended green
                                    logger.info(f"Emergency preemption activated at {tls_id} for vehicle {vehicle_id}")
                                    break
                
                # Clear path for emergency vehicle
                for veh_id, veh_state in self.vehicle_states.items():
                    if veh_id != vehicle_id and veh_state.current_edge in next_edges:
                        try:
                            # Request lane change if possible
                            if traci.vehicle.couldChangeLane(veh_id, 1):
                                traci.vehicle.changeLane(veh_id, 1, 5)
                            elif traci.vehicle.couldChangeLane(veh_id, -1):
                                traci.vehicle.changeLane(veh_id, -1, 5)
                        except traci.exceptions.TraCIException:
                            continue
            
        except Exception as e:
            logger.error(f"Emergency vehicle management failed: {str(e)}")

    def _update_vehicle_routes(self):
        """Update vehicle routes based on current traffic conditions."""
        try:
            for vehicle_id, state in self.vehicle_states.items():
                if state.reroute_attempts >= self.MAX_REROUTE_ATTEMPTS:
                    continue
                
                # Check if reroute is needed based on congestion
                current_edge = state.current_edge
                if self.traffic_metrics[current_edge].congestion_index > self.ADAPTIVE_ROUTING_THRESHOLD:
                    optimal_route = self._find_optimal_route(vehicle_id)
                    if optimal_route:
                        try:
                            traci.vehicle.setRoute(vehicle_id, optimal_route)
                            state.reroute_attempts += 1
                            state.last_reroute_time = traci.simulation.getTime()
                            logger.info(f"Vehicle {vehicle_id} rerouted to avoid congestion")
                        except traci.exceptions.TraCIException as e:
                            logger.error(f"Route replacement failed for vehicle {vehicle_id}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Vehicle route update failed: {str(e)}")
            raise

    def run_traffic_management(self):
        """Main traffic management loop with comprehensive monitoring."""
        try:
            step = 0
            last_stats_time = 0
            stats_interval = 300  # Log statistics every 5 minutes
            
            while traci.simulation.getMinExpectedNumber() > 0:
                # Basic simulation step
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                
                # Regular updates
                self._update_vehicle_states()
                self.traffic_metrics = self._compute_edge_metrics()
                
                # Emergency vehicle management (high priority)
                self._manage_emergency_vehicles()
                
                # Periodic optimization
                if step % self.OPTIMIZATION_INTERVAL == 0:
                    self._manage_intersection_timing()
                    self._update_vehicle_routes()
                
                # Continuous speed control
                for edge_id, metrics in self.traffic_metrics.items():
                    if metrics.congestion_index > self.CONGESTION_THRESHOLDS['moderate']:
                        self._apply_speed_control(edge_id, metrics.congestion_index)
                
                # Periodic statistics logging
                if current_time - last_stats_time >= stats_interval:
                    stats = self._collect_statistics()
                    logger.info(f"System Statistics at {current_time}s: {stats}")
                    last_stats_time = current_time
                
                step += 1
                
            logger.info("Simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Traffic management loop failed: {str(e)}")
            raise
        finally:
            try:
                traci.close()
            except:
                pass

    def _collect_statistics(self) -> Dict:
        """Collect comprehensive system statistics."""
        try:
            current_time = traci.simulation.getTime()
            
            # Initialize statistics structure
            stats = {
                'timestamp': datetime.now().isoformat(),
                'simulation_time': current_time,
                'vehicle_statistics': {
                    'total_active': len(self.vehicle_states),
                    'by_type': defaultdict(int),
                    'avg_speed': 0.0,
                    'avg_waiting_time': 0.0,
                    'total_distance_traveled': 0.0,
                    'total_co2_emissions': 0.0
                },
                'network_statistics': {
                    'total_congestion_index': 0.0,
                    'avg_network_speed': 0.0,
                    'total_queue_length': 0,
                    'bottleneck_locations': [],
                    'critical_edges': []
                },
                'emergency_response': {
                    'active_emergency_vehicles': 0,
                    'avg_emergency_speed': 0.0,
                    'emergency_response_coverage': 0.0
                },
                'traffic_control': {
                    'signals_optimized': 0,
                    'rerouted_vehicles': 0,
                    'speed_controls_active': 0
                }
            }
            
            # Vehicle statistics
            vehicle_speeds = []
            emergency_speeds = []
            total_distance = 0.0
            total_co2 = 0.0
            
            for vehicle_id, state in self.vehicle_states.items():
                # Update vehicle type counts
                stats['vehicle_statistics']['by_type'][state.type] += 1
                
                # Collect speed data
                vehicle_speeds.append(state.speed)
                if state.type == 'emergency':
                    emergency_speeds.append(state.speed)
                    stats['emergency_response']['active_emergency_vehicles'] += 1
                
                # Calculate emissions and distance
                try:
                    total_distance += traci.vehicle.getDistance(vehicle_id)
                    total_co2 += traci.vehicle.getCO2Emission(vehicle_id)
                except traci.exceptions.TraCIException:
                    continue
            
            # Update vehicle statistics
            if vehicle_speeds:
                stats['vehicle_statistics']['avg_speed'] = np.mean(vehicle_speeds)
                stats['vehicle_statistics']['total_distance_traveled'] = total_distance
                stats['vehicle_statistics']['total_co2_emissions'] = total_co2
            
            if emergency_speeds:
                stats['emergency_response']['avg_emergency_speed'] = np.mean(emergency_speeds)
            
            # Network statistics
            edge_congestion = []
            edge_speeds = []
            total_queue = 0
            
            for edge_id, metrics in self.traffic_metrics.items():
                edge_congestion.append(metrics.congestion_index)
                edge_speeds.append(metrics.avg_speed)
                total_queue += metrics.queue_length
                
                # Identify bottlenecks (congestion > 0.8)
                if metrics.congestion_index > 0.8:
                    stats['network_statistics']['bottleneck_locations'].append({
                        'edge_id': edge_id,
                        'congestion_index': metrics.congestion_index,
                        'queue_length': metrics.queue_length
                    })
                
                # Identify critical edges (high flow + high congestion)
                if metrics.flow_rate > 1000 and metrics.congestion_index > 0.7:
                    stats['network_statistics']['critical_edges'].append({
                        'edge_id': edge_id,
                        'flow_rate': metrics.flow_rate,
                        'congestion_index': metrics.congestion_index
                    })
            
            if edge_congestion:
                stats['network_statistics']['total_congestion_index'] = np.mean(edge_congestion)
            if edge_speeds:
                stats['network_statistics']['avg_network_speed'] = np.mean(edge_speeds)
            stats['network_statistics']['total_queue_length'] = total_queue
            
            # Traffic control statistics
            stats['traffic_control']['signals_optimized'] = len(self.signal_states)
            stats['traffic_control']['rerouted_vehicles'] = sum(
                1 for v in self.vehicle_states.values() if v.reroute_attempts > 0
            )
            stats['traffic_control']['speed_controls_active'] = sum(
                1 for m in self.traffic_metrics.values() if m.congestion_index > self.CONGESTION_THRESHOLDS['moderate']
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics collection failed: {str(e)}")
            return {}

def main():
    """Main entry point with error handling and logging."""
    logger.info("Starting Traffic Management System")
    
    try:
        # Initialize and run traffic manager
        traffic_manager = AdvancedTrafficManager()
        traffic_manager.run_traffic_management()
        
    except FileNotFoundError as e:
        logger.critical(f"Required file not found: {str(e)}")
        sys.exit(1)
    except traci.exceptions.TraCIException as e:
        logger.critical(f"SUMO TraCI error: {str(e)}")
        sys.exit(2)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        sys.exit(3)
    finally:
        # Ensure proper cleanup
        if traci.isLoaded():
            traci.close()
        logger.info("Traffic Management System shutdown complete")

if __name__ == '__main__':
    main()
