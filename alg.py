import os
import traci
import sumolib
import numpy as np
import networkx as nx
from scipy.stats import entropy
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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

class TrafficMetrics:
    def __init__(self):
        self.volume: float = 0
        self.speed_variance: float = 0
        self.speed_entropy: float = 0
        self.density: float = 0
        self.avg_speed: float = 0
        self.congestion_index: float = 0

class AdvancedTrafficManager:
    def __init__(self):
        self.sumo_config = {
            'gui': True,
            'config_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\test.sumocfg',
            'net_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\network.net.xml',
            'route_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\kochi.rou.xml',
            
        }
        
        # System parameters
        self.OPTIMIZATION_INTERVAL = 30  # Steps between optimizations
        self.CONGESTION_THRESHOLDS = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
        
        # Vehicle dynamics
        self.SPEED_LIMITS = {
            'urban': 13.89,  # 50 km/h in m/s
            'highway': 27.78,  # 100 km/h in m/s
            'residential': 8.33  # 30 km/h in m/s
        }
        
        self.PCU_VALUES = {
            'passenger': 1.0,
            'truck': 3.0,
            'trailer': 3.5,
            'bus': 2.5,
            'motorcycle': 0.5,
            'emergency': 1.0,  # Lower PCU but highest priority
            'bicycle': 0.2
        }
        
        self.PRIORITY_WEIGHTS = {
            'emergency': 10.0,
            'bus': 5.0,
            'passenger': 1.0,
            'truck': 2.0,
            'motorcycle': 1.0,
            'bicycle': 1.0
        }
        
        # Traffic management parameters
        self.MAX_REROUTE_ATTEMPTS = 3
        self.MIN_REROUTE_INTERVAL = 300  # Minimum time between reroutes
        self.CONGESTION_HISTORY_SIZE = 20
        self.ADAPTIVE_ROUTING_THRESHOLD = 0.6
        
        # Initialize data structures
        self.network_graph = nx.DiGraph()
        self.traffic_metrics = defaultdict(TrafficMetrics)
        self.vehicle_states: Dict[str, VehicleState] = {}
        self.edge_congestion_history: Dict[str, List[float]] = defaultdict(list)
        self.emergency_routes: Set[str] = set()
        
        # Load network
        self._initialize_system()

    def _initialize_system(self):
        """Initialize SUMO and create network graph."""
        try:
            # Verify file existence
            for key, path in self.sumo_config.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Required file not found: {path}")
            
            # Initialize SUMO
            sumo_binary = sumolib.checkBinary('sumo-gui' if self.sumo_config['gui'] else 'sumo')
            sumo_cmd = [
                sumo_binary,
                '-c', self.sumo_config['config_file'],
                '--net-file', self.sumo_config['net_file'],
                '--route-files', self.sumo_config['route_file'],
                
                '--time-to-teleport', '-1',
                '--waiting-time-memory', '10000',
                '--random'
            ]
            
            traci.start(sumo_cmd)
            self.net = sumolib.net.readNet(self.sumo_config['net_file'])
            self._build_network_graph()
            
            logger.info("Traffic Management System initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise

    def _build_network_graph(self):
        """Build NetworkX graph from SUMO network."""
        try:
            for edge in self.net.getEdges():
                edge_id = edge.getID()
                from_node = edge.getFromNode().getID()
                to_node = edge.getToNode().getID()
                
                # Edge attributes
                attrs = {
                    'length': edge.getLength(),
                    'speed': edge.getSpeed(),
                    'lanes': len(edge.getLanes()),
                    'priority': edge.getPriority(),
                    'capacity': edge.getSpeed() * len(edge.getLanes()) * 0.8,  # Theoretical capacity
                    'type': edge.getFunction()
                }
                
                self.network_graph.add_edge(from_node, to_node, 
                                         edge_id=edge_id, **attrs)
                
            logger.info(f"Network graph built with {self.network_graph.number_of_nodes()} nodes and {self.network_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Network graph building failed: {str(e)}")
            raise

    def _update_vehicle_states(self):
        """Update vehicle states with current simulation data."""
        try:
            current_time = traci.simulation.getTime()
            new_states = {}
            
            for vehicle_id in traci.vehicle.getIDList():
                try:
                    vehicle_type = traci.vehicle.getVehicleClass(vehicle_id)
                    current_route = traci.vehicle.getRoute(vehicle_id)
                    
                    # Get or create vehicle state
                    state = self.vehicle_states.get(vehicle_id, None)
                    reroute_attempts = state.reroute_attempts if state else 0
                    last_reroute_time = state.last_reroute_time if state else 0
                    
                    new_states[vehicle_id] = VehicleState(
                        id=vehicle_id,
                        type=vehicle_type,
                        position=traci.vehicle.getPosition(vehicle_id),
                        speed=traci.vehicle.getSpeed(vehicle_id),
                        route=current_route,
                        current_edge=traci.vehicle.getRoadID(vehicle_id),
                        destination=current_route[-1],
                        reroute_attempts=reroute_attempts,
                        priority=self.PRIORITY_WEIGHTS.get(vehicle_type, 1.0),
                        last_reroute_time=last_reroute_time,
                        waiting_time=traci.vehicle.getWaitingTime(vehicle_id)
                    )
                    
                except traci.exceptions.TraCIException as e:
                    logger.warning(f"Failed to update state for vehicle {vehicle_id}: {str(e)}")
                    continue
            
            self.vehicle_states = new_states
            
        except Exception as e:
            logger.error(f"Vehicle state update failed: {str(e)}")
            raise

    def _compute_edge_metrics(self) -> Dict[str, TrafficMetrics]:
        """Compute comprehensive traffic metrics for each edge."""
        try:
            edge_metrics = defaultdict(TrafficMetrics)
            edge_speeds = defaultdict(list)
            edge_volumes = defaultdict(float)
            
            # Collect raw data
            for vehicle in self.vehicle_states.values():
                edge = vehicle.current_edge
                if edge.startswith(':'): continue  # Skip internal edges
                
                pcu = self.PCU_VALUES.get(vehicle.type, 1.0)
                edge_speeds[edge].append(vehicle.speed)
                edge_volumes[edge] += pcu
            
            # Compute metrics for each edge
            for edge in self.net.getEdges():
                edge_id = edge.getID()
                speeds = edge_speeds.get(edge_id, [])
                volume = edge_volumes.get(edge_id, 0.0)
                
                metrics = TrafficMetrics()
                metrics.volume = volume
                
                if speeds:
                    metrics.avg_speed = np.mean(speeds)
                    metrics.speed_variance = np.var(speeds) if len(speeds) > 1 else 0
                    metrics.speed_entropy = entropy(speeds) if len(speeds) > 1 else 0
                
                # Compute density and congestion index
                edge_length = edge.getLength()
                edge_lanes = len(edge.getLanes())
                capacity = edge_length * edge_lanes * 0.8  # Theoretical capacity
                
                metrics.density = volume / (edge_length * edge_lanes) if edge_length > 0 else 0
                metrics.congestion_index = volume / capacity if capacity > 0 else 0
                
                edge_metrics[edge_id] = metrics
                
                # Update historical data
                self.edge_congestion_history[edge_id].append(metrics.congestion_index)
                if len(self.edge_congestion_history[edge_id]) > self.CONGESTION_HISTORY_SIZE:
                    self.edge_congestion_history[edge_id].pop(0)
            
            return edge_metrics
            
        except Exception as e:
            logger.error(f"Edge metrics computation failed: {str(e)}")
            raise

    def _calculate_route_cost(self, route: List[str], vehicle_type: str) -> float:
        """Calculate weighted route cost considering multiple factors."""
        try:
            total_cost = 0.0
            
            for i in range(len(route) - 1):
                edge_id = route[i]
                if edge_id.startswith(':'):  # Skip internal edges
                    continue
                
                metrics = self.traffic_metrics[edge_id]
                edge = self.net.getEdge(edge_id)
                
                # Base cost factors
                length_cost = edge.getLength()
                congestion_cost = metrics.congestion_index * 2.0
                speed_cost = (edge.getSpeed() / self.SPEED_LIMITS['urban'])
                
                # Historical congestion factor
                hist_congestion = np.mean(self.edge_congestion_history[edge_id][-5:]) if self.edge_congestion_history[edge_id] else 0
                
                # Priority factor for emergency vehicles
                priority_factor = 0.5 if vehicle_type == 'emergency' and edge_id in self.emergency_routes else 1.0
                
                # Combined cost with weights
                edge_cost = (
                    length_cost * 0.3 +
                    congestion_cost * 0.4 +
                    speed_cost * 0.2 +
                    hist_congestion * 0.1
                ) * priority_factor
                
                total_cost += edge_cost
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Route cost calculation failed: {str(e)}")
            raise

    def _find_optimal_route(self, vehicle_id: str) -> Optional[List[str]]:

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
            
            # Find K shortest paths
            try:
                paths_generator = nx.shortest_simple_paths(
                    self.network_graph,
                    origin_node,
                    dest_node,
                    weight='length'
                )
                paths = list(itertools.islice(paths_generator, 3))  # Get only top 3 paths
            except nx.NetworkXNoPath:
                logger.warning(f"No path found for vehicle {vehicle_id}")
                return None
            
            # Convert node paths to edge paths and evaluate costs
            best_route = None
            best_cost = float('inf')
            
            for path in paths:
                edge_path = []
                for i in range(len(path) - 1):
                    edges = self.network_graph.get_edge_data(path[i], path[i+1])
                    if edges:
                        edge_path.append(edges.get('edge_id'))
                    else:
                        logger.warning(f"No edge data found between {path[i]} and {path[i+1]}")
                        continue
                
                if edge_path:
                    route_cost = self._calculate_route_cost(edge_path, vehicle.type)
                    if route_cost < best_cost:
                        best_cost = route_cost
                        best_route = edge_path
            
            return best_route if best_route else None
            
        except Exception as e:
            logger.error(f"Optimal route finding failed for vehicle {vehicle_id}: {str(e)}")
            return None


    def _apply_speed_control(self, edge_id: str, congestion_level: float):
        """Apply sophisticated speed control based on traffic conditions."""
        try:
            edge = self.net.getEdge(edge_id)
            metrics = self.traffic_metrics[edge_id]
            
            # Base speed calculation
            max_speed = edge.getSpeed()
            min_speed = max_speed * 0.4  # Never reduce speed below 40% of limit
            
            # Adaptive speed factor based on multiple conditions
            congestion_factor = 1 - (congestion_level ** 2)  # Non-linear reduction
            density_factor = 1 - (metrics.density * 0.5)
            variance_factor = 1 - (min(metrics.speed_variance, 100) / 100)
            
            # Combined speed factor
            speed_factor = min(
                1.0,
                congestion_factor * 0.5 +
                density_factor * 0.3 +
                variance_factor * 0.2
            )
            
            # Calculate target speed
            target_speed = max(min_speed, max_speed * speed_factor)
            
            # Apply speed control to vehicles on this edge
            for vehicle_id, state in self.vehicle_states.items():
                if state.current_edge == edge_id:
                    if state.type == 'emergency':
                        # Emergency vehicles maintain higher speeds
                        traci.vehicle.setSpeed(vehicle_id, max_speed)
                    else:
                        traci.vehicle.setSpeed(vehicle_id, target_speed)
            
        except Exception as e:
            logger.error(f"Speed control application failed for edge {edge_id}: {str(e)}")

    def _manage_emergency_vehicles(self):
        """Priority management for emergency vehicles."""
        try:
            # Identify active emergency vehicles
            emergency_vehicles = {
                vid: state for vid, state in self.vehicle_states.items()
                if state.type == 'emergency'
            }
            
            if not emergency_vehicles:
                self.emergency_routes.clear()
                return
            
            # Update emergency routes
            new_emergency_routes = set()
            for vehicle_id, state in emergency_vehicles.items():
                # Clear path ahead
                current_route = state.route
                new_emergency_routes.update(current_route)
                
                # Adjust signals for emergency vehicles
                current_edge = state.current_edge
                if not current_edge.startswith(':'):
                    next_tls = self._get_next_traffic_light(current_edge)
                    if next_tls:
                        traci.trafficlight.setPhase(next_tls, 0)  # Green phase
                        traci.trafficlight.setPhaseDuration(next_tls, 30)  # Extended green
            
            self.emergency_routes = new_emergency_routes
            
        except Exception as e:
            logger.error(f"Emergency vehicle management failed: {str(e)}")

    def _get_next_traffic_light(self, edge_id: str) -> Optional[str]:
        """Get the next traffic light after the given edge."""
        try:
            edge = self.net.getEdge(edge_id)
            to_node = edge.getToNode()
            
            # Check if the node has a traffic light
            if to_node.getType() == 'traffic_light':
                return to_node.getID()
            
            return None
            
        except Exception as e:
            logger.error(f"Traffic light lookup failed for edge {edge_id}: {str(e)}")
            return None

    def _manage_intersection_timing(self):
        """Adaptive traffic signal control based on real-time conditions."""
        try:
            for tls_id in traci.trafficlight.getIDList():
                # Get incoming lanes for this traffic light
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                
                # Calculate demand for each approach
                approach_demand = defaultdict(float)
                
                for lane_id in controlled_lanes:
                    edge_id = lane_id.split('_')[0]
                    if edge_id in self.traffic_metrics:
                        metrics = self.traffic_metrics[edge_id]
                        approach_demand[edge_id] = metrics.congestion_index
                
                if not approach_demand:
                    continue
                
                # Adjust signal timing based on demand
                max_demand = max(approach_demand.values())
                if max_demand > self.CONGESTION_THRESHOLDS['high']:
                    # Extend green time for heavily congested approaches
                    current_phase = traci.trafficlight.getPhase(tls_id)
                    traci.trafficlight.setPhaseDuration(tls_id, 45)  # Extended phase
                
        except Exception as e:
            logger.error(f"Intersection timing management failed: {str(e)}")

    def _update_vehicle_routes(self):
        """Strategic route updates for vehicles based on current conditions."""
        try:
            # Sort vehicles by priority and congestion level
            vehicles_to_reroute = []
            current_time = traci.simulation.getTime()
            
            for vehicle_id, state in self.vehicle_states.items():
                if state.current_edge.startswith(':'):
                    continue
                
                # Check if vehicle needs rerouting
                current_congestion = self.traffic_metrics[state.current_edge].congestion_index
                time_since_last_reroute = current_time - state.last_reroute_time
                
                if (current_congestion > self.ADAPTIVE_ROUTING_THRESHOLD and 
                    time_since_last_reroute > self.MIN_REROUTE_INTERVAL and
                    state.reroute_attempts < self.MAX_REROUTE_ATTEMPTS):
                    
                    vehicles_to_reroute.append((
                        vehicle_id,
                        state.priority * (1 + current_congestion),
                        state
                    ))
            
            # Sort by priority score
            vehicles_to_reroute.sort(key=lambda x: x[1], reverse=True)
            
            # Apply rerouting
            for vehicle_id, _, state in vehicles_to_reroute:
                new_route = self._find_optimal_route(vehicle_id)
                
                if new_route and new_route != state.route:
                    try:
                        traci.vehicle.setRoute(vehicle_id, new_route)
                        self.vehicle_states[vehicle_id].route = new_route
                        self.vehicle_states[vehicle_id].reroute_attempts += 1
                        self.vehicle_states[vehicle_id].last_reroute_time = current_time
                        
                        logger.info(f"Rerouted vehicle {vehicle_id} successfully")
                        
                    except traci.exceptions.TraCIException as e:
                        logger.warning(f"Failed to reroute vehicle {vehicle_id}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Vehicle route updates failed: {str(e)}")

    def _collect_statistics(self) -> Dict:
        """Collect and compute system-wide statistics."""
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'simulation_time': traci.simulation.getTime(),
                'active_vehicles': len(self.vehicle_states),
                'system_metrics': {
                    'average_speed': 0.0,
                    'average_waiting_time': 0.0,
                    'total_congestion_index': 0.0,
                    'emergency_vehicles_active': len([v for v in self.vehicle_states.values() if v.type == 'emergency']),
                    'rerouted_vehicles': sum(1 for v in self.vehicle_states.values() if v.reroute_attempts > 0)
                }
            }
            
            # Calculate averages
            if self.vehicle_states:
                speeds = [v.speed for v in self.vehicle_states.values()]
                waiting_times = [v.waiting_time for v in self.vehicle_states.values()]
                
                stats['system_metrics']['average_speed'] = np.mean(speeds)
                stats['system_metrics']['average_waiting_time'] = np.mean(waiting_times)
            
            if self.traffic_metrics:
                congestion_indices = [m.congestion_index for m in self.traffic_metrics.values()]
                stats['system_metrics']['total_congestion_index'] = np.mean(congestion_indices)
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics collection failed: {str(e)}")
            return {}

    def run_traffic_management(self):
        """Main traffic management loop."""
        try:
            step = 0
            
            while traci.simulation.getMinExpectedNumber() > 0:
                # Basic simulation step
                traci.simulationStep()
                
                # Regular updates
                self._update_vehicle_states()
                self.traffic_metrics = self._compute_edge_metrics()
                
                # Periodic optimization
                if step % self.OPTIMIZATION_INTERVAL == 0:
                    self._manage_emergency_vehicles()
                    self._manage_intersection_timing()
                    self._update_vehicle_routes()
                    
                    # Collect and log statistics
                    stats = self._collect_statistics()
                    logger.info(f"System Statistics: {stats}")
                
                # Apply continuous speed control
                for edge_id, metrics in self.traffic_metrics.items():
                    self._apply_speed_control(edge_id, metrics.congestion_index)
                
                step += 1
                
        except Exception as e:
            logger.error(f"Traffic management loop failed: {str(e)}")
            raise
        finally:
            try:
                traci.close()
            except:
                pass

def main():
    """Main entry point for the traffic management system."""
    try:
        # Initialize and run traffic manager
        traffic_manager = AdvancedTrafficManager()
        traffic_manager.run_traffic_management()
        
    except Exception as e:
        logger.critical(f"System failure: {str(e)}")
        raise
    finally:
        # Ensure proper cleanup
        if traci.isLoaded():
            traci.close()

if __name__ == '__main__':
    main()