import os
import traci
import sumolib
import numpy as np
from datetime import datetime
import logging
from typing import Dict
from collections import defaultdict
from scipy.stats import entropy  # Import entropy function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_metrics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrafficMetrics:
    def __init__(self):
        self.volume: float = 0
        self.speed_variance: float = 0
        self.speed_entropy: float = 0
        self.density: float = 0
        self.avg_speed: float = 0
        self.congestion_index: float = 0

class SimplifiedTrafficManager:
    def __init__(self):
        self.sumo_config = {
            'gui': True,
            'config_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\test.sumocfg',
            'net_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\network.net.xml',
            'route_file': r'C:\Users\ghana\OneDrive\Desktop\greenwave\kochi.rou.xml',
        }
        
        # Initialize data structures
        self.traffic_metrics: Dict[str, TrafficMetrics] = defaultdict(TrafficMetrics)
        
        # Load network
        self._initialize_system()

    def _initialize_system(self):
        """Initialize SUMO and load the network."""
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
            
            logger.info("Traffic Metrics System initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise

    def _compute_edge_metrics(self) -> Dict[str, TrafficMetrics]:
        """Compute comprehensive traffic metrics for each edge."""
        try:
            edge_metrics = defaultdict(TrafficMetrics)
            edge_speeds = defaultdict(list)
            edge_volumes = defaultdict(float)
            
            # Collect raw data
            for vehicle_id in traci.vehicle.getIDList():
                try:
                    vehicle_type = traci.vehicle.getVehicleClass(vehicle_id)
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    
                    if current_edge.startswith(':'): continue  # Skip internal edges
                    
                    pcu = 1.0  # Simplified PCU value (1 for all vehicles)
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    
                    edge_speeds[current_edge].append(speed)
                    edge_volumes[current_edge] += pcu
                    
                except traci.exceptions.TraCIException as e:
                    logger.warning(f"Failed to process vehicle {vehicle_id}: {str(e)}")
                    continue
            
            # Compute metrics for each edge
            for edge in self.net.getEdges():
                edge_id = edge.getID()
                speeds = edge_speeds.get(edge_id, [])
                volume = edge_volumes.get(edge_id, 0.0)
                
                metrics = TrafficMetrics()
                metrics.volume = volume
                
                if speeds:
                    metrics.avg_speed = float(np.mean(speeds))  # Convert to float for readability
                    metrics.speed_variance = float(np.var(speeds)) if len(speeds) > 1 else 0.0
                    metrics.speed_entropy = float(entropy(speeds)) if len(speeds) > 1 else 0.0
                
                # Compute density and congestion index
                edge_length = edge.getLength()
                edge_lanes = len(edge.getLanes())
                capacity = edge_length * edge_lanes * 0.8  # Theoretical capacity
                
                metrics.density = float(volume / (edge_length * edge_lanes)) if edge_length > 0 else 0.0
                metrics.congestion_index = float(volume / capacity) if capacity > 0 else 0.0
                
                edge_metrics[edge_id] = metrics
            
            return edge_metrics
            
        except Exception as e:
            logger.error(f"Edge metrics computation failed: {str(e)}")
            raise

    def _collect_statistics(self) -> Dict:
        """Collect and compute system-wide statistics in a human-readable format."""
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'simulation_time': float(traci.simulation.getTime()),  # Convert to float
                'active_vehicles': traci.vehicle.getIDCount(),
                'system_metrics': {
                    'average_speed': 0.0,
                    'average_waiting_time': 0.0,
                    'total_congestion_index': 0.0,
                }
            }
            
            # Calculate averages
            speeds = []
            waiting_times = []
            congestion_indices = []
            
            for vehicle_id in traci.vehicle.getIDList():
                speeds.append(traci.vehicle.getSpeed(vehicle_id))
                waiting_times.append(traci.vehicle.getWaitingTime(vehicle_id))
            
            for metrics in self.traffic_metrics.values():
                congestion_indices.append(metrics.congestion_index)
            
            if speeds:
                stats['system_metrics']['average_speed'] = float(np.mean(speeds))  # Convert to float
            if waiting_times:
                stats['system_metrics']['average_waiting_time'] = float(np.mean(waiting_times))  # Convert to float
            if congestion_indices:
                stats['system_metrics']['total_congestion_index'] = float(np.mean(congestion_indices))  # Convert to float
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics collection failed: {str(e)}")
            return {}

    def run_traffic_metrics(self):
        """Main loop for calculating and logging traffic metrics."""
        try:
            step = 0
            
            while traci.simulation.getMinExpectedNumber() > 0:
                # Basic simulation step
                traci.simulationStep()
                
                # Compute metrics
                self.traffic_metrics = self._compute_edge_metrics()
                
                # Log statistics every 10 steps
                if step % 10 == 0:
                    stats = self._collect_statistics()
                    logger.info(
                        f"Traffic Metrics:\n"
                        f"  Timestamp: {stats['timestamp']}\n"
                        f"  Simulation Time: {stats['simulation_time']:.2f} seconds\n"
                        f"  Active Vehicles: {stats['active_vehicles']}\n"
                        f"  Average Speed: {stats['system_metrics']['average_speed']:.2f} m/s\n"
                        f"  Average Waiting Time: {stats['system_metrics']['average_waiting_time']:.2f} seconds\n"
                        f"  Total Congestion Index: {stats['system_metrics']['total_congestion_index']:.4f}"
                    )
                
                step += 1
                
        except Exception as e:
            logger.error(f"Traffic metrics loop failed: {str(e)}")
            raise
        finally:
            try:
                traci.close()
            except:
                pass

def main():
    """Main entry point for the traffic metrics system."""
    try:
        # Initialize and run traffic metrics system
        traffic_manager = SimplifiedTrafficManager()
        traffic_manager.run_traffic_metrics()
        
    except Exception as e:
        logger.critical(f"System failure: {str(e)}")
        raise
    finally:
        # Ensure proper cleanup
        if traci.isLoaded():
            traci.close()

if __name__ == '__main__':
    main()