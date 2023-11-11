from communication.v2v_communication import V2VCommunication
from communication.traffic_event_manager import TrafficEventManager
from intelligence.swarm_intelligence import SwarmIntelligence
from intelligence.dynamic_route_planner import DynamicRoutePlanner
from infrastructure.road_network import RoadNetwork
from infrastructure.traffic_simulator import TrafficSimulator
from routing.eco_routing import EcoRouting
from routing.route_calculator import RouteCalculator
from vehicle import Vehicle

def main():
    num_vehicles = int(input("Enter the number of vehicles: "))
    initial_speed = int(input("Enter the initial speed of vehicles: "))
    time_steps = int(input("Enter the number of simulation time steps: "))

    vehicles = [Vehicle(vehicle_id=i, speed=initial_speed) for i in range(1, num_vehicles + 1)]

    communication = [V2VCommunication(vehicle_id=i) for i in range(1, num_vehicles + 1)]
    traffic_event_manager = TrafficEventManager()
    swarm_intelligence = SwarmIntelligence(vehicles=vehicles)
    road_network = RoadNetwork()
    dynamic_route_planner = DynamicRoutePlanner()
    eco_routing = EcoRouting(vehicles=vehicles, road_network=road_network)

    road_network.generate_synthetic_data()

    traffic_simulator = TrafficSimulator(
        vehicles=vehicles,
        communication=communication,
        dynamic_route_planner=dynamic_route_planner
    )

    traffic_simulator.simulate_traffic(time_steps)

if __name__ == "__main__":
    main()