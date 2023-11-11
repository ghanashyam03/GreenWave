from communication.v2v_communication import V2VCommunication
from communication.traffic_event_manager import TrafficEventManager
from intelligence.swarm_intelligence import SwarmIntelligence
from intelligence.dynamic_route_planner import DynamicRoutePlanner
from infrastructure.road_network import RoadNetwork

class TrafficSimulator:
    def __init__(self, vehicles, communication, dynamic_route_planner):
        self.vehicles = vehicles
        self.communication = communication
        self.dynamic_route_planner = dynamic_route_planner

    def simulate_traffic(self, time_steps):
        for time_step in range(time_steps):
            self.communication_simulator()
            self.adjust_speeds()
            self.update_routes()
            self.display_output(time_step)

    def communication_simulator(self):
        for vehicle in self.vehicles:
            if vehicle.vehicle_id != 1:
                event_message = TrafficEventManager.generate_traffic_event()
                self.communication[0].send_message(event_message, target_vehicle=vehicle)

    def adjust_speeds(self):
        swarm_intelligence = SwarmIntelligence(vehicles=self.vehicles)
        swarm_intelligence.adjust_speeds()

    def update_routes(self):
        for vehicle in self.vehicles:
            if vehicle.vehicle_id != 1:
                dynamic_route = self.dynamic_route_planner.generate_dynamic_route()
                vehicle.route = dynamic_route

    def display_output(self, time_step):
        print(f"\nTime {time_step + 1}")
        for vehicle in self.vehicles:
            print(f"Vehicle {vehicle.vehicle_id}: Speed {vehicle.speed}, Route {vehicle.route}")
        print("\n")
