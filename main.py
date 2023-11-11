from flask import Flask, render_template, request
from communication.v2v_communication import V2VCommunication
from intelligence.swarm_intelligence import SwarmIntelligence
from infrastructure.road_network import RoadNetwork
from infrastructure.traffic_simulator import TrafficSimulator
from intelligence.dynamic_route_planner import DynamicRoutePlanner
from routing.eco_routing import EcoRouting
from vehicle import Vehicle

app = Flask(__name__, template_folder="webapp/templates")

num_vehicles = 0
initial_speed = 0
time_steps = 0
vehicles = []
communication = []
swarm_intelligence = None
eco_routing = None
traffic_simulator = None
dynamic_route_planner = None

def initialize_components():
    global num_vehicles, initial_speed, time_steps, vehicles, communication, swarm_intelligence, eco_routing, traffic_simulator, dynamic_route_planner

    vehicles = [Vehicle(vehicle_id=i, speed=initial_speed) for i in range(1, num_vehicles + 1)]
    communication = [V2VCommunication(vehicle_id=i) for i in range(1, num_vehicles + 1)]
    swarm_intelligence = SwarmIntelligence(vehicles=vehicles)
    eco_routing = EcoRouting(vehicles=vehicles, road_network=None)  # Replace 'None' with your road_network instance
    dynamic_route_planner = DynamicRoutePlanner()  # Initialize your dynamic route planner here

    traffic_simulator = TrafficSimulator(
        vehicles=vehicles,
        communication=communication,
        dynamic_route_planner=dynamic_route_planner
    )

def simulate_traffic():
    global num_vehicles, initial_speed, time_steps, vehicles, swarm_intelligence, traffic_simulator

    vehicles = [Vehicle(vehicle_id=i, speed=initial_speed) for i in range(1, num_vehicles + 1)]
    communication = [V2VCommunication(vehicle_id=i) for i in range(1, num_vehicles + 1)]
    swarm_intelligence = SwarmIntelligence(vehicles=vehicles)
    eco_routing = EcoRouting(vehicles=vehicles, road_network=None)  # Replace 'None' with your road_network instance
    dynamic_route_planner = DynamicRoutePlanner()  # Initialize your dynamic route planner here

    traffic_simulator = TrafficSimulator(
        vehicles=vehicles,
        communication=communication,
        dynamic_route_planner=dynamic_route_planner
    )

    for step in range(1, time_steps + 1):
        # Placeholder logic: Update speed for each vehicle at each time step
        for vehicle_id, vehicle in enumerate(vehicles, start=1):
            # Use adjust_speeds instead of calculate_new_speed
            swarm_intelligence.adjust_speeds()

        update_routes(step)
        traffic_simulator.simulate_traffic(step)

def update_routes(time):
    global vehicles, eco_routing, dynamic_route_planner

    # Use EcoRouting to update routes based on the current state
    eco_routing.update_routes()

    # Use DynamicRoutePlanner to generate dynamic routes
    dynamic_route = dynamic_route_planner.generate_dynamic_route()

@app.route('/', methods=['GET', 'POST'])
def index():
    global num_vehicles, initial_speed, time_steps

    if request.method == 'POST':
        num_vehicles = int(request.form['num_vehicles'])
        initial_speed = int(request.form['initial_speed'])
        time_steps = int(request.form['time_steps'])

        initialize_components()
        simulate_traffic()

    # Get simulation results
    simulation_results = get_simulation_results()

    return render_template('simulation.html',
                           num_vehicles=num_vehicles,
                           initial_speed=initial_speed,
                           time_steps=time_steps,
                           simulation_results=simulation_results)

def get_simulation_results():
    global time_steps, vehicles

    results = []

    for step in range(1, time_steps + 1):
        step_results = []
        for vehicle_id, vehicle in enumerate(vehicles, start=1):
            speed = vehicle.get_speed_at_time(step)
            step_results.append(f"Time {step}, Vehicle {vehicle_id}: Speed {speed}")

        results.append(step_results)

    return results

if __name__ == "__main__":
    app.run(debug=True)

