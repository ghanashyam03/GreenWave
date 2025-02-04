import os
from sumolib import checkBinary
import traci

def create_route_file(route_file, speed, start, destination):
    # Create a route file referencing the network's edge IDs
    with open(route_file, 'w') as route:
        route.write(f"""<routes>
    <vType id="car" accel="1.0" decel="5.0" sigma="0.5" length="5.0" maxSpeed="{speed}" />
    <route id="route1" edges="{start} {destination}" />
    <vehicle id="veh1" type="car" route="route1" depart="0" />
</routes>""")

def create_config_file(config_file, network_file, route_file):
    # Create a configuration file
    with open(config_file, 'w') as config:
        config.write(f"""<configuration>
    <input>
        <net-file value="{network_file}"/>
        <route-files value="{route_file}"/>
    </input>
</configuration>""")

def run_simulation(config_file):
    sumo_binary = checkBinary('sumo-gui')  # Use 'sumo' for GUI visualization
    traci.start([sumo_binary, "-c", config_file])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        vehicle_id = "veh1"
        x, y = traci.vehicle.getPosition(vehicle_id)
        print(f"Vehicle {vehicle_id} is at position: x={x}, y={y}")

    traci.close()

# Inputs from the user
network_file = "network.net.xml"  # Replace with the path to your uploaded file
route_file = "route.rou.xml"
config_file = "simulation.sumocfg"

# Dynamic input from the user
speed = float(input("Enter the vehicle's maximum speed (m/s): "))
start = input("Enter the starting edge ID: ")
destination = input("Enter the destination edge ID: ")

# Create route and config files
create_route_file(route_file, speed, start, destination)
create_config_file(config_file, network_file, route_file)

# Run the simulation
run_simulation(config_file)
    