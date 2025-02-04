import traci
import traci.constants as tc
import os

def create_sumo_simulation(speed, location):
    # Check SUMO_HOME environment variable
    if 'SUMO_HOME' not in os.environ:
        raise EnvironmentError("Please set the SUMO_HOME environment variable to your SUMO installation path.")

    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    sumo_config = "network.sumocfg"  # Replace with your SUMO configuration file path

    # Start the simulation
    traci.start([sumo_binary, "-c", sumo_config])

    try:
        step = 0
        print(f"Simulation started. Setting speed: {speed}, location: {location}")
        
        while step < 100:  # Run for 100 simulation steps or until manually stopped
            traci.simulationStep()

            # Assuming a vehicle is present in the network
            vehicle_id = "veh0"  # Replace with your vehicle ID
            if step == 10:  # Apply settings after a few simulation steps
                traci.vehicle.add(vehicle_id, routeID="route0")  # Replace routeID with actual route
                traci.vehicle.moveToXY(vehicle_id, edgeID="", lane=0, x=location[0], y=location[1])
                traci.vehicle.setSpeed(vehicle_id, speed)

            step += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        traci.close()
        print("Simulation finished.")   

if __name__ == "__main__":
    # Example input
    user_speed = float(input("Enter the vehicle speed (m/s): "))
    user_location = tuple(map(float, input("Enter the location (x, y) as comma-separated values: ").split(',')))

    create_sumo_simulation(user_speed, user_location)
