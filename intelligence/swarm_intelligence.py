import random

class SwarmIntelligence:
    def __init__(self, vehicles):
        self.vehicles = vehicles

    def adjust_speeds(self):
        for vehicle in self.vehicles:
            if vehicle != self.vehicles[0]:  # Skip the adjustment for the first vehicle
                # Adjust speed based on random factors and messages received
                random_factor = random.uniform(0.9, 1.1)
                vehicle.speed = int(vehicle.speed * random_factor)
                vehicle.speed -= len(vehicle.messages_received) * 2  # Adjust by 2 units per message

                # Cap speed to a minimum of 20
                vehicle.speed = max(20, vehicle.speed)

