# vehicle.py
class Vehicle:
    def __init__(self, vehicle_id, speed=60):
        self.vehicle_id = vehicle_id
        self.speed = speed
        self.messages_received = []
        self.eco_routing_score = 1  # Default eco-routing score
        self.route = []  # List to store the vehicle's route
        self.speed_history = {0: speed}  # Speed at time 0

    def receive_message(self, message):
        self.messages_received.append(message)

    def get_speed_at_time(self, time):
        return self.speed_history.get(time, 0)

    def update_speed(self, time, speed):
        self.speed = speed
        self.speed_history[time] = speed
