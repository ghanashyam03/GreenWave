import random

class TrafficEventManager:
    @staticmethod
    def generate_traffic_event():
        events = ["Accident", "Road Construction", "Heavy Traffic", "Police Checkpoint"]
        return random.choice(events)
