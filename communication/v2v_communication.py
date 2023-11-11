class V2VCommunication:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.messages_received = []

    def send_message(self, message, target_vehicle):
        target_vehicle.receive_message(message)

    def receive_message(self, message):
        self.messages_received.append(message)

