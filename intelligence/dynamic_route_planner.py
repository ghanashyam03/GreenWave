import random

class DynamicRoutePlanner:
    @staticmethod
    def generate_dynamic_route():
        routes = [
            [(1, 2), (2, 3), (3, 4)],
            [(1, 3), (3, 4)],
            [(1, 2), (2, 3), (3, 4), (4, 5)]
        ]
        return random.choice(routes)
