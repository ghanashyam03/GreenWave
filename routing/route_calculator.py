import networkx as nx

class RouteCalculator:
    @staticmethod
    def calculate_route(road_network, source, target):
        return nx.shortest_path(road_network.graph, source=source, target=target)
