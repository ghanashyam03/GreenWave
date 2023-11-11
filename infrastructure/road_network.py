import networkx as nx

class RoadNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.generate_synthetic_data()

    def generate_synthetic_data(self):
        # Generate a synthetic road network
        self.graph.add_edge(1, 2, length=10)
        self.graph.add_edge(2, 3, length=15)
        self.graph.add_edge(3, 4, length=8)
        self.graph.add_edge(4, 5, length=12)

    def get_edge_length(self, edge):
        return self.graph[edge[0]][edge[1]]['length']
