import networkx as nx
import numpy as np


class NodesOD:

    def __init__(self, graph):

        self.node_lookup = {i: node for i, node in enumerate(graph.nodes())}
        self.index_lookup = {v: k for k, v in self.node_lookup.items()}

        count = len(self.node_lookup)
        self.od = np.zeros((count, count))

        for oi, origin in self.node_lookup.items():
            for di, destination in self.node_lookup.items():
                dist = nx.shortest_path_length(
                    graph,
                    source=origin,
                    target=destination,
                    weight='weight',
                )
                self.od[oi][di] = dist

    def get(self, o, d):
        return self.od[self.index_lookup[o], self.index_lookup[d]]



