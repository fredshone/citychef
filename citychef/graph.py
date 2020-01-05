import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
import itertools


class TreeNetwork:

    def __init__(self, grid):

        self.g = nx.DiGraph()
        idx = 0

        for block in grid.traverse():
            block.build_block_net(idx, self.g)
            idx += 1

        self.pos = {k: v['pos'] for k, v in self.g.nodes.items()}
        self.node_lookup = {i: name for i, name in enumerate(self.pos.keys())}
        self.node_index_lookup = {name: i for i, name in enumerate(self.pos.keys())}
        self.locs = np.array([p for p in self.pos.values()])


class DelaunayNetwork:

    def __init__(self, locs):
        self.locs = locs
        tri = Delaunay(locs)
        self.g = nx.DiGraph()
        for p in tri.vertices:
            for i, j in itertools.combinations(p, 2):
                self.g.add_node(i, pos=locs[i])
                self.g.add_node(j, pos=locs[j])

                d = distance(locs[i], locs[j])
                d = self.length(d)
                s = self.freespeed(d)
                t = self.time(d, s)

                self.g.add_edge(i, j, weight=t, distance=d, freespeed=s)
                self.g.add_edge(j, i, weight=t, distance=d, freespeed=s)

        self.pos = {k: v['pos'] for k, v in self.g.nodes.items()}
        self.node_lookup = {i: name for i, name in enumerate(self.pos.keys())}
        self.node_index_lookup = {name: i for i, name in enumerate(self.pos.keys())}

    @staticmethod
    def length(d):
        d += np.random.poisson(1)
        return d

    @staticmethod
    def freespeed(d):
        if d < 1:
            return 50 / 3600
        return 100 / 3600  # km/s

    @staticmethod
    def time(d, speed):
        return d / speed


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


class NodesODAstar:

    def __init__(self, graph):

        self.node_lookup = {i: node for i, node in enumerate(graph.nodes())}
        self.index_lookup = {v: k for k, v in self.node_lookup.items()}

        count = len(self.node_lookup)
        self.matrix = np.zeros((count, count))

        def distance_heuristic(o, d):
            (x1, y1) = graph.nodes[o]['pos']
            (x2, y2) = graph.nodes[d]['pos']
            return abs(x1 - x2) + abs(y1 - y2)

        for oi, origin in self.node_lookup.items():
            for di, destination in self.node_lookup.items():
                # dist = nx.shortest_path_length(
                #     graph,
                #     source=origin,
                #     target=destination,
                #     weight='weight',
                # )
                dist = nx.astar_path_length(
                    graph,
                    source=origin,
                    target=destination,
                    heuristic=distance_heuristic,
                    weight='weight')
                self.matrix[oi][di] = dist

    def get(self, o, d):
        return self.matrix[o, d]

    def lookup(self, o, d):
        return self.matrix[self.index_lookup[o], self.index_lookup[d]]


class NodesOD:

    def __init__(self, graph):

        self.node_lookup = {i: node for i, node in enumerate(graph.nodes())}
        self.index_lookup = {v: k for k, v in self.node_lookup.items()}

        count = len(self.node_lookup)
        self.matrix = np.zeros((count, count))

        for oi in range(count):
            origin = self.node_lookup[oi]
            for di in range(oi, count):
                destination = self.node_lookup[di]
                try:
                    dist = nx.shortest_path_length(
                        graph,
                        source=origin,
                        target=destination,
                        weight='weight',  # note that this is time not distance
                    )
                except nx.NetworkXNoPath:
                    dist = -1

                self.matrix[oi][di] = dist
                self.matrix[di][oi] = dist

    def get(self, o, d):
        return self.matrix[o, d]

    def lookup(self, o, d):
        return self.matrix[self.index_lookup[o], self.index_lookup[d]]


class PTRoute:
    def __init__(self, network, node_weights, node_lookup, existing_routes, max_length=30, straightness_weight=2):

        self.network = network
        self.node_weights = node_weights
        self.node_lookup = node_lookup
        self.existing_routes = existing_routes
        self.max_length = max_length
        self.straightness_weight = straightness_weight

        self.g = nx.DiGraph()
        self.g_return = nx.DiGraph()
        self.tail = None
        self.centroid = None
        self.stops = 1
        self.population = 0
        self.length = 0

        self.weighted_random_init(node_weights)
        while self.stepping():
            self.stops += 1

    def weighted_random_init(self, node_weights):
        node_weights = node_weights / node_weights.sum()
        n = np.random.choice(self.network.nodes, p=node_weights)
        pos = self.network.nodes[n]['pos']

        self.g.add_node(n, pos=pos)
        self.tail = n
        self.update_centroid()

    def update_centroid(self):
        self.centroid = np.array([d.get('pos') for n, d in self.g.nodes.data()]).mean(axis=0)

    def add_step(self, n):
        pos = self.network.nodes[n]['pos']
        self.g.add_node(n, pos=pos)

        self.g.add_edge(self.tail, n)
        for key, value in self.network[self.tail][n].items():
            self.g[self.tail][n][key] = value
        self.length += self.network[self.tail][n]['distance']

        self.g_return.add_edge(n, self.tail)
        for key, value in self.network[n][self.tail].items():
            self.g_return[n][self.tail][key] = value

        self.tail = n
        self.update_centroid()

    def stepping(self):

        # check length
        if self.stops == self.max_length:
            return False

        options = {n for n in self.network[self.tail]}

        # prevent reversing previous step
        tail_edge = self.g.in_edges(self.tail)
        if tail_edge:
            last_tail = list(tail_edge)[0][0]
            options -= {last_tail}
            if not options:
                return False

        # prevent repeating edges
        for option in list(options):
            if (self.tail, option) in self.g.edges:
                options -= {option}

        options = list(options)
        if not options:
            return False

        if len(options) == 1:  # force step
            n = options[0]
            self.add_step(n)
            return True

        # get scores
        num_options = len(options)
        scores = np.zeros((3, num_options))

        for i, option in enumerate(options):

            # dist
            pos = self.network.nodes[option]['pos']
            scores[0, i] = np.sqrt(np.sum((pos - self.centroid) ** 2))

            # density
            scores[1, i] = self.node_weights[self.node_lookup[option]]

            # repeated
            repeats = 0
            for existing_route in self.existing_routes:
                if (self.tail, option) in existing_route.g.edges:
                    repeats += 1
                if (option, self.tail) in existing_route.g.edges:
                    repeats += 1
            scores[2, i] = 0.5 ** repeats

        # standardise dist and skew
        scores[0] = (scores[0] / scores[0].max()) ** 3
        # weight
        scores[0] = scores[0] * self.straightness_weight

        # build probs
        scores = scores.sum(axis=0)
        probs = scores / scores.sum()

        # choose
        n = np.random.choice(options, p=probs)
        self.add_step(n)
        self.population += self.node_weights[self.node_lookup[n]]  # update population

        return True

    def build(self):
        """
        add complete return route, add edge weights from parent network if exists, add schedule and vehicles
        :return: None
        """
        raise NotImplementedError

