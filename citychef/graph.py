import networkx as nx
import numpy as np
import geopandas as gp
from shapely.geometry import LineString
from scipy.spatial import Delaunay
import itertools
from matplotlib import pyplot as plt

from .tree import RegularBlock, IrregularBlock
from . import spatial


class TreeNetwork:

    def __init__(self, bbox, facility, grid='regular', max_points=500, label="highway"):
        self.bbox = bbox
        self.facility = facility
        self.grid_type = grid
        self.max_points = max_points
        self.label = label

        grid = self.build_grid_network()

        self.g = nx.DiGraph()
        idx = 0

        for block in grid.traverse():
            block.build_block_net(idx, self.g)
            idx += 1

        self.pos = {k: v['pos'] for k, v in self.g.nodes.items()}
        self.node_lookup = {i: name for i, name in enumerate(self.pos.keys())}
        self.node_index_lookup = {name: i for i, name in enumerate(self.pos.keys())}
        self.locs = np.array([p for p in self.pos.values()])

    def build_grid_network(self, random_length=None):

        # add index to locs so we can keep track of them
        data = np.zeros((self.facility.size, 3))
        data[:, 0] = range(self.facility.size)
        data[:, 1:] = self.facility.locs

        if self.grid_type == 'regular':
            return RegularBlock(
                bbox=self.bbox, data=data, max_points=self.max_points, random_length=random_length, label=self.label
            )
        elif self.grid_type == 'irregular':
            return IrregularBlock(
                bbox=self.bbox, data=data, max_points=self.max_points, random_length=random_length, label=self.label
            )
        else:
            raise UserWarning('Grid type must be either "regular" or "irregular".')

    @property
    def min_link_length(self):
        min_distance = np.inf
        for u, v, data in self.g.edges(data=True):
            if data['distance'] < min_distance:
                min_distance = data['distance']
        return min_distance

    @property
    def max_link_length(self):
        max_distance = -np.inf
        for u, v, data in self.g.edges(data=True):
            if data['distance'] > max_distance:
                max_distance = data['distance']
        return max_distance

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
            # fig.patch.set_visible(False)
            # ax.axis('off')
        nx.draw_networkx(
            self.g,
            pos=self.pos,
            ax=ax,
            edge_color='black',
            alpha=1,
            arrows=False,
            width=1,
            with_labels=False,
            node_size=2,
            node_color='black'
        )


class DelaunayNetwork:

    def __init__(self, locs, label=("railway", "rail")):
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

                self.g.add_edge(i, j, weight=t, distance=d, freespeed=s, label=label)
                self.g.add_edge(j, i, weight=t, distance=d, freespeed=s, label=label)

        self.pos = {k: v['pos'] for k, v in self.g.nodes.items()}
        self.node_lookup = {i: name for i, name in enumerate(self.pos.keys())}
        self.node_index_lookup = {name: i for i, name in enumerate(self.pos.keys())}

    @staticmethod
    def length(d):
        d += np.random.poisson(1)
        return d

    @staticmethod
    def freespeed(d):  # km/s
        if d < 1:
            return 50
        return 100

    @staticmethod
    def time(d, speed):  # seconds
        return ((d/1000) / speed) * 3600

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
            # fig.patch.set_visible(False)
            # ax.axis('off')
        nx.draw_networkx(
            self.g,
            pos=self.pos,
            ax=ax,
            edge_color='red',
            alpha=1,
            arrows=False,
            width=1,
            with_labels=False,
            node_size=2,
            node_color='red'
        )


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


class Transit:

    def __init__(self, network, facilities, density_radius=1000):

        self.routes = []
        self.num_routes = None
        self.g = None

        self.network = network
        self.facilities = facilities

        self.density = spatial.density(network.locs, facilities, density_radius=density_radius)

    def build_routes(self, num_routes=None, max_length=30000, min_length=10000, straightness=2):

        self.num_routes = num_routes
        if num_routes is None:
            self.num_routes = np.random.poisson(self.facilities.size / 2000)

        counter = 0
        while len(self.routes) < self.num_routes:
            route = PTRoute(
                network=self.network.g,
                node_weights=self.density**2,
                node_lookup=self.network.node_index_lookup,
                existing_routes=self.routes,
                max_length=max_length,
                straightness_weight=straightness,
            )
            if route.stops >= min_length:
                self.routes.append(route)
                counter = 0
            else:
                counter += 1
                if counter > 100:
                    raise TimeoutError

        return self.routes

    @property
    def graph(self):
        return nx.compose_all([r.g for r in self.routes])

    @property
    def min_link_length(self):
        g = self.graph
        min_distance = np.inf
        for u, v, data in g.edges(data=True):
            if data['distance'] < min_distance:
                min_distance = data['distance']
        return min_distance

    @property
    def max_link_length(self):
        g = self.graph
        max_distance = -np.inf
        for u, v, data in g.edges(data=True):
            if data['distance'] > max_distance:
                max_distance = data['distance']
        return max_distance

    def plot(self, ax=None, line_colour='red'):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(24, 24))
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
            # fig.patch.set_visible(False)
            # ax.axis('off')

        def gen_colour():
            cols = [
                'pink',
                'red',
                'lightyellow',
                'y',
                'orange',
                'lightblue',
                'b',
                'darkblue',
                'lightgreen',
                'g',
                'darkgreen',
                'purple',
                'violet',
                'grey',
                'gold'
            ]
            return np.random.choice(cols)

        # self.network.plot(ax=ax)

        for r in self.routes:
            g = r.g
            pos = {k: v['pos'] for k, v in g.nodes.items()}
            c = gen_colour()
            style = np.random.choice(['dotted', 'dashdot'])
            nx.draw_networkx_nodes(
                g, pos=pos, ax=ax, node_color='r', style=style, node_size=250, alpha=.5
            )
            nx.draw_networkx_nodes(
                g, pos=pos, ax=ax, node_color='white', style=style, node_size=100, alpha=.5
            )
            nx.draw_networkx_nodes(
                g, pos=pos, ax=ax, node_color=c, style=style, node_size=50, alpha=.5
            )
            nx.draw_networkx_edges(
                g, pos=pos, ax=ax, style=style, edge_color=line_colour, arrows=False, width=2.5
            )

    def interpolate_routes(self):
        for route in self.routes:
            route.interpolate_route()

    def jitter_locations(self, maximum):
        for route in self.routes:
            route.jitter_locations(maximum)


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
        self.population = 0
        self.length = 0

        self.start_node = self.weighted_random_init(node_weights)
        self.stops = 1
        while self.stepping():
            self.stops += 1

    def weighted_random_init(self, node_weights):
        node_weights = node_weights / node_weights.sum()
        n = np.random.choice(self.network.nodes, p=node_weights)
        pos = self.network.nodes[n]['pos']

        self.g.add_node(n, pos=pos)
        self.tail = n
        self.update_centroid()

        return n

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

    @property
    def min_link_length(self):
        min_distance = np.inf
        for u, v, data in self.g.edges(data=True):
            if data['distance'] < min_distance:
                min_distance = data['distance']
        return min_distance

    @property
    def max_link_length(self):
        max_distance = -np.inf
        for u, v, data in self.g.edges(data=True):
            if data['distance'] > max_distance:
                max_distance = data['distance']
        return max_distance

    def interpolate_route(self):
        new_route = nx.DiGraph()
        for u, v, data in self.g.edges(data=True):
            u_v = f"{u}--{v}"

            u_pos = self.g.nodes[u]['pos']
            v_pos = self.g.nodes[v]['pos']

            pos_x = (u_pos[0] + v_pos[0]) / 2
            pos_y = (u_pos[1] + v_pos[1]) / 2
            pos = (pos_x, pos_y)
            distance = data['distance'] / 2

            new_route.add_edge(u, u_v, distance=distance)
            new_route.add_edge(u_v, v, distance=distance)
            new_route.nodes[u]['pos'] = u_pos
            new_route.nodes[v]['pos'] = v_pos
            new_route.nodes[u_v]['pos'] = pos

        self.g = new_route

    def jitter_locations(self, maximum):

        for u in self.g.nodes():
            pos = self.g.nodes[u]['pos']

            sample = np.random.uniform(
                low=-maximum,
                high=maximum,
                size=2
            )

            jitter_pos = (pos[0] + sample[0], pos[1] + sample[1])
            nx.set_node_attributes(self.g, {u: {'pos': jitter_pos}})


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


def nx_to_geojson(g, path, epsg="EPSG:27700", to_epsg=None):

        links = []

        for idx, (u, v, d) in enumerate(g.edges(data=True)):
            linestring = LineString([g.nodes[u]['pos'], g.nodes[v]['pos']])
            index = f"00{idx}"
            links.append({
                    'id': index,
                    'distance': d.get("distance"),
                    'freespeed': d.get("freespeed"),
                    'label': d.get("label", ("unknown", "unknown"))[1],
                    'geometry': linestring,
                    })

        gdf = gp.GeoDataFrame(links, geometry="geometry", crs=epsg)
        if to_epsg is not None:
            gdf = gdf.to_crs(to_epsg)
        gdf.to_file(path, driver='GeoJSON')
