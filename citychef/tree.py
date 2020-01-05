import numpy as np
from enum import Enum, IntEnum
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd


def build_zones(grid):
    geom_data = []
    geom_maps = []
    centroids = []
    idx = 0

    for block in grid.traverse():

        if block.leaf:
            geom_data.append({'block_id': idx, 'density': block.density(), 'geometry': block.district()})
            geom_maps.append(block.build_point_data(idx))
            centroids.append(block.centre)
        idx += 1

    hh_grid_ids = np.concatenate(geom_maps)
    hh_grid_ids = hh_grid_ids[hh_grid_ids[:, 0].argsort()]
    hh_grid_ids = hh_grid_ids[:, 1]

    block_df = pd.DataFrame(geom_data)
    block_gdf = gpd.GeoDataFrame(block_df, geometry='geometry')

    return hh_grid_ids, block_gdf, np.array(centroids)


class BaseBlock:

    class Child(IntEnum):
        SW = 0
        NW = 1
        NE = 2
        SE = 3

    class Direction(Enum):
        SW = 0
        NW = 1
        NE = 2
        SE = 3
        N = 4
        S = 5
        W = 6
        E = 7

    def __init__(self, bbox, data, max_points, parent=None, depth=0):

        """
        :param bbox: [minx, miny, maxx, maxy]
        :param locs: [[index, x, y],]
        :param max_points: int
        """
        self.children = []
        self.leaf = False
        self.name = None
        self.junctions = None
        self.parent = parent
        self.depth = depth
        self.num_points = len(data)

        self.bbox = bbox
        self.centre = self.get_centre()
        self.data = data

        if self.num_points > max_points:
            self.divide(bbox, data, max_points)
        else:
            self.leaf = True

    def get_centre(self):
        raise NotImplemented

    def divide(self, bbox, data, max_points):

        left = data[:, 1] < self.centre[0]
        bottom = data[:, 2] < self.centre[1]

        # bottom left
        minx, miny, maxx, maxy = bbox[0, 0], bbox[0, 1], self.centre[0], self.centre[1]
        self.children.append(
            self.__class__(
                bbox=np.array([[minx, miny], [maxx, maxy]]),
                data=data[left & bottom],
                max_points=max_points,
                parent=self,
                depth=self.depth + 1
            )
        )

        # top left
        minx, miny, maxx, maxy = bbox[0, 0], self.centre[1], self.centre[0], bbox[1, 1]
        self.children.append(
            self.__class__(
                bbox=np.array([[minx, miny], [maxx, maxy]]),
                data=data[left & ~bottom],
                max_points=max_points,
                parent=self,
                depth=self.depth + 1
            )
        )

        # top right
        minx, miny, maxx, maxy = self.centre[0], self.centre[1], bbox[1, 0], bbox[1, 1]
        self.children.append(
            self.__class__(
                bbox=np.array([[minx, miny], [maxx, maxy]]),
                data=data[~left & ~bottom],
                max_points=max_points,
                parent=self,
                depth=self.depth + 1
            )
        )

        # bottom right
        minx, miny, maxx, maxy = self.centre[0], bbox[0, 1], bbox[1, 0], self.centre[1]
        self.children.append(
            self.__class__(
                bbox=np.array([[minx, miny], [maxx, maxy]]),
                data=data[~left & bottom],
                max_points=max_points,
                parent=self,
                depth=self.depth + 1
            )
        )

    def is_leaf(self):
        return not self.children

    def traverse(self):
        yield self
        for t in self.children:
            yield from t.traverse()

    def traverse_leaves(self):

        if self.leaf:
            yield self
        else:
            for t in self.children:
                t.traverse_leaves()
                yield t
            yield self

    def build_block_net(self, idx, G):

        (minx, miny), (maxx, maxy) = self.bbox
        centre_x, centre_y = self.centre[0], self.centre[1]
        offset_n = (maxy - centre_y) / 2
        offset_s = (centre_y - miny) / 2
        offset_e = (maxx - centre_x) / 2
        offset_w = (centre_x - minx) / 2

        def length(d):
            d += np.random.poisson(1)
            return d

        def freespeed(d):

            if d > 5:  # freeway
                speed = 130 / 3600  # km/s
                return speed
            if d > 2:
                speed = 100 / 3600  # km/s
                return speed
            if d > 1:
                speed = 60 / 3600  # km/s
                return speed
            if d > .5:
                speed = 40 / 3600  # km/s
                return speed
            speed = 30 / 3600  # km/s
            return speed

        def time(d, speed):
            return d / speed

        self.junctions = {
            "centre": (f"{idx}_centre", (centre_x, centre_y), 0),
            "north": (f"{idx}_north", (centre_x, centre_y + offset_n), offset_n),
            "south": (f"{idx}_south", (centre_x, centre_y - offset_s), offset_s),
            "east": (f"{idx}_east", (centre_x + offset_e, centre_y), offset_e),
            "west": (f"{idx}_west", (centre_x - offset_w, centre_y), offset_w),
        }

        for n, (name, pos, offset) in self.junctions.items():

            G.add_node(name, pos=pos)

            if n == "centre":
                continue
            if n == "north":
                d = length(offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(self.junctions["centre"][0], name, weight=t, distance=d, freespeed=s)
                G.add_edge(name, self.junctions["centre"][0], weight=t, distance=d, freespeed=s)
            if n == "south":
                d = length(offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(self.junctions["centre"][0], name, weight=t, distance=d, freespeed=s)
                G.add_edge(name, self.junctions["centre"][0], weight=t, distance=d, freespeed=s)
            if n == "east":
                d = length(offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(self.junctions["centre"][0], name, weight=t, distance=d, freespeed=s)
                G.add_edge(name, self.junctions["centre"][0], weight=t, distance=d, freespeed=s)
            if n == "west":
                d = length(offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(self.junctions["centre"][0], name, weight=t, distance=d, freespeed=s)
                G.add_edge(name, self.junctions["centre"][0], weight=t, distance=d, freespeed=s)

        if self.parent:

            # need to connect
            if self.parent.children[self.Child.SW] == self:
                # connect self north to parent west
                name, pos, offset = self.junctions['north']
                p_name, p_pos, p_offset = self.parent.junctions['west']
                d = length(offset + p_offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(name, p_name, weight=t, distance=d, freespeed=s)
                G.add_edge(p_name, name, weight=t, distance=d, freespeed=s)
                # connect self east to parent south
                name, pos, offset = self.junctions['east']
                p_name, p_pos, p_offset = self.parent.junctions['south']
                d = length(offset + p_offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(name, p_name, weight=t, distance=d, freespeed=s)
                G.add_edge(p_name, name, weight=t, distance=d, freespeed=s)

            elif self.parent.children[self.Child.NW] == self:
                # connect self south to parent west
                name, pos, offset = self.junctions['south']
                p_name, p_pos, p_offset = self.parent.junctions['west']
                d = length(offset + p_offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(name, p_name, weight=t, distance=d, freespeed=s)
                G.add_edge(p_name, name, weight=t, distance=d, freespeed=s)
                # connect self east to parent north
                name, pos, offset = self.junctions['east']
                p_name, p_pos, p_offset = self.parent.junctions['north']
                d = length(offset + p_offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(name, p_name, weight=t, distance=d, freespeed=s)
                G.add_edge(p_name, name, weight=t, distance=d, freespeed=s)

            elif self.parent.children[self.Child.NE] == self:
                # connect self south to parent east
                name, pos, offset = self.junctions['south']
                p_name, p_pos, p_offset = self.parent.junctions['east']
                d = length(offset + p_offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(name, p_name, weight=t, distance=d, freespeed=s)
                G.add_edge(p_name, name, weight=t, distance=d, freespeed=s)
                # connect self west to parent north
                name, pos, offset = self.junctions['west']
                p_name, p_pos, p_offset = self.parent.junctions['north']
                d = length(offset + p_offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(name, p_name, weight=t, distance=d, freespeed=s)
                G.add_edge(p_name, name, weight=t, distance=d, freespeed=s)

            elif self.parent.children[self.Child.SE] == self:
                # connect self north to parent east
                name, pos, offset = self.junctions['north']
                p_name, p_pos, p_offset = self.parent.junctions['east']
                d = length(offset + p_offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(name, p_name, weight=t, distance=d, freespeed=s)
                G.add_edge(p_name, name, weight=t, distance=d, freespeed=s)
                # connect self west to parent south
                name, pos, offset = self.junctions['west']
                p_name, p_pos, p_offset = self.parent.junctions['south']
                d = length(offset + p_offset)
                s = freespeed(d)
                t = time(d, s)
                G.add_edge(name, p_name, weight=t, distance=d, freespeed=s)
                G.add_edge(p_name, name, weight=t, distance=d, freespeed=s)

    def get_neighbor_of_greater_or_equal_size(self, direction):
        if direction == self.Direction.N:
            if self.parent is None:
                return None
            if self.parent.children[self.Child.SW] == self:  # Is 'self' SW child?
                return self.parent.children[self.Child.NW]
            if self.parent.children[self.Child.SE] == self:  # Is 'self' SE child?
                return self.parent.children[self.Child.NE]

            node = self.parent.get_neighbor_of_greater_or_equal_size(direction)
            if node is None or node.is_leaf():
                return node

            # 'self' is guaranteed to be a north child
            return (node.children[self.Child.SW]
                    if self.parent.children[self.Child.NW] == self  # Is 'self' NW child?
                    else node.children[self.Child.SE])

        elif direction == self.Direction.E:
            if self.parent is None:
                return None
            if self.parent.children[self.Child.NW] == self:
                return self.parent.children[self.Child.NE]
            if self.parent.children[self.Child.SW] == self:
                return self.parent.children[self.Child.SE]

            node = self.parent.get_neighbor_of_greater_or_equal_size(direction)
            if node is None or node.is_leaf():
                return node

            # 'self' is guaranteed to be a east child
            return (node.children[self.Child.NW]
                    if self.parent.children[self.Child.NE] == self
                    else node.children[self.Child.SW])

        elif direction == self.Direction.S:
            if self.parent is None:
                return None
            if self.parent.children[self.Child.NE] == self:
                return self.parent.children[self.Child.SE]
            if self.parent.children[self.Child.NW] == self:
                return self.parent.children[self.Child.SW]

            node = self.parent.get_neighbor_of_greater_or_equal_size(direction)
            if node is None or node.is_leaf():
                return node

            # 'self' is guaranteed to be a south child
            return (node.children[self.Child.NE]
                    if self.parent.children[self.Child.SE] == self
                    else node.children[self.Child.NW])

        elif direction == self.Direction.W:
            if self.parent is None:
                return None
            if self.parent.children[self.Child.NE] == self:
                return self.parent.children[self.Child.NW]
            if self.parent.children[self.Child.SE] == self:
                return self.parent.children[self.Child.SW]

            node = self.parent.get_neighbor_of_greater_or_equal_size(direction)
            if node is None or node.is_leaf():
                return node

            # 'self' is guaranteed to be a west child
            return (node.children[self.Child.NE]
                    if self.parent.children[self.Child.NW] == self
                    else node.children[self.Child.SE])

        return []

    def find_neighbors_of_smaller_size(self, neighbor, direction):
        candidates = [] if neighbor is None else [neighbor]
        neighbors = []

        if direction == self.Direction.N:
            while len(candidates) > 0:
                if candidates[0].is_leaf():
                    neighbors.append(candidates[0])
                else:
                    candidates.append(candidates[0].children[self.Child.SW])
                    candidates.append(candidates[0].children[self.Child.SE])

                candidates.remove(candidates[0])

            return neighbors

        elif direction == self.Direction.E:
            while len(candidates) > 0:
                if candidates[0].is_leaf():
                    neighbors.append(candidates[0])
                else:
                    candidates.append(candidates[0].children[self.Child.NW])
                    candidates.append(candidates[0].children[self.Child.SW])

                candidates.remove(candidates[0])

            return neighbors

        elif direction == self.Direction.S:
            while len(candidates) > 0:
                if candidates[0].is_leaf():
                    neighbors.append(candidates[0])
                else:
                    candidates.append(candidates[0].children[self.Child.NE])
                    candidates.append(candidates[0].children[self.Child.NW])

                candidates.remove(candidates[0])

            return neighbors

        elif direction == self.Direction.W:
            while len(candidates) > 0:
                if candidates[0].is_leaf():
                    neighbors.append(candidates[0])
                else:
                    candidates.append(candidates[0].children[self.Child.NE])
                    candidates.append(candidates[0].children[self.Child.SE])

                candidates.remove(candidates[0])

            return neighbors

    def get_neighbors(self, direction):
        neighbor = self.get_neighbor_of_greater_or_equal_size(direction)
        neighbors = self.find_neighbors_of_smaller_size(neighbor, direction)
        return neighbors

    def district(self):
        ((minx, miny), (maxx, maxy)) = self.bbox
        return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

    def get_point_ids(self):
        return self.data[:, 0]

    def build_point_data(self, idx):
        ones = np.ones(self.num_points)
        idx_array = ones * idx
        point_ids = self.get_point_ids()
        return np.stack((point_ids, idx_array), axis=1)

    def area(self):
        ((minx, miny), (maxx, maxy)) = self.bbox
        return (maxx - minx) * (maxy - miny)

    def density(self):
        return self.num_points / self.area()


class RegularBlock(BaseBlock):

    def get_centre(self):
        return self.bbox.mean(axis=0)


class IrregularBlock(BaseBlock):

    def get_centre(self):
        skew = np.random.choice([.45, .5, .55])
        diff = self.bbox[0] - self.bbox[1]
        return self.bbox[1] + diff * skew
