import numpy as np
from enum import Enum, IntEnum


class RegularBlock:

    children = []
    leaf = False
    name = None

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

    def __init__(self, bbox, locs, max_points, parent=None, depth=0):

        """
        :param bbox: [[minx, miny], [maxx, maxy]]
        :param locs: [[index, x, y],]
        :param max_points: int
        """
        self.parent = parent
        self.depth = depth
        num_points = len(locs)
        print(num_points)
        centre = bbox.mean(axis=0)
        print(centre)

        if num_points > max_points:
            self.divide(bbox, centre, locs, max_points)
        else:
            self.leaf = True
            self.locs = locs
            self.centre = centre
            self.bbox = bbox
            print(bbox)
            self.num_points = num_points

    def divide(self, bbox, centre, locs, max_points):

        left = locs[:, 1] < centre[0]
        bottom = locs[:, 2] < centre[1]

        # bottom left
        minx, miny, maxx, maxy = bbox[0, 0], bbox[0, 1], centre[0], centre[1]
        self.children.append(
            RegularBlock(
                np.array([[minx, miny], [maxx, maxy]]),
                locs[left & bottom],
                max_points,
                parent=self,
                depth=self.depth + 1
            )
        )

        # top left
        minx, miny, maxx, maxy = bbox[0, 0], centre[1], centre[0], bbox[1, 1]
        self.children.append(
            RegularBlock(
                np.array([[minx, miny], [maxx, maxy]]),
                locs[left & ~bottom],
                max_points,
                parent=self,
                depth=self.depth + 1
            )
        )

        # top right
        minx, miny, maxx, maxy = centre[0], centre[1], bbox[1, 0], bbox[1, 1]
        self.children.append(
            RegularBlock(
                np.array([[minx, miny], [maxx, maxy]]),
                locs[~left & ~bottom],
                max_points,
                parent=self,
                depth=self.depth + 1
            )
        )

        # bottom right
        minx, miny, maxx, maxy = centre[0], bbox[0, 1], bbox[1, 0], centre[1]
        self.children.append(
            RegularBlock(
                np.array([[minx, miny], [maxx, maxy]]),
                locs[~left & bottom],
                max_points,
                parent=self,
                depth=self.depth + 1
            )
        )

    def is_leaf(self):
        return not self.children

    def traverse(self):

        if self.leaf:
            yield self
        else:
            for t in self.children:
                t.traverse()
                yield t
            yield self

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

    def roads(self):

        if self.leaf:

            centre = self.centre
            x = centre[0]
            y = centre[1]
            bottom = self.bbox[0, 1]
            left = self.bbox[0, 0]
            top = self.bbox[1, 1]
            right = self.bbox[1, 0]

            # top neighbours
            top_neighbours = self.get_neighbors(self.Direction.N)
            points = []
    