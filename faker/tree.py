class RegularBlock:

    children = []
    leaf = False

    def __init__(self, xx, yy, max_points, minx, miny, maxx, maxy):
        self.num_points = len(xx)
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        x = (self.maxx-self.minx)/2
        y = (self.maxy-self.miny)/2

        if self.num_points > max_points:
            self.divide(xx, yy, x, y, max_points)
        else:
            self.leaf = True
            self.x = x
            self.y = y

    def divide(self, xx, yy, x, y, max_points):
        left = xx < x
        bottom = yy < y

        # bottom left
        self.children.append(
            RegularBlock(
                (
                    xx[left & bottom], yy[left & bottom], max_points, self.minx, self.miny, x, y
                )
            )
        )

        # top left
        self.children.append(
            RegularBlock(
                (
                    xx[left & ~bottom], yy[left & ~bottom], max_points, x, self.miny, self.maxx, y
                )
            )
        )

        # top right
        self.children.append(
            RegularBlock(
                (
                    xx[~left & ~bottom], yy[~left & ~bottom], max_points, x, y, self.maxx, self.miny
                )
            )
        )

        # bottom right
        self.children.append(
            RegularBlock(
                (
                    xx[~left & bottom], yy[~left & bottom], max_points, self.maxx, y, x, self.miny
                )
            )
        )

    def traverse_all(self):

        if self.leaf:
            yield self
        else:
            for t in self.children:
                t.traverse()
                yield t
            yield self
        pass

    def traverse_leaves(self):

        if self.leaf:
            yield self
        else:
            for t in self.children:
                t.traverse()
                yield t
        pass









