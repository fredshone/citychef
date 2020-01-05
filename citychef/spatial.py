import numpy as np
import logging
from shapely.geometry import Point
from sklearn.neighbors import KDTree


logging.basicConfig(level=logging.INFO)


def collect_bbox(units):
    locs = units[0].locs
    minx, miny, maxx, maxy = min(locs[:,0]), min(locs[:,1]), max(locs[:,0]), max(locs[:,1])
    for unit in units:
        locs = unit.locs
        pminx, pminy, pmaxx, pmaxy = min(locs[:,0]), min(locs[:,1]), max(locs[:,0]), max(locs[:,1])
        if pminx < minx:
            minx = pminx
        if pminy < miny:
            miny = pminy
        if pmaxx > maxx:
            maxx = pmaxx
        if pmaxy > maxy:
            maxy = pmaxy
    return np.array([[minx,miny],[maxx,maxy]])


def distance_index_nearest_node(features, objectives):
    tree = KDTree(objectives)
    dist, ind = tree.query(features.locs, dualtree=True, k=1)

    return dist.reshape(1, -1)[0], ind.reshape(1, -1)[0]


def minmax(array, axis=0):
    if len(array.shape) > 1:
        return (array - array.min(axis=axis)) / (array.max(axis=axis) - array.min(axis=axis))
    else:
        return (array - min(array)) / (max(array) - min(array))


def density(features, objectives, r=1):
    """
    Count of objectives within a given radius r to points in features"""
    if not isinstance(features, np.ndarray):
        features = features.locs
    if not isinstance(objectives, np.ndarray):
        objectives = objectives.locs

    tree = KDTree(objectives)
    density = tree.query_radius(features, count_only=True, r=r)
    return minmax(density)


def distances_to_closest(features, objectives, num):
    assert len(objectives.locs >= num)
    tree = KDTree(objectives.locs)
    dist_closest, _ = tree.query(features.locs, dualtree=True, k=num)

    if num > 1:
        dist_closest = dist_closest.sum(axis=1)
    else:
        dist_closest = dist_closest.reshape(-1)

    return minmax(dist_closest)


class Centres:

    def __init__(self, bbox, density=1):
        """
        :param bbox: np.array of xmin, ymin, xmax, ymax
        :param density: target points per unit area
        """
        (xmin, ymin), (xmax, ymax) = bbox
        area = (xmax - xmin) * (ymax - ymin)  # area of extended rectangle
        self.num = np.random.poisson(area * density)  # Poisson number of points

        self.locs = np.zeros((self.num, 2))
        self.locs[:, 0] = xmin + (xmax - xmin) * np.random.uniform(0, 1, self.num)
        self.locs[:, 1] = ymin + (ymax - ymin) * np.random.uniform(0, 1, self.num)

    @property
    def size(self):
        return self.num

    @property
    def x(self):
        return self.locs[:, 0]

    @property
    def y(self):
        return self.locs[:, 1]

    @property
    def points(self):
        for x, y in self.locs:
            yield Point(x, y)

    def __repr__(self):
        return f"{self.size} centres"


class Clusters:

    def __init__(self, centres, size=10000, sigma=1):

        self.parents = centres

        numb_units = np.random.poisson(size / centres.size, centres.size)

        ids = []
        for i, units in enumerate(numb_units):
            ids.extend([i] * units)
        self.ids = np.array(ids)

        self.count = sum(numb_units)  # total number of points

        self.centres = np.zeros((self.count, 2))  # centre coordinates for all
        self.centres[:, 0] = np.repeat(centres.locs[:, 0], numb_units)
        self.centres[:, 1] = np.repeat(centres.locs[:, 1], numb_units)

        self.offsets = np.zeros((self.count, 2))
        self.offsets[:, 0] = np.random.normal(0, sigma, self.count)  # (relative) x coordinates
        self.offsets[:, 1] = np.random.normal(0, sigma, self.count)  # (relative) y coordinates

        # translate points (ie parents points are the centres of cluster disks)
        self.locs = np.zeros((self.count, 2))
        self.locs[:, 0] = self.centres[:, 0] + self.offsets[:, 0]
        self.locs[:, 1] = self.centres[:, 1] + self.offsets[:, 1]

        # adjust bbox
        maxx, minx = max(self.locs[:, 0]), min(self.locs[:, 0])
        maxy, miny = max(self.locs[:, 1]), min(self.locs[:, 1])
        self.bbox = np.array(((minx, miny), (maxx, maxy)))

    @property
    def size(self):
        return self.count

    @property
    def x(self):
        return self.locs[:, 0]

    @property
    def y(self):
        return self.locs[:, 1]

    @property
    def cx(self):
        return self.centres[:, 0]

    @property
    def cy(self):
        return self.centres[:, 1]

    @property
    def ox(self):
        return self.offsets[:, 0]

    @property
    def oy(self):
        return self.offsets[:, 1]

    @property
    def points(self):
        for x, y in self.locs:
            yield Point(x, y)

    def dist_to_centres(self):
        return np.sqrt(((self.locs - self.centres) ** 2).sum(axis=1))

    def __repr__(self):
        return f"{self.size} units, {self.parents.size} centres"


def rand_poisson_points(
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1,
        density=1,
):
    logging.debug('Starting random poisson point process')
    # rectangle dimensions
    xdelta = xmax - xmin
    ydelta = ymax - ymin
    areatotal = xdelta * ydelta  # area of extended rectangle

    logging.debug('Simulate Poisson point process')
    numb_points = np.random.poisson(areatotal * density)  # Poisson number of points
    # x and y coordinates of Poisson points for the parent
    xx = xmin + xdelta * np.random.uniform(0, 1, numb_points)
    yy = ymin + ydelta * np.random.uniform(0, 1, numb_points)

    return xx, yy


def thomas_cluster_process(
        xmin=0,  # note that random point processes can extend beyond boundaries
        xmax=1,
        ymin=0,
        ymax=1,
        lambda_parent=1,  # density of parent Poisson point process
        lambda_daughter=100,  # mean number of points in each cluster
        sigma=1  # sigma for normal variables (ie random locations) of daughters
):
    logging.debug('Starting thomas cluster gen')

    logging.debug('Simulate Poisson point process for the parents')
    xx_parent, yy_parent = rand_poisson_points(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        density=lambda_parent,  # density of parent Poisson point process
    )

    logging.debug(f'Simulate Poisson point process for the daughters (from {len(xx_parent)} centres)')
    numb_points_daughter = np.random.poisson(lambda_daughter, len(xx_parent))

    centre_ids = []
    for i, daughters in enumerate(numb_points_daughter):
        centre_ids.extend([i] * daughters)

    numb_points = sum(numb_points_daughter)  # total number of points

    logging.debug(f'Generate the (relative) locations in Cartesian coordinates {numb_points}')
    xx0 = np.random.normal(0, sigma, numb_points)  # (relative) x coordinates
    yy0 = np.random.normal(0, sigma, numb_points)  # (relative) y coordinates

    logging.debug('replicate parent points (ie centres of disks/clusters)')
    xx_parent_repeated = np.repeat(xx_parent, numb_points_daughter)
    yy_parent_repeated = np.repeat(yy_parent, numb_points_daughter)

    # translate points (ie parents points are the centres of cluster disks)
    xx = xx_parent_repeated + xx0
    yy = yy_parent_repeated + yy0

    return xx, yy, xx_parent_repeated, yy_parent_repeated, xx_parent, yy_parent, np.array(centre_ids)
