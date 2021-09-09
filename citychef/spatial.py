import numpy as np
import logging
from shapely.geometry import Point
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gp


logging.basicConfig(level=logging.INFO)


class Centres:

    def __init__(self, bbox, density=1, number=None):
        """
        :param bbox: np.array of xmin, ymin, xmax, ymax
        :param density: target points per unit area
        """
        (xmin, ymin), (xmax, ymax) = bbox
        area = (xmax - xmin) * (ymax - ymin)  # area of extended rectangle

        if number is None:
            self.num = np.random.poisson(area * density)  # Poisson number of points
        else:
            self.num = number

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

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis('equal')
        ax.scatter(self.locs[:, 0], self.locs[:, 1], alpha=1, marker='x', s=100, c='black')

    def __repr__(self):
        self.plot()
        return f"{self.size} centres"


class Clusters:

    def __init__(self, parents, size=10000, sigma=1):

        self.parents = parents

        numb_units = np.random.poisson(size / parents.size, parents.size)

        ids = []
        for i, units in enumerate(numb_units):
            ids.extend([i] * units)
        self.ids = np.array(ids)

        self.count = sum(numb_units)  # total number of points

        self.centres = np.zeros((self.count, 2))  # centre coordinates for all
        self.centres[:, 0] = np.repeat(parents.locs[:, 0], numb_units)
        self.centres[:, 1] = np.repeat(parents.locs[:, 1], numb_units)

        self.offsets = np.zeros((self.count, 2))
        self.offsets[:, 0] = np.random.normal(0, sigma, self.count)  # (relative) x coordinates
        self.offsets[:, 1] = np.random.normal(0, sigma, self.count)  # (relative) y coordinates

        # translate points (ie parents points are the centres of cluster disks)
        self.locs = np.zeros((self.count, 2))
        self.locs[:, 0] = self.centres[:, 0] + self.offsets[:, 0]
        self.locs[:, 1] = self.centres[:, 1] + self.offsets[:, 1]

    @property
    def bbox(self):
        maxx, minx = max(self.locs[:, 0]), min(self.locs[:, 0])
        maxy, miny = max(self.locs[:, 1]), min(self.locs[:, 1])
        return np.array(((minx, miny), (maxx, maxy)))

    def crop_to_bbox(self, bbox):
        (minx, miny), (maxx, maxy) = bbox
        mask = (self.locs[:,0] < maxx) & (self.locs[:,0] > minx) & (self.locs[:,1] < maxy) & (self.locs[:,1] > miny)
        self.locs = self.locs[mask]
        self.ids = self.ids[mask]
        self.centres = self.centres[mask]
        self.offsets = self.offsets[mask]
        return self

    @property
    def size(self):
        return len(self.locs)

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


def write_buildings_geojson(facilities, path, epsg="EPSG:27700", to_epsg=None):
    idxs = []
    activities = []
    geoms = []
    for name, facs in facilities.items():
        idxs.extend(list(range(facs.size)))
        activities.extend([name]*facs.size)
        geoms.extend(facs.points)
    data = {"index": idxs, "activity": activities, "geometry": geoms}
    # print(data)
    gdf = gp.GeoDataFrame(data, geometry="geometry", crs=epsg)
    if to_epsg is not None:
        gdf = gdf.to_crs(to_epsg)
    gdf.to_file(path, driver='GeoJSON')


def plot_facilities(facilities, centres=None, ax=None, alpha=.4, s=4):

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')

    def yield_cols():
        cols = ['blue', 'red', 'green', 'orange', 'yellow', 'brown', 'purple', 'grey']
        while True:
            for c in cols:
                yield c

    colour = yield_cols()
    handles = []
    for name, facility in facilities.items():
        c = next(colour)
        ax.scatter(facility.x, facility.y, alpha=alpha, s=s, marker='s', c=c)
        handles.append(mpatches.Patch(color=c, label=name))
    ax.legend(title='Facilities:', handles=handles)

    if centres:
        centres.plot(ax=ax)


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


def rand_poisson_points_normal(
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1,
        density=1,
        sigma=None
):
    logging.debug('Starting random poisson point process')
    # rectangle dimensions
    xdelta = xmax - xmin
    ydelta = ymax - ymin
    areatotal = xdelta * ydelta  # area of extended rectangle

    if sigma is None:
        sigma = (xmax - xmin) * .5 / density

    logging.debug('Simulate Poisson point process')
    numb_points = np.random.poisson(areatotal * density)  # Poisson number of points
    # x and y coordinates of Poisson points for the parent
    xx = xmin + xdelta * np.random.normal(0.5, sigma, numb_points)
    yy = ymin + ydelta * np.random.normal(0.5, sigma, numb_points)

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


def collect_bbox(units):
    locs = units[list(units)[0]].locs
    minx, miny, maxx, maxy = min(locs[:,0]), min(locs[:,1]), max(locs[:,0]), max(locs[:,1])
    for unit in units.values():
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


def density(features, objectives, density_radius=1):
    """
    Count of objectives within a given radius r to points in features"""
    if not isinstance(features, np.ndarray):
        features = features.locs
    if not isinstance(objectives, np.ndarray):
        objectives = objectives.locs

    tree = KDTree(objectives)
    density = tree.query_radius(features, count_only=True, r=density_radius)
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
