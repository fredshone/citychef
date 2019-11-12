import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

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
