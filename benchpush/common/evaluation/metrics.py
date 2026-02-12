from typing import List, Collection

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from skimage.draw import draw

from benchpush.common.geometry.polygon import poly_area, poly_centroid
from benchpush.common.swath import compute_swath_cost

from shapely.geometry import Polygon

__all__ = [
    'min_obs_dist',
    'path_smoothness',
    'velocity_smoothness',
    'curvature',
    'obs_occupancy',
    'path_length',
    'tracking_error',
    'total_work_done',
    'euclid_dist'
]


def min_obs_dist(costmap: np.ndarray, ship_pos: Collection, ship_vertices: np.ndarray, scale: float = 1, debug=False):
    """
    Finds the minimum euclidean distance between ship footprint and obstacles
    in the costmap. Treats all nonzero entries in costmap as an obstacle so
    make sure costs are not set for the borders.
    Distance transform from https://people.cs.uchicago.edu/~pff/papers/dt.pdf

    This function essentially builds a sort of Voronoi field
    "Voronoi Field can be augmented with a global attractive potential, yielding a field
    that has no local minima and is therefore suitable for global navigation."
    https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf
    """
    orig_map = costmap.copy()
    costmap = costmap.astype(bool).astype('uint8')
    costmap[costmap == 0] = 255  # assumes unoccupied cells are 0
    costmap[costmap == 1] = 0  # everything else is considered occupied
    dist = cv2.distanceTransform(src=costmap,  # this cv2 call takes 1 ms
                                 distanceType=cv2.DIST_L2,
                                 maskSize=cv2.DIST_MASK_PRECISE) / scale

    # get the cells on map for ship footprint
    x, y, theta = ship_pos
    R = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # could do interpolation here if costmap resolution is too low
    rot_vi = ship_vertices @ R.T + [x, y]
    rr, cc = draw.polygon(rot_vi[:, 1], rot_vi[:, 0])

    # plot to debug
    if debug:
        f, ax = plt.subplots(1, 4, figsize=(10, 5))
        ax[0].imshow(orig_map, origin='lower')
        ax[1].imshow(costmap, origin='lower')
        ax[1].add_patch(patches.Polygon(rot_vi, True, fill=False))
        ax[2].imshow(dist, origin='lower')
        ax[2].plot(x, y, 'rx')
        im = ax[3].imshow(dist[min(rr):max(rr), min(cc):max(cc)], origin='lower')
        f.colorbar(im, ax=ax[3])
        print(min(dist[rr, cc]))

    # return the smallest distance from cells inside ship footprint
    # distance of 0 means ship is in collision with an obstacle
    return min(dist[rr, cc])

def obs_to_goal_difference(obs_initial, obs_final, goal_points, boundary_polygon=None):
    """
    Computes the difference between the initial and final obstacles and the goal
    """
    total_diff = 0
    
    for ob_a, ob_b in zip(obs_initial, obs_final):
        if(boundary_polygon is not None and type(boundary_polygon) == Polygon):
            ob_a_polygon = Polygon(ob_a)
            if(not boundary_polygon.intersects(ob_a_polygon)):
                continue

        min_a_dist, min_b_dist = np.inf, np.inf
        centre_a = poly_centroid(ob_a)
        centre_b = poly_centroid(ob_b)
        for point in goal_points:
            min_a_dist = min(min_a_dist, euclid_dist(centre_a, (point.x, point.y)))
            min_b_dist = min(min_b_dist, euclid_dist(centre_b, (point.x, point.y)))
        
        total_diff += min_a_dist - min_b_dist

    return total_diff

def total_work_done(obs_initial, obs_final, debug=False, mass=None):
    """
    Computes an approximation of the total work done by ship by computing
    the displacements of the obstacles and multiplying by obstacle area

    Assumes obstacles are identically ordered across obstacle lists
    """
    work = 0
    for ob_a, ob_b, in zip(obs_initial, obs_final):
        # compute area of ob
        area = poly_area(ob_a)
        centre_a = np.abs(poly_centroid(ob_a))
        centre_b = np.abs(poly_centroid(ob_b))

        if mass is not None:
            work += euclid_dist(centre_a, centre_b) * mass
        else:
            work += euclid_dist(centre_a, centre_b) * area

        # plot to debug
        if debug:
            f, ax = plt.subplots()
            ax.imshow(np.zeros((76, 12)), origin='lower')
            ax.plot(*centre_a, 'rx')
            ax.add_patch(patches.Polygon(ob_a, True, fill=False))
            ax.plot(*centre_b, 'rx')
            ax.add_patch(patches.Polygon(ob_b, True, fill=True))
            print(area, poly_area(ob_b))
            plt.show()

    return work


def path_smoothness(path: np.ndarray) -> float:
    return ((np.diff(path[1:], axis=0) - np.diff(path[:-1], axis=0)) ** 2).sum()


def velocity_smoothness(path: np.ndarray = None, velocity: np.ndarray = None, time_steps: List = None) -> float:
    assert path is not None or velocity is not None
    if time_steps is None:
        time_steps = np.arange(len(velocity or path))
    if path is not None:
        assert len(time_steps) == len(path)
        dx_dt = np.gradient(path[:, 0], time_steps)
        dy_dt = np.gradient(path[:, 1], time_steps)
        velocity = np.asarray([dx_dt, dy_dt]).T
    else:
        assert len(time_steps) == len(velocity)
    diff_v = np.diff(velocity, axis=0)  # n - 1 x 2 where n is the number of time steps
    diff_t = np.diff(time_steps)  # n - 1
    return np.sqrt(
        np.sum(np.asarray([i / j for i, j in zip(diff_v, diff_t)]) ** 2, axis=1)  # n - 1
    ).sum() / (len(time_steps) - 1)  # take an average over n - 1 time steps


def curvature(path: np.ndarray = None, velocity: np.ndarray = None, time_steps: List = 1) -> float:
    assert path is not None or velocity is not None
    if type(time_steps) == List:
        assert len(time_steps) == len(path)  # if no time steps provided assumes dt is uniform
    if path is not None:
        dx_dt = np.gradient(path[:, 0], time_steps)
        dy_dt = np.gradient(path[:, 1], time_steps)
    else:
        dx_dt, dy_dt = velocity.T
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    c = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt ** 2 + dy_dt ** 2) ** 1.5

    # plot to debug
    # plt.scatter(path[:, 0], path[:, 1], c=cm.get_cmap('hot')(c), edgecolor='none')

    return c.sum()  # / Metrics.path_length(path)  # returns a normalized scalar  # TODO: look here https://arxiv.org/pdf/2107.11467.pdf


def obs_occupancy(costmap, path, ship_vertices):
    swath, _ = compute_swath_cost(costmap, path, ship_vertices)
    return np.array(costmap, dtype=bool)[swath].sum()


def path_length(path, cumsum=False):
    assert path.shape[1] == 2, 'path shape should be nx2 where each row are x,y coordinates'
    p = np.sum(np.diff(path, axis=0) ** 2, axis=1) ** 0.5  # this is an approximation
    if cumsum:
        return p.cumsum()
    return p.sum()


def tracking_error(pose, path, get_idx=False):
    # find the point on path that minimizes 2 norm distance with pose
    idx = np.argmin([euclid_dist(pose[:2], i[:2]) for i in path])
    e_x, e_y = pose[:2] - path[idx][:2]
    e_yaw = np.diff(np.unwrap([path[idx][2], pose[2]])).item()
    assert abs(e_yaw) <= np.pi
    track_e = np.abs([e_x, e_y, e_yaw])

    if get_idx:
        return track_e, idx
    return track_e


def euclid_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
