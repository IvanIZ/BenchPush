import os
import pickle
from typing import Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from skimage import transform
from skimage.draw import draw

Swath = Dict[Tuple, np.ndarray]
CACHED_SWATH = '.swath.pkl'  # save to disk swaths so no need to regenerate every time


def generate_swath(ship, prim, cache=True, model_inference=False, cache_path=None) -> Swath:
    """
    Generate swath for each motion primitive given the ship footprint
    For each edge we generate the swath centered on a square array
    This makes it easy to rotate the swath about the image centre

    The resolution of the swath (i.e. the size of a grid cell) is
    the same as the resolution of the costmap, i.e. they are 1:1
    """
    if cache_path is None:
        cache_path = CACHED_SWATH

    if os.path.isfile(cache_path) and cache:
        print('LOADING CACHED SWATH! Confirm, this is expected behaviour...')
        return pickle.load(open(cache_path, 'rb'))

    # this is super hacky, but basically we make the halves larger than
    # the ship vertices to fix the issue of overlap across concatenated swaths
    big_r_half = np.asarray([[a, np.sign(b) * (abs(b) + ship.width / 2)] for a, b in ship.right_half])
    big_l_half = np.asarray([[a, np.sign(b) * (abs(b) + ship.width / 2)] for a, b in ship.left_half])

    R = lambda t: np.asarray([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    swath_dict = {}
    # generate the swaths for each 0, 90, 180, 270 degrees
    # since our headings are uniformly discretized along the unit circle
    for i, h in enumerate(range(0, prim.num_headings, prim.num_headings // 4)):
        for origin, edge_set in prim.edge_set_dict.items():
            start_pos = [prim.max_prim + ship.max_ship_length // 2] * 2 + [origin[2]]

            for edge in edge_set:
                array = np.zeros([(prim.max_prim + ship.max_ship_length // 2) * 2 + 1] * 2, dtype=bool)
                path = prim.paths[(origin, edge)]  # assumes path is sampled finely enough to get the swath
                path = prim.rotate_path(path, np.pi / 2 * i)

                for p in path.T:
                    x, y, theta = p
                    x, y = x + start_pos[0], y + start_pos[1]
                    rot_vi = np.array([[x], [y]]) + R(theta) @ ship.vertices.T
                    rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :])
                    array[rr, cc] = True

                # this final step removes the overlap problem
                # from concatenating swaths during graph search
                if model_inference:     # for ice model, only remove ship tail from swath start
                    for p, verts in zip([path.T[0]], [big_l_half]):
                        x, y, theta = p
                        x, y = x + start_pos[0], y + start_pos[1]
                        rot_vi = np.array([[x], [y]]) + R(theta) @ verts.T
                        rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :])
                        array[rr, cc] = False

                # else:                   # for planning, remove ship tail from start, remove ship head from end
                #     for p, verts in zip([path.T[0], path.T[-1]], [big_l_half, big_r_half]):
                #         x, y, theta = p
                #         x, y = x + start_pos[0], y + start_pos[1]
                #         rot_vi = np.array([[x], [y]]) + R(theta) @ verts.T
                #         rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :])
                #         array[rr, cc] = False

                else:                   # for planning, remove entire ship from start
                    for p, verts in zip([path.T[0], path.T[0]], [big_l_half, big_r_half]):
                        x, y, theta = p
                        x, y = x + start_pos[0], y + start_pos[1]
                        rot_vi = np.array([[x], [y]]) + R(theta) @ verts.T
                        rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :])
                        array[rr, cc] = False

                swath_dict[edge, h + origin[2]] = array

    if cache:
        with open(cache_path, 'wb') as file:
            pickle.dump(swath_dict, file)

    return swath_dict


def view_swath(swath_dict: Swath, key: Tuple = None) -> None:
    if key is None:
        # get a random key from swath dict
        idx = np.random.randint(0, len(swath_dict), 1)[0]
        key = list(swath_dict.keys())[idx]
        print(key)
    shape = swath_dict[key].shape
    plt.plot(shape[0] / 2, shape[0] / 2, 'bx')
    plt.plot(shape[0] / 2 + key[0][0], shape[0] / 2 + key[0][1], 'bx')
    plt.imshow(swath_dict[key], origin='lower')
    plt.show()
    # plt.savefig(str(key) + '.pdf')


def view_all_swaths(swath_dict: Swath) -> None:
    for k, _ in swath_dict.items():
        view_swath(swath_dict, k)


def rotate_swath(swath, theta: float) -> Swath:
    return transform.rotate(swath, - theta * 180 / np.pi, order=0, preserve_range=True)


def compute_swath_cost(cost_map: np.ndarray,
                       path: np.ndarray,
                       ship_vertices: np.ndarray,
                       compute_cumsum=False,
                       plot=False
                       ) -> Tuple[np.ndarray, Union[float, list]]:
    """
    Generate the swath given a costmap, path, and ship footprint. This swath will be a little
    different from the one generated by A* search since in A* we use the processed swath dict.
    This method is useful when generating the swath on the final smoothed path.
    NOTE, this is a very expensive operation and should only be used for plotting/debugging purposes!

    This also assumes the path is sampled at a high enough frequency to capture the swath
    """
    # keep track of swath
    swath = np.zeros_like(cost_map, dtype=bool)
    cumsum = [] if compute_cumsum else None

    # compute the swath for each point
    for x, y, theta in path:  # full path is of shape n x 3
        R = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # rotate/translate vertices of ship from origin to sampled point with heading = theta
        rot_vi = np.array([[x], [y]]) + R @ ship_vertices.T

        # draw rotated ship polygon and put occupied cells into a mask
        rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :], shape=cost_map.shape)
        swath[rr, cc] = True

        if compute_cumsum:
            cumsum.append(cost_map[swath].sum())

    if plot:
        f, ax = plt.subplots()
        ax.imshow(cost_map, origin='lower')
        ax.plot(path[:, 0], path[:, 1], 'r')
        swath_im = np.zeros(swath.shape + (4,))  # init RGBA array
        swath_im[:] = colors.to_rgba('m')
        swath_im[:, :, 3] = swath  # set pixel transparency to 0 if pixel value is 0
        # plot the full swath
        ax.imshow(swath_im, origin='lower', alpha=0.3, zorder=10)
        plt.show()

    if compute_cumsum:
        return swath, cumsum

    return swath, cost_map[swath].sum()
