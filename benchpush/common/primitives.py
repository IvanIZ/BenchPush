import os
import pickle
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from benchpush.common.dubins_helpers.control import get_points_on_dubins_path
from benchpush.common.utils.utils import rotation_matrix

CACHED_PRIM_PATHS = '.prim_paths.pkl'  # save to disk primitive paths so no need to regenerate every time


class Primitives:
    """
    Class which stores the finite set of ship actions or motion primitives
    """
    def __init__(
            self,
            scale: float = 1,
            world_scale: float = 1,
            turning_radius: float = 1,
            num_headings: int = 8 or 16,
            step_size: float = 0.25,
            cache=True,
    ):
        """
        :param scale: scale factor for the primitives (scale to costmap scale)
        :param world_scale: scaling factor to some real-world realistic scale
        :param turning_radius: turning radius of the ship in metres
        :param num_headings: number of headings in the discretized heading space
        :param step_size: step size for sampling points on the path
        :param cache: whether to cache the primitive paths to disk
        """
        scale = scale * world_scale
        self.scale = scale
        self.turning_radius = turning_radius * self.scale
        self.num_headings = num_headings
        self.step_size = step_size

        edge_set_dict = self.get_primitives(self.num_headings)
        # scale the edge sets and turn them into npy arrays
        self.edge_set_dict = {}
        for k, v in edge_set_dict.items():
            self.edge_set_dict[k] = [(i[0] * scale, i[1] * scale, i[2]) for i in v]

        self.num_base_h = len(self.edge_set_dict.keys())
        self.paths, self.path_lengths = self.get_dubins_paths(cache)
        self.max_prim = self.get_max_prim()

        # compute the spacing between base headings
        self.spacing = 2 * np.pi / self.num_headings

        # for debugging purposes
        self.prim_count = {k1: {k2: 0 for k2 in self.edge_set_dict[k1]} for k1 in self.edge_set_dict}

    def plot(self, theta: float = 0, all_headings=False):
        """ plots all the primitives in the edge set dict """
        start_headings = np.asarray([0, 1, 2, 3]) * np.pi / 2 if all_headings else [0]

        for origin, edge_set in self.edge_set_dict.items():
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title('Theta: {} (rad)\nStart heading: {}'.format(round(theta, 5), origin[2]))
            color = 'b'

            for start_h in start_headings:
                R = rotation_matrix(theta + start_h)
                for edge in edge_set:
                    path = self.paths[(origin, tuple(edge))]
                    x, y, _ = R @ path
                    ax.plot(x, y, color)

                    # add text for each edge specifying the configuration
                    ax.text(x[-1], y[-1], str(edge), fontsize=8)

                    # plots the headings
                    # ax.plot(x, y, '.r')
                    # t = [(i + theta) % (2 * np.pi) for i in path[2]]
                    # px = x + np.cos(t)
                    # py = y + np.sin(t)
                    # for i in range(len(px)):
                    #     plt.plot([x[i], px[i]], [y[i], py[i]], 'r-')

            ax.set_aspect('equal')
            plt.savefig('prim.png')
            plt.show()

    def get_max_prim(self):
        return int(
            round(max(self.path_lengths.values()))
        )

    def get_dubins_paths(self, cache=True):
        """ sample points to get a dubins path for each primitive """
        if os.path.isfile(CACHED_PRIM_PATHS) and cache:
            print('LOADING CACHED PRIM PATHS! Confirm, this is expected behaviour...')
            return pickle.load(open(CACHED_PRIM_PATHS, 'rb'))

        paths = {}
        path_lengths = {}
        for origin, edge_set in self.edge_set_dict.items():
            for edge in edge_set:
                x, y, theta, path_length = get_points_on_dubins_path(
                    p1=origin, p2=edge, num_headings=self.num_headings, initial_heading=0,
                    turning_radius=self.turning_radius, step_size=self.step_size, eps=1e-10
                )
                paths[(origin, edge)] = np.asarray([x, y, theta])
                path_lengths[(origin, edge)] = path_length

        if cache:
            with open(CACHED_PRIM_PATHS, 'wb') as file:
                pickle.dump((paths, path_lengths), file)

        return paths, path_lengths

    def update_prim_count(self, prim_count):
        for k1, v1 in prim_count.items():
            for k2, v2 in v1.items():
                self.prim_count[k1][k2] += v2

    @staticmethod
    def rotate_path(path, theta: float):
        R = rotation_matrix(theta)
        x, y, _ = R @ path
        t = [(i + theta) % (2 * np.pi) for i in path[2]]
        return np.asarray([x, y, t])

    @staticmethod
    def rotate_prim(theta: float, edge_set_dict):
        # note the headings are left untouched
        R = rotation_matrix(theta)
        new_edge_set = {}
        for k, v in edge_set_dict.items():
            new_v = v @ R.T
            new_edge_set[k] = new_v

        return new_edge_set

    @staticmethod
    def get_primitives(num_headings) -> Dict[Tuple, List]:
        """
        Motion primitives were generated using the method described in the paper
        'Computing a Minimal Set of t-Spanning Motion Primitives for Lattice Planners'.
        Configurations are of the form (x, y, theta) and assume Dubins car kinematics
        """
        # all values are in lattice units
        if num_headings == 8:
            return {
                # lattice unit (l.u.) = 0.5 m and r = 2 l.u.
                (0, 0, 0): [
                    (1, 0, 0),
                    (2, 0, 0),
                    (3, 0, 0),
                    
                    (2, 1, 1),
                    (2, -1, 7),
                    (2, 2, 2),
                    (2, -2, 6),
                    (3, 1, 0),
                    (3, -1, 0),

                ],
                (0, 0, 1): [
                    (0, 3, 3),

                    (1, 1, 1),
                    (1.5, 1.5, 1),
                    (2, 2, 1),

                    (1, 2, 2),

                    (2, 1, 0),
                    (2, 3, 1),
                    (3, 0, 7),
                    (3, 2, 1)
                ]
            }
        elif num_headings == 16:
            return {
                # l.u. = 0.5 m and r = 1 l.u.
                (0, 0, 0): [
                    (1, 0, 0),
                    (1, 0, 1),
                    (1, 0, 15),
                    (1, 1, 4),
                    (1, -1, 12),
                    (2, 0, 2),
                    (2, 0, 14),
                    (2, 1, 0),
                    (2, -1, 0),
                    (2, 1, 1),
                    (2, -1, 15),
                    (2, 2, 1),
                    (2, 2, 2),
                    (2, 2, 3),
                    (2, 2, 4),
                    (2, -2, 12),
                    (2, -2, 13),
                    (2, -2, 14),
                    (2, -2, 15),
                    (2, -3, 0),
                    (2, 3, 1),
                    (2, 3, 2),
                    (2, -3, 14),
                    (2, -3, 15),
                    # (2, -4, 0),
                    (3, 0, 3),
                    (3, 0, 13),
                    (3, -1, 2),
                    (3, 1, 14),
                ],
                (0, 0, 1): [
                    (0, 3, 3),
                    (0, 3, 4),
                    (0, 3, 5),
                    (0, 3, 6),
                    (1, 0, 0),
                    (1, 0, 15),
                    (1, 1, 2),
                    (1, 1, 3),
                    (1, 2, 2),
                    (1, 2, 3),
                    (1, 3, 1),
                    (1, 3, 2),
                    (2, 0, 14),
                    (2, 1, 0),
                    (2, 1, 1),
                    (2, 1, 2),
                    (2, 1, 3),
                    (2, 1, 15),
                    (2, -1, 15),
                    (2, 2, 0),
                    (2, 2, 4),
                    (2, -2, 12),
                    (2, 3, 0),
                    (3, 0, 3),
                    (3, 0, 13),
                    (3, -2, 1)
                ],
                (0, 0, 2): [
                    (0, 2, 4),
                    (0, 2, 5),
                    (0, 2, 6),
                    (0, 3, 2),
                    (0, 3, 6),
                    (0, 3, 7),
                    (1, 1, 1),
                    (1, 1, 2),
                    (1, 1, 3),
                    (1, 2, 1),
                    (1, 2, 2),
                    (1, 2, 3),
                    (1, 3, 1),
                    (1, 3, 2),
                    (2, 0, 0),
                    (2, 0, 14),
                    (2, 0, 15),
                    (2, 1, 1),
                    (2, 1, 2),
                    (2, 1, 3),
                    (2, 2, 0),
                    (2, 2, 4),
                    (2, 2, 5),
                    (2, 2, 15),
                    (3, 0, 2),
                    (3, 0, 13),
                    (3, 0, 14),
                    (3, -1, 0),
                    (3, 1, 2),
                    (3, 1, 3),
                ],
                (0, 0, 3): [
                    (-2, 3, 3),
                    (-1, 2, 5),
                    (0, 1, 4),
                    (0, 1, 5),
                    (0, 2, 6),
                    (0, 3, 1),
                    (0, 3, 7),
                    (1, 1, 1),
                    (1, 1, 2),
                    (1, 2, 1),
                    (1, 2, 2),
                    (1, 2, 3),
                    (1, 2, 4),
                    (1, 2, 5),
                    (2, 1, 1),
                    (2, 1, 2),
                    (2, 2, 0),
                    (2, 2, 4),
                    (3, 0, 0),
                    (3, 0, 1),
                    (3, 0, 14),
                    (3, 0, 15),
                    (3, 1, 2),
                    (3, 1, 3),
                    (3, 2, 4),
                ]
            }
        else:
            print("Num headings '{}' not defined!".format(num_headings))
            exit(1)


if __name__ == '__main__':
    # for testing purposes
    p = Primitives(scale=3, world_scale=38, turning_radius=2, num_headings=8, step_size=10, cache=False)
    # p = Primitives(scale=1, turning_radius=1, num_headings=16, cache=False)
    p.plot(0, all_headings=False)
