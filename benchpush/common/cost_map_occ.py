import logging
import os
import pickle
import random
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from skimage import draw

from benchpush.common.geometry.polygon import *
from benchpush.common.utils.utils import scale_axis_labels, rotation_matrix


# define an arbitrary max cost applied to a cell in the costmap
MAX_COST = 1e10


class CostMap_Occupancy:
    """
    Discretizes the environment into a 2D map and assigns a cost to each grid cell.
    Costmap for the predictive planner.
    This class serves more as a helper class for the predictive planner.
    """
    def __init__(self, cfg, scale: float, m: int, n: int, alpha: float = 10,
                 ship_mass: float = 1, horizon: int = None, margin: int = 1):
        """
        :param scale: the scaling factor for the costmap, divide by scale to get world units
        :param m: the height in world units of the channel
        :param n: the width in world units of the channel
        :param alpha: weight for the collision cost term
        :param ship_mass: mass of the ship in kg
        :param horizon: the horizon ahead of the ship that is considered for computing costs
        :param margin: the number of pixels to apply a max cost to at the boundaries of the channel
        """
        self.scale = scale  # scales everything by this factor
        self.cost_map = np.zeros((int(m * scale), int(n * scale)))
        self.alpha = alpha
        self.ship_mass = ship_mass
        self.obstacles = []
        self.horizon = horizon * scale if horizon else None
        self.margin = margin

        self.logger = logging.getLogger(__name__)

        # apply a cost to the boundaries of the channel
        self.boundary_cost()
        self.update_count = 0
        
    @property
    def shape(self):
        return self.cost_map.shape

    def boundary_cost(self) -> None:
        if not self.margin:
            return
        self.cost_map[:, :self.margin] = MAX_COST
        self.cost_map[:, -self.margin:] = MAX_COST

    def populate_costmap(self, centre, radius, pixels, normalization) -> None:
        rr, cc = pixels
        centre_x, centre_y = centre

        for (row, col) in zip(rr, cc):
            dist = np.sqrt((row - centre_y) ** 2 + (col - centre_x) ** 2)
            new_cost = max(0, (radius ** 2 - dist ** 2) / radius ** 2)
            old_cost = self.cost_map[row, col]
            self.cost_map[row, col] = min(MAX_COST, max(new_cost * normalization, old_cost))

        # make sure there are no pixels with 0 cost
        assert np.all(self.cost_map[rr, cc] > 0)


    def update(self, occ_map=None) -> None:
        """ Updates the costmap with the new obstacles and ship position and velocity 
        NOTE: same update function as above but for ROS
        """
        # clear costmap and obstacles
        self.cost_map[:] = 0
        self.cost_map[0:occ_map.shape[0]] = occ_map
        self.boundary_cost()



    def plot(self, obstacles: List[Any] = None, ship_pos=None, ship_vertices=None, prim=None, show_closest_ob=False, goal=None):
        f, ax = plt.subplots(figsize=(6, 10))
        # plot the costmap
        cost_map = self.cost_map.copy()
        cost_map[cost_map == np.max(cost_map)] = np.nan  # set the max to nan
        im = ax.imshow(cost_map, origin='lower', cmap='plasma')

        if obstacles is not None:
            obstacles = [{'vertices': np.asarray(obs) * self.scale} for obs in obstacles]
        else:
            # if no obstacles passed in then just show obs from self
            obstacles = self.obstacles

        # plot the polygons
        for obs in obstacles:
            if 'vertices' in obs:
                ax.add_patch(
                    patches.Polygon(obs['vertices'], True, fill=False)
                )

            if 'centre' in obs:
                # plot the centre of each polygon
                x, y = obs['centre']
                plt.plot(x, y, 'rx', markersize=15)

                # plot circle around polygon with computed radius
                p = np.arange(0, 2 * np.pi, 0.01)
                plt.plot(x + obs['radius'] * np.cos(p),
                         y + obs['radius'] * np.sin(p), 'c', linewidth=3)

        ax.set_title('Costmap unit: {}x{} m'.format(1 / self.scale, 1 / self.scale))
        scale_axis_labels([ax], scale=self.scale)
        # ax.set_xlabel('')
        # ax.set_xticks([])
        # ax.set_ylabel('')
        # ax.set_yticks([])
        f.colorbar(im, ax=ax)

        # plot ship if necessary
        if ship_pos is not None:
            assert ship_vertices is not None
            theta = ship_pos[2]
            R = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            ax.add_patch(patches.Polygon(ship_vertices @ R.T + [ship_pos[0], ship_pos[1]], True, fill=True))

            # plot the motion primitives if necessary
            if prim is not None:
                ax.set_title('Costmap unit: {}x{} m \n Lattice unit: {} m, Turning radius: {} l.u.'
                             .format(1 / self.scale, 1 / self.scale,
                                     prim.scale / self.scale,
                                     prim.turning_radius / prim.scale))
                origin = (0, 0, 0)
                edge_set = prim.edge_set_dict[origin]
                R2 = rotation_matrix(theta)
                for edge in edge_set:
                    path = prim.paths[(origin, tuple(edge))]
                    x, y, _ = R2 @ path
                    ax.plot([i + ship_pos[0] for i in x],
                            [j + ship_pos[1] for j in y], 'r', linewidth=0.5)

            if show_closest_ob:
                from benchpush.common.evaluation.metrics import min_obs_dist
                # d, ob = Metrics.obs_dist(CostMap.get_ob_vertices(self.obstacles), ship_pos, ship_vertices)
                # ax.add_patch(patches.Polygon(ob, True, fill=True, fc='r'))
                min_d = min_obs_dist(self.cost_map, ship_pos, ship_vertices)

                # debug code to make sure metric works
                new_footprint = np.asarray(
                    [[np.sign(a) * (abs(a) + min_d), np.sign(b) * (abs(b) + min_d)] for a, b in ship_vertices]
                )
                ax.add_patch(
                    patches.Polygon(new_footprint @ R.T + [ship_pos[0], ship_pos[1]], True, fill=False, ec='w'))

        if goal is not None:
            ax.axhline(goal, color='r', linestyle='--')

        # f.savefig('costmap.png', dpi=300)
        plt.show()

    def save_state_to_disk(self, filename='costmap', costmap_dir='../costmaps') -> None:
        if not os.path.isdir(costmap_dir):
            os.makedirs(costmap_dir)
        if filename:
            fp = os.path.join(costmap_dir, filename + '.pk')
            with open(fp, 'wb') as fd:
                pickle.dump(self, fd)
                self.logger.info("Successfully saved costmap object to file path '{}'".format(fp))

    @staticmethod
    def generate_obstacles(
            num_obs: int,
            min_r: int,
            max_r: int,
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
            allow_overlap=False,
            seed=None,
            **kwargs
    ) -> Tuple[List[dict], List[np.ndarray]]:

        # list to keep track of generated obs
        obstacles = []

        # generate a set of random circles
        _, circles = draw.random_shapes(image_shape=(max_y - min_y, max_x - min_x),
                                        max_shapes=num_obs, min_shapes=num_obs,
                                        max_size=max_r * 2, min_size=min_r * 2,
                                        multichannel=False, shape='circle',
                                        allow_overlap=allow_overlap, random_seed=seed)  # num_trials=100 by defaults

        # iterate over each circle and generate a polygon
        for circ in circles:
            # get bounding box coordinates
            r, c = circ[1]  # top right and bottom left coords
            radius = abs(r[0] - r[1]) / 2
            centre = (c[0] + radius + min_x, r[0] + radius + min_y)

            # now generate polygon given centre and radius
            vertices = generate_polygon(diameter=radius * 2, origin=centre)

            if vertices is not None:
                obstacles.append({
                    'vertices': vertices,
                    'centre': centre,
                    'radius': radius,
                })

        return obstacles, [ob['vertices'] for ob in obstacles]

    # @staticmethod
    # def get_obs_from_poly(polygons: List):
    #     return [
    #         np.asarray(
    #             [
    #                 v.rotated(-poly.body.angle) + poly.body.position
    #                 for v in poly.get_vertices()
    #             ]
    #         )
    #         for poly in polygons
    #     ]
    
    @staticmethod
    def get_obs_from_poly(polygons: List):
        R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return [
            np.asarray(poly.get_vertices()) @ R(poly.body.angle).T + poly.body.position
            for poly in polygons
        ]

    @staticmethod
    def resample_vertices(a, decimals=1):
        b = np.round(np.asarray(a), decimals)
        idx = np.unique(b, axis=0, return_index=True)[1]
        return np.asarray(a)[sorted(idx)]


def main():
    # seed for deterministic results
    seed = 1  # None to disable
    np.random.seed(seed)
    random.seed(seed)

    # generate obstacles
    obs_dict, obstacles = CostMap.generate_obstacles(
        num_obs=10,
        min_r=3,
        max_r=7,
        min_x=0,
        max_x=20,
        min_y=0,
        max_y=50,
        seed=seed
    )

    # initialize costmap
    costmap = CostMap(
        scale=4,
        m=50, n=20,
    )

    # update obstacles with costmap
    costmap.update(obstacles)

    # plot costmap with a dummy ship
    s = costmap.scale
    ship_pos = (10 * s, 20 * s, np.pi / 3)
    ship_vertices = np.asarray([[-4, 1], [-4, -1], [2, -1], [4, 0], [2, 1]]) * s
    costmap.plot(ship_pos=ship_pos,
                 ship_vertices=ship_vertices,
                 show_closest_ob=True)


if __name__ == "__main__":
    # run main to test costmap generation
    main()
