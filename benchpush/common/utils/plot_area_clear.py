import math
import os
from typing import List, Tuple, Iterable, Union

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, colors, cm

from .utils import scale_axis_labels, rotation_matrix


class Plot:
    """
    Aggregates all plotting objects into a single class
    """

    def __init__(
            self,
            costmap: np.ndarray,
            obstacles: List,
            path: np.ndarray = None,
            path_nodes: Tuple[List, List] = tuple(),
            nodes_expanded: dict = None,
            smoothing_nodes: Tuple[List, List] = tuple(),
            swath: np.ndarray = None,
            swath_cost: float = None,
            robot_pos: Union[Tuple, np.ndarray] = None,
            robot_vertices: np.ndarray = None,
            turning_radius: float = None,
            inf_stream=False,
            map_figsize=(5, 10),
            sim_figsize=(10, 10),
            target: Tuple[float, float] = None,
            legend=False,
            scale: float = 1,
            boundary_polygon: float = None,
    ):
        R = lambda theta: np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        self.ax = []
        self.path_line = []
        self.full_path = path  # shape is 3 x n
        self.inf_stream = inf_stream
        self.target = target
        self.node_scat = None

        self.boundary_polygon = boundary_polygon
        self.boundary_patches = []

        self.map = bool(map_figsize)
        if self.map:
            # init two fig and ax objects
            # the first is for plotting the updated costmap, node plot, swath, and path
            # the second is for plotting the simulated ship, polygons
            if nodes_expanded:
                self.map_fig, ax = plt.subplots(1, 2, figsize=map_figsize, sharex='all', sharey='all')
                self.node_ax, self.map_ax = ax
                self.ax.extend([self.node_ax, self.map_ax])

                # plot the nodes that were expanded
                self.create_node_plot(nodes_expanded)

            else:
                self.map_fig, self.map_ax = plt.subplots(1, 1, figsize=map_figsize)
                self.ax.append(self.map_ax)

            # add title
            self.map_ax.set_title('Costmap')

            # plot the costmap
            if costmap.sum() > 0:
                costmap[costmap == np.max(costmap)] = np.nan  # set the max to nan
                cmap = 'viridis'
            else:
                cmap = 'Greys'
            self.costmap_image = self.map_ax.imshow(costmap, origin='lower', cmap=cmap)

            # show the path on both the map and sim plot
            if self.full_path is not None:
                self.path_line.extend([
                    *self.map_ax.plot(self.full_path[0], self.full_path[1], 'g'),
                ])

            # plot the nodes along the path and the nodes added from the smoothing step
            self.nodes_line = []
            if len(path_nodes) != 0:
                self.nodes_line.append(*self.map_ax.plot(*path_nodes, 'bx'))
            if smoothing_nodes:
                self.nodes_line.append(*self.map_ax.plot(*smoothing_nodes, 'gx'))

                
            # plot the boundary polygon
            if boundary_polygon is not None:
                self.boundary_patches.append(
                    self.map_ax.add_patch(
                        patches.Polygon(boundary_polygon, True, fill=False, edgecolor='green')
                    )
                )


            if swath is not None:
                # init swath image
                swath_im = np.zeros(swath.shape + (4,))  # init RGBA array
                # fill in the RGB values
                swath_im[:] = colors.to_rgba('m')
                swath_im[:, :, 3] = swath  # set pixel transparency to 0 if pixel value is 0
                # plot the full swath
                self.swath_image = self.map_ax.imshow(swath_im, origin='lower', alpha=0.3)

            if swath_cost:
                self.map_ax.set_title('Swath cost: {:.2f}'.format(swath_cost))

            if legend:
                # create a path for swath
                patch_swath = patches.Patch(color=colors.to_rgba('m'), alpha=0.6)

                # add legend
                self.map_fig.legend(
                    (*self.nodes_line, self.path_line[0], patch_swath),
                    ('path nodes', 'smoothing nodes', 'path', 'swath')  # , loc=(0.5, 0)
                )

            # add the patches for the ice
            for obs in obstacles:
                self.map_ax.add_patch(
                    patches.Polygon(obs['vertices'], True, fill=False)
                )

            if robot_vertices is not None:
                assert robot_pos is not None
                if len(np.shape(robot_pos)) > 1:
                    # we have a list of poses
                    self.map_ax.plot(robot_pos[0], robot_pos[1], 'b-', label='ship path')
                    robot_pos = robot_pos[:, -1]

                self.map_ax.add_patch(
                    patches.Polygon(robot_vertices @ R(robot_pos[2]).T + robot_pos[:2], True, fill=False, color='red')
                )

                if turning_radius is not None:
                    x = np.arange(0, 2 * np.pi, 0.01)
                    self.map_ax.plot(
                        (robot_pos[0] - turning_radius * np.sin(robot_pos[2]) + turning_radius * np.cos(x)).tolist(),
                        (robot_pos[1] + turning_radius * np.cos(robot_pos[2]) + turning_radius * np.sin(x)).tolist(), 'g'
                    )
                    self.map_ax.plot(
                        (robot_pos[0] + turning_radius * np.sin(robot_pos[2]) + turning_radius * np.cos(x)).tolist(),
                        (robot_pos[1] - turning_radius * np.cos(robot_pos[2]) + turning_radius * np.sin(x)).tolist(), 'g'
                    )

        self.sim = bool(sim_figsize)
        if self.sim:

            self.sim_fig, self.sim_ax = plt.subplots(figsize=sim_figsize)
            # remove axes ticks and labels to speed up animation
            self.sim_ax.set_xlabel('')
            self.sim_ax.set_xticks([])
            self.sim_ax.set_ylabel('')
            self.sim_ax.set_yticks([])
            self.ax.append(self.sim_ax)

            # show the ship poses
            self.sim_ax.plot(robot_pos[0], robot_pos[1], 'b-', label='ship path')

            if self.full_path is not None:
                self.path_line.append(
                    *self.sim_ax.plot(self.full_path[0], self.full_path[1], 'r', label='planned path')
                )

            # add the patches for the ice
            self.sim_obs_patches = []
            for obs in obstacles:
                self.sim_obs_patches.append(
                    self.sim_ax.add_patch(
                        patches.Polygon(obs['vertices'], True, fill=True, fc='lightblue', ec='k')
                    )
                )
            
            # plot the boundary polygon
            if boundary_polygon is not None:
                self.boundary_patches.append(
                    self.sim_ax.add_patch(
                        patches.Polygon(boundary_polygon, True, fill=False, edgecolor='green')
                    )
                )

            #  add patch for ship
            if robot_vertices is not None:
                self.ship_patch = self.sim_ax.add_patch(
                    patches.Polygon(robot_vertices @ R(robot_pos[2]).T + robot_pos[:2], True, fill=True,
                                    edgecolor='black', facecolor='white', linewidth=2)
                )

            # keeps track of how far ship has traveled in subsequent steps
            self.prev_ship_pos = robot_pos

            # display target on path
            if target:
                self.target, *_ = self.sim_ax.plot(*target, 'xm', label='target', zorder=4)

            # self.sim_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 20})

        # set the axes limits for all plots
        for ax in self.ax:
            ax.axis([0, costmap.shape[1], 0, 30])
            ax.set_aspect('equal')

        if scale > 1:
            scale_axis_labels(self.ax, scale)

    def close(self):
        plt.close('all')

    def update_map(self, cost_map: np.ndarray, obstacles: List, ship_vertices=None, ship_pos=None) -> None:
        # update the costmap plot
        self.costmap_image.set_data(cost_map)

        self.map_ax.patches = []
        for obs in obstacles:
            self.map_ax.add_patch(
                patches.Polygon(obs['vertices'], True, fill=False)
            )

        if ship_vertices is not None:
            assert ship_pos is not None
            theta = ship_pos[2]
            R = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            self.map_ax.add_patch(
                patches.Polygon(ship_vertices @ R.T + ship_pos[:2], True, fill=False, color='red')
            )

    def update_path(
            self,
            full_path: np.ndarray,
            full_swath: np.ndarray = None,
            swath_cost: float = None,
            path_nodes: Tuple[List, List] = None,
            smoothing_nodes: Tuple[List, List] = None,
            nodes_expanded: dict = None,
            target: Tuple[float, float] = None,
            ship_state: Tuple[List, List] = None,
            past_path: Tuple[List, List] = None,
            start_y: float = None
    ) -> None:
        p_x, p_y, _ = full_path
        # show the new path
        for line in self.path_line:
            line.set_data(p_x, p_y)

        if self.target:
            self.target.set_data(*target)

        if ship_state:
            if self.ship_state_line is not None:
                self.ship_state_line[0].remove()
            self.ship_state_line = self.sim_ax.plot(ship_state[0], ship_state[1], 'b-', linewidth=1)

        if past_path:
            if self.past_path_line is not None:
                self.past_path_line[0].remove()
            self.past_path_line = self.sim_ax.plot(past_path[0], past_path[1], 'r--', linewidth=1)

        if self.map:
            # update the node plot
            if nodes_expanded:
                self.create_node_plot(nodes_expanded)

            # update the nodes lines
            if path_nodes is not None:
                self.nodes_line[0].set_data(path_nodes[0], path_nodes[1])
            if smoothing_nodes is not None:
                self.nodes_line[1].set_data(smoothing_nodes[0], smoothing_nodes[1])

            if full_swath is not None:
                swath_im = np.zeros(full_swath.shape + (4,))  # init RGBA array
                # fill in the RGB values
                swath_im[:] = colors.to_rgba('m')
                swath_im[:, :, 3] = full_swath  # set pixel transparency to 0 if pixel value is 0
                # update the swath image
                self.swath_image.set_data(swath_im)

            if swath_cost:
                self.map_ax.set_title('Swath cost: {:.2f}'.format(swath_cost))

            # update y axis
            ymin, ymax = self.map_ax.get_ylim()
            if self.inf_stream or full_path[1, -1] > ymax:
                offset = full_path[1, 0] - self.full_path[1, 0]
                self.map_ax.set_ylim([ymin + offset, ymax + offset])
                self.full_path = full_path


    def update_path_scatter(
            self,
            full_path: np.ndarray,
            start_y: float = None
    ) -> None:

        if len(full_path) == 3:
            p_x, p_y, _ = full_path
        else:
            p_x, p_y = full_path
        
        self.sim_ax.scatter(p_x, p_y, c='r', label='planned path', s=0.5)


    def update_robot(self, body, shape, move_yaxis_threshold=20) -> None:
        heading = body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vs = np.asarray(shape.get_vertices()) @ R.T + np.asarray(body.position)
        self.ship_patch.set_xy(vs)

        # compute how much ship has moved in the y direction since last step
        offset = np.array([0, body.position.y - self.prev_ship_pos[1]])
        self.prev_ship_pos = body.position  # update prev ship position

    def update_obstacles(self, polygons: List = None, obstacles: List = None) -> None:
        if polygons:
            for poly, patch in zip(polygons, self.sim_obs_patches):
                heading = poly.body.angle
                R = np.asarray([[math.cos(heading), -math.sin(heading)],
                                [math.sin(heading), math.cos(heading)]])
                # vs = np.asarray(poly.get_vertices()) @ R + np.asarray(poly.body.position)
                vs = np.asarray(poly.get_vertices()) @ R.T + np.asarray(poly.body.position)
                patch.set_xy(vs)

        elif obstacles:
            for obs, patch in zip(obstacles, self.sim_obs_patches):
                patch.set_xy(obs)

    def animate_map(self, save_fig_dir=None, suffix=0):
        # draw artists for map plot
        for artist in [*self.nodes_line, self.swath_image, self.path_line[0],
                       self.costmap_image, self.map_ax.yaxis]:
            self.map_ax.draw_artist(artist)

        # draw artists for node plot
        if self.node_scat:
            for artist in [self.node_scat, self.node_ax.yaxis]:
                self.node_ax.draw_artist(artist)

        self.map_fig.canvas.blit(self.map_fig.bbox)
        self.map_fig.canvas.flush_events()
        self.save(save_fig_dir, suffix)

    def animate_sim(self, save_fig_dir=None, suffix=0):
        # draw artists for map plot
        # for artist in [self.horizon_line, self.ship_patch, self.target,
        #                *self.sim_obs_patches, *self.path_line]:
        #     if artist is not None:
        #         self.sim_ax.draw_artist(artist)

        for artist in [self.ship_patch, *self.sim_obs_patches, *self.path_line, *self.boundary_patches]:
            if artist is not None:
                self.sim_ax.draw_artist(artist)

        # self.sim_fig.canvas.blit(self.sim_fig.bbox)
        self.sim_fig.canvas.flush_events()
        self.save(save_fig_dir, suffix, fig='sim')

    def save(self, save_fig_dir, suffix, im_format='png', fig='map'):
        if save_fig_dir:
            if not os.path.isdir(save_fig_dir):
                os.makedirs(save_fig_dir)
            fp = os.path.join(save_fig_dir, str(suffix) + '.' + im_format)  # pdf is useful in inkscape
            if fig == 'map':
                self.map_fig.savefig(fp)
            else:
                plt.axis('off')
                self.sim_fig.savefig(fp, bbox_inches="tight", transparent=False)
            return fp


    def get_sim_artists(self) -> Iterable:
        # this is only useful when blit=True in FuncAnimation
        # which requires returning a list of artists that have changed in the sim fig
        return (
            self.target, *self.path_line, self.ship_patch, *self.sim_obs_patches,
        )

    def create_node_plot(self, nodes_expanded: dict):
        c, data = self.aggregate_nodes(nodes_expanded)
        if self.node_scat is None:
            self.node_scat = self.node_ax.scatter(data[:, 0], data[:, 1], s=2, c=c, cmap='viridis')
            self.node_ax.set_title('Node plot {}'.format(len(nodes_expanded)))
        else:
            # set x and y data
            self.node_scat.set_offsets(data)
            # set colors
            self.node_scat.set_array(np.array(c))
            # update title
            self.node_ax.set_title('Node plot {}'.format(len(nodes_expanded)))

    @staticmethod
    def aggregate_nodes(nodes_expanded):
        c = {(k[0], k[1]): 0 for k in nodes_expanded}
        xy = c.copy()
        for k, val in nodes_expanded.items():
            key = (k[0], k[1])
            c[key] += 1
            if not xy[key]:
                x, y, _ = val
                xy[key] = [x, y]
        c = list(c.values())
        data = np.asarray(list(xy.values()))
        return c, data

    @staticmethod
    def show_prims(ax, pos, theta, prim_paths):
        R = rotation_matrix(theta)
        for path in prim_paths:
            x, y, _ = R @ path
            ax.plot([i + pos[0] for i in x],
                    [j + pos[1] for j in y], 'r', linewidth=0.5)

    @staticmethod
    def show_prims_from_nodes_edges(ax, prim, nodes, edges):
        for n, e in zip(nodes[:-1], edges):
            paths = [prim.paths[(e[0], k)] for k in prim.edge_set_dict[e[0]]]
            Plot.show_prims(ax, (n[0], n[1]), n[2] - e[0][2] * prim.spacing, paths)

    @staticmethod
    def add_ship_patch(ax, vertices, x, y, theta, ec='black', fc='white'):
        R = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        ax.add_patch(
            patches.Polygon(vertices @ R.T + [x, y], True, fill=True, edgecolor=ec, facecolor=fc)
        )
