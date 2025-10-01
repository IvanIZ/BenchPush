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
            horizon: float = None,
            goal: float = None,
            goal_point: Tuple[float, float] = None,
            inf_stream=False,
            map_figsize=(5, 10),
            sim_figsize=(10, 10),
            target: Tuple[float, float] = None,
            y_axis_limit=100,
            legend=False,
            scale: float = 1,
            boundaries: List = None,
            maze =  None
            # corners: List = None
    ):
        R = lambda theta: np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        self.ax = []
        self.path_line = []
        self.full_path = path  # shape is 3 x n
        self.horizon = horizon if horizon != np.inf else None
        self.goal = goal
        self.goal_point = goal_point
        self.inf_stream = inf_stream
        self.target = target
        self.node_scat = None

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

            # add the patches for the boundaries
            if boundaries:
                self.sim_bounds_patches = []
                for bound in boundaries:
                    if bound['type'] == 'corner':
                        for poly in bound['vertices']:
                            self.sim_bounds_patches.append(
                                self.sim_ax.add_patch(
                                    patches.Polygon(poly, True, fill=True, fc='black', ec='black')
                                )
                            )
                    elif bound['type'] == 'receptacle':
                        self.sim_bounds_patches.append(
                            self.sim_ax.add_patch(
                                patches.Polygon(bound['vertices'], True, fill=True, fc='green', ec=None)
                            )
                        )
                    else:
                        self.sim_bounds_patches.append(
                            self.sim_ax.add_patch(
                                patches.Polygon(bound['vertices'], True, fill=True, fc='black', ec=None)
                            )
                        )

            # add the patches for the ice
            self.sim_obs_patches = []
            for i in range(len(obstacles)):
                obs = obstacles[i]
                self.sim_obs_patches.append(
                    self.sim_ax.add_patch(
                        patches.Polygon(obs['vertices'], True, fill=True, fc='lightblue', ec=None, gid=i)
                    )
                )

            #  add patch for ship
            if robot_vertices is not None:
                self.ship_patch = self.sim_ax.add_patch(
                    patches.Polygon(robot_vertices @ R(robot_pos[2]).T + robot_pos[:2], True, fill=True,
                                    edgecolor=None, facecolor='red', linewidth=2)
                )

            if self.horizon:
                self.horizon_line = self.sim_ax.axhline(y=self.horizon + self.full_path[1, 0], color='orange',
                                                        linestyle='--', linewidth=3.0, label='intermediate goal')

            if self.goal:
                self.goal_line = self.sim_ax.axhline(y=self.goal, color='g', linestyle='-',
                                                     linewidth=3.0, label='final goal')
                self.sim_fig.canvas.draw()

            if self.goal_point:
                self.goal_line, *_ = self.sim_ax.plot(*self.goal_point, 'go', label='goal', zorder=3)

            self.ship_state_line = None
            self.past_path_line = None

            # keeps track of how far ship has traveled in subsequent steps
            self.prev_ship_pos = robot_pos

            # display target on path
            if target:
                self.target, *_ = self.sim_ax.plot(*target, 'xm', label='target', zorder=4)

            # self.sim_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 20})

        # set the axes limits for all plots
        for ax in self.ax:
            ax.axis([0, costmap.shape[1], 0, y_axis_limit])
            #ax.axis([-costmap.shape[1] / 2, costmap.shape[1] / 2, -y_axis_limit / 2, y_axis_limit / 2])
            ax.set_aspect('equal')

        if scale > 1:
            scale_axis_labels(self.ax, scale)

    def close(self):
        # plt.close(self.obs_fig)
        plt.close(self.sim_fig)

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

        if self.sim:
            # update goal line segment
            if self.horizon:
                if start_y and self.horizon + start_y < self.goal_line.get_ydata()[0]:
                    self.horizon_line.set_ydata(self.horizon + start_y)
                else:
                    self.horizon_line.set_visible(False)


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

        if self.sim:
            # update goal line segment
            if self.horizon:
                if start_y and self.horizon + start_y < self.goal_line.get_ydata()[0]:
                    self.horizon_line.set_ydata(self.horizon + start_y)
                else:
                    self.horizon_line.set_visible(False)


    def update_ship(self, body, shape, move_yaxis_threshold=20) -> None:
        heading = body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vs = np.asarray(shape.get_vertices()) @ R.T + np.asarray(body.position)
        self.ship_patch.set_xy(vs)

        # compute how much ship has moved in the y direction since last step
        offset = np.array([0, body.position.y - self.prev_ship_pos[1]])
        self.prev_ship_pos = body.position  # update prev ship position

        # update y axis if necessary
        if (
                self.inf_stream and
                body.position.y > move_yaxis_threshold and
                body.position.y + self.horizon < self.goal
        ):
            ymin, ymax = self.sim_ax.get_ylim()
            self.sim_ax.set_ylim([ymin + offset[1], ymax + offset[1]])

    def update_obstacles(self, polygons: List = None, obstacles: List = None, obs_idx: int = None, update_patch: bool = False) -> None:
        if polygons:
            for poly, patch in zip(polygons, self.sim_obs_patches):
                heading = poly.body.angle
                R = np.asarray([[math.cos(heading), -math.sin(heading)],
                                [math.sin(heading), math.cos(heading)]])
                # vs = np.asarray(poly.get_vertices()) @ R + np.asarray(poly.body.position)
                vs = np.asarray(poly.get_vertices()) @ R.T + np.asarray(poly.body.position)
                patch.set_xy(vs)

        elif obstacles:
            if update_patch:
                for patch in self.sim_obs_patches:
                    if patch.get_gid() == obs_idx:
                        patch.remove()
                        self.sim_obs_patches.remove(patch)
                        self.sim_fig.canvas.draw_idle()

            else:
                for obs, patch in zip(obstacles, self.sim_obs_patches):
                    patch.set_xy(obs)
            
                if obs_idx is not None:
                    for i in range(len(self.sim_obs_patches)):
                        if i == obs_idx:
                            self.sim_obs_patches[i].set_facecolor('red')  # Change fill color
                        else:
                            self.sim_obs_patches[i].set_facecolor('lightblue')  # Change fill color
            

    def plot_maze(self, maze_walls: List, width) -> None:
        for wall in maze_walls:
            self.sim_ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', linewidth=width)
        

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

        for artist in [self.ship_patch, *self.sim_obs_patches, *self.path_line, self.goal_line]:
            if artist is not None:
                self.sim_ax.draw_artist(artist)

        self.sim_fig.canvas.blit(self.sim_fig.bbox)
        self.sim_fig.canvas.flush_events()
        # self.save(save_fig_dir, suffix, fig='sim')

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
