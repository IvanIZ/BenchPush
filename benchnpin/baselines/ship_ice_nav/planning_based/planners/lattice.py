import os
import time
import numpy as np
import copy

from benchnpin.baselines.ship_ice_nav.planning_based.utils.a_star_search import AStar
from benchnpin.common.cost_map import CostMap
from benchnpin.common.primitives import Primitives
from benchnpin.common.ship import Ship
from benchnpin.common.swath import generate_swath, view_all_swaths
from benchnpin.common.utils.utils import Path
from benchnpin.common.utils.utils import DotDict


class LatticePlanner():

    def __init__(self, config_file=None):

        if config_file is None:
            config_file = 'lattice_config.yaml'

        # construct absolute path to the env_config folder
        cfg_file = os.path.join(os.path.dirname(__file__), 'planner_configs', config_file)
        self.cfg = DotDict.load_from_file(cfg_file)
        self.obs = None

    
    def plan(self, ship_pos, goal, obs):
        """
        :param ship_pos: the 3-DOF ship pose (x, y, theta) in meter scale
        :param goal: the goal pose for planning. NOTE we only use the goal's y-value
        :param obs: obstacles in the shape of (num_obstacles, num_vertices, 2). Note that this is a list
        """

        ship_pos_scaled = np.array([ship_pos[0] * self.cfg.costmap.scale, ship_pos[1] * self.cfg.costmap.scale, ship_pos[2]])

        costmap = CostMap(horizon=self.cfg.a_star.horizon,
                        ship_mass=self.cfg.ship.mass, **self.cfg.costmap)
        ship = Ship(scale=self.cfg.costmap.scale, **self.cfg.ship)
        prim = Primitives(cache=False, **self.cfg.prim)
        swath_dict = generate_swath(ship, prim, cache=False,  model_inference=False)
        # ship.plot(prim.turning_radius)
        # prim.plot()
        # view_all_swaths(swath_dict); exit()
        debug_ice_model = False

        ship_no_padding = Ship(scale=self.cfg.costmap.scale, vertices=self.cfg.ship.vertices, padding=0, mass=self.cfg.ship.mass)
        swath_dict_no_padding = generate_swath(ship_no_padding, prim, cache=False, model_inference=True)
        # view_all_swaths(swath_dict_no_padding); exit()

        a_star = AStar(cmap=costmap,
            prim=prim,
            ship=ship,
            swath_dict=swath_dict,
            swath_dict_no_padding=swath_dict_no_padding,
            ship_no_padding=ship_no_padding,
            use_ice_model=False,
            **self.cfg.a_star)

        path_obj = Path()

        # keep track of planner rate
        compute_time = []

        # Do planning here ================================================

        # start timer
        t0 = time.time()

        # stop planning if the remaining total distance is less than a ship length in meter
        if goal[1] - ship_pos[1] <= 2:
            return None

        # compute next goal NOTE: could be simplified in ROS version
        goal_y = goal[1] * self.cfg.costmap.scale

        # check if there is new obstacle information
        costmap.update(obs, ship_pos_scaled[1] - ship.max_ship_length / 2,
                    vs=(self.cfg.controller.target_speed * self.cfg.costmap.scale + 1e-8))

        # compute path to goal
        ship_pos = copy.deepcopy(ship_pos_scaled)     # probably don't need it but just to be safe
        search_result = a_star.search(
            start=(ship_pos[0], ship_pos[1], ship_pos[2]),
            goal_y=goal_y,
            occ_map=None,
            centroids=None,
            footprint=None,
            ship_vertices=self.cfg.ship.vertices, 
            use_ice_model=False,
            debug=debug_ice_model, 
            prediction_horizon=None, 
        )
        
        # fail to find path
        if not search_result:
            print("Planner failed to find a path!")
        else:
            
            # unpack result
            (full_path, full_swath), \
            (node_path, node_path_length), \
            (node_path_smth, new_nodes), \
            (nodes_expanded, g_score, swath_cost, length, edge_seq, swath_costs, swath_ins, horizontal_shifts, path_len_keys) = search_result
            x1, y1, _ = node_path  # this is the original node path prior to smoothing
            x2, y2 = new_nodes if len(new_nodes) != 0 else (0, 0)  # these are the nodes added from smoothing
            
            send_new_path = path_obj.update(full_path, full_swath, costmap.cost_map, ship_pos_scaled[1],
                                            threshold_dist=self.cfg.get('threshold_dist', 0) * length,
                                            threshold_cost=self.cfg.get('threshold_cost'))

            # send path, return path in original scale
            # shape will be n x 3
            path_true_scale = np.c_[(path_obj.path[:2] / self.cfg.costmap.scale).T, path_obj.path[2]]  # TODO: confirm heading is ok


        compute_time.append((time.time() - t0))

        if send_new_path:
            return path_true_scale
        else:
            return None
