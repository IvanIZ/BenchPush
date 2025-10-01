import os
import time

import numpy as np

from benchpush.baselines.ship_ice_nav.planning_based.utils.path_evaluator import PredictivePathEvaluator
from benchpush.common.primitives import Primitives
from benchpush.common.ship import Ship
from benchpush.common.swath import generate_swath, view_all_swaths
from benchpush.common.utils.utils import Path
from benchpush.common.cost_map_occ import CostMap_Occupancy
from benchpush.baselines.ship_ice_nav.planning_based.utils.a_star_predictive import AStar_Predictive   
from benchpush.common.utils.utils import DotDict
import copy

# global vars for dir/file names
PLOT_DIR = 'plots'
PATH_DIR = 'paths'
METRICS_FILE = 'metrics.txt'


class PredictivePlanner():

    def __init__(self):

        # construct absolute path to the env_config folder
        cfg_file = os.path.join(os.path.dirname(__file__), 'planner_configs', 'lattice_config.yaml')
        self.cfg = DotDict.load_from_file(cfg_file)
        self.obs = None
    
    def plan(self, ship_pos, goal, occ_map, footprint, conc):
        """
        :param ship_pos: the 3-DOF ship pose (x, y, theta) in meter scale
        :param goal: the goal pose for planning. NOTE we only use the goal's y-value
        :param occ_map: occupancy map observation for planning
        :param footprint: footprint observation for planning
        :param conc: concentration value in decimals (should be among 0.1, 0.2, 0.3, 0.4, and 0.5)
        """


        ship_pos_scaled = np.array([ship_pos[0] * self.cfg.costmap.scale, ship_pos[1] * self.cfg.costmap.scale, ship_pos[2]])
        costmap = CostMap_Occupancy(cfg=self.cfg, horizon=self.cfg.a_star.horizon,
                        ship_mass=self.cfg.ship.mass, **self.cfg.costmap)

        ship = Ship(scale=self.cfg.costmap.scale, **self.cfg.ship)
        prim = Primitives(cache=False, **self.cfg.prim)
        swath_dict = generate_swath(ship, prim, cache=False,  model_inference=False)

        ship_no_padding = Ship(scale=self.cfg.costmap.scale, vertices=self.cfg.ship.vertices, padding=0, mass=self.cfg.ship.mass)
        swath_dict_no_padding = generate_swath(ship_no_padding, prim, cache=False, model_inference=True)
        # view_all_swaths(swath_dict_no_padding); exit()

        a_star = AStar_Predictive(cmap=costmap,
                ke_map=None,
                concentration=conc,
                prim=prim,
                ship=ship,
                swath_dict=swath_dict,
                swath_dict_no_padding=swath_dict_no_padding,
                ship_no_padding=ship_no_padding,
                use_ice_model=True,
                **self.cfg.a_star)

        path_eval = PredictivePathEvaluator(prim=prim, cmap_scale=self.cfg.costmap.scale, concentration=conc)
        path_obj = Path()

        # keep track of planner rate
        compute_time = []

        # start main planner loop
        print("Running predictive planning...")

        # start timer
        t0 = time.time()

        # stop planning if the remaining total distance is less than a ship length in meter
        if goal[1] - ship_pos[1] <= 2:
            return None

        # compute next goal NOTE: could be simplified in ROS version
        goal_y = goal[1] * self.cfg.costmap.scale

        # check if there is new obstacle information
        costmap.update(occ_map=occ_map)

        # compute path to goal
        ship_pos = copy.deepcopy(ship_pos_scaled)     # probably don't need it but just to be safe
        plan_start = time.time()
        search_result = a_star.search(
            start=(ship_pos[0], ship_pos[1], ship_pos[2]),
            goal_y=goal_y,
            occ_map=occ_map,
            centroids=None,
            footprint=footprint,
            ship_vertices=self.cfg.ship.vertices, 
            use_ice_model=True,
            debug=False, 
            prediction_horizon=None, 
        )
        print("planning time: ", time.time() - plan_start)

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

            # get swath costs for old path on new observation using predictive occ diff (NOTE only do this when this is not the first path)
            if path_obj.node_path is not None:

                old_swath_costs = path_eval.eval_path(occ_map=np.copy(occ_map), node_path=path_obj.node_path.T, ship_pose=ship_pos_scaled, 
                                                    swath_ins=path_obj.swath_ins, horizontal_shifts=path_obj.horizontal_shifts, ship_vertices=self.cfg.ship.vertices,
                                                    path_len_keys=path_obj.path_len_keys, debug=False)
                
                new_swath_costs = path_eval.eval_path(occ_map=np.copy(occ_map), node_path=node_path.T, ship_pose=ship_pos_scaled, 
                                                    swath_ins=swath_ins, horizontal_shifts=horizontal_shifts, ship_vertices=self.cfg.ship.vertices,
                                                    path_len_keys=path_len_keys, debug=False)

            else:
                old_swath_costs = None
                new_swath_costs = None
        
            drift_threshold = 0.25
            send_new_path, old_cost, new_cost = path_obj.update_occDiff(old_swath_costs=old_swath_costs, node_path=node_path.T, swath_costs=new_swath_costs, ship_pos=ship_pos_scaled,
                                            threshold_dist=self.cfg.get('threshold_dist', 0) * length,
                                            threshold_cost=self.cfg.get('threshold_cost'), 
                                            drift_threshold=drift_threshold)

        # update path object
        if send_new_path:
            path_obj.node_path = node_path
            path_obj.swath_ins = swath_ins
            path_obj.horizontal_shifts = horizontal_shifts
            path_obj.path_len_keys = path_len_keys
            path_obj.path = full_path

        # send path, return path in original scale
        # shape will be n x 3
        path_true_scale = np.c_[(path_obj.path[:2] / self.cfg.costmap.scale).T, path_obj.path[2]]  # TODO: confirm heading is ok

        compute_time.append((time.time() - t0))
        if send_new_path:
            return path_true_scale
        else:
            return None