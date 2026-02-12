import logging
import os
from operator import ge, le
from typing import Union

import matplotlib.ticker as tkr
import numpy as np
import yaml
from scipy.interpolate import interp1d
from yaml import Loader

M__2_PI = 2 * np.pi


def heading_to_world_frame(heading: int, theta_0: float, num_headings: int):
    """
    :param heading: ordinal or cardinal heading from ships frame of reference
    :param theta_0: angle between ship and fixed/world coordinates
    :param num_headings: number of headings in the discretized heading space
    """
    return (heading * M__2_PI / num_headings + theta_0) % M__2_PI


def resample_path(path, step_size):
    # edge case, path is of shape N x 3
    if len(path) == 1:
        return path

    # resample
    t = [0, *np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))]
    f_x = interp1d(t, path[:, 0])
    f_y = interp1d(t, path[:, 1])

    t_new = np.arange(t[0], t[-1], step_size)
    path = np.asarray([f_x(t_new), f_y(t_new)]).T

    # compute the heading along path
    theta = np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))

    # transform path to original scaling and then return
    return np.c_[path[:-1], theta]


class Path:
    def __init__(self, path: np.ndarray = None, swath: np.ndarray = None, node_path=None, swath_costs=None, swath_ins=None, 
                 horizontal_shifts=None, path_len_keys=None):
        self.path = path  # shape is 3 x n and order is start -> end
        self.swath = swath
        self.node_path = node_path
        self.swath_costs = swath_costs
        
        # path eval info
        self.swath_ins = swath_ins
        self.horizontal_shifts = horizontal_shifts
        self.path_len_keys = path_len_keys


    def update(self, path, swath, costmap, ship_pos_y, threshold_dist=None, threshold_cost=0.95):
        if self.path is None:
            self.path = path
            self.swath = swath
            return True

        if not threshold_dist or (self.path[1][-1] - ship_pos_y) < threshold_dist:
            self.path = path
            self.swath = swath
            return True

        # compute cost for new path up to old path max y
        new_swath = swath.copy()
        new_swath[int(self.path[1][-1]):] = 0
        new_swath[:int(ship_pos_y)] = 0
        new_cost = costmap[new_swath].sum()  # + path_length(self.clip_path(path, self.path[1][-1], le)[:2].T)

        # compute cost of old path starting from current ship pos y
        old_swath = self.swath.copy()
        old_swath[int(self.path[1][-1]):] = 0
        old_swath[:int(ship_pos_y)] = 0
        old_cost = costmap[old_swath].sum()  # + path_length(self.clip_path(self.path, ship_pos_y, ge)[:2].T)

        # check if the new path is better than the old path by some threshold
        # from experiments we get better performance if we only consider
        # the swath cost in the path cost comparison rather than swath cost + path length
        if new_cost < old_cost * threshold_cost:
            self.path = path
            self.swath = swath
            return True

        return False
    

    def update_occDiff(self, old_swath_costs, node_path, swath_costs, ship_pos, threshold_dist=None, 
                       threshold_cost=0.95, costmap_scale=5, drift_threshold=0.5):
        """
        node_path shape: (n_nodes, 3)
        """
        ship_pos_y = ship_pos[1]
        
        # new path has only one node --> start node is already at goal
        if len(node_path) <= 1:
            return False, None, None
        
        # first time planning
        if self.node_path is None:
            return True, None, None

        # exceed distance threshold
        if not threshold_dist or (self.path[1][-1] - ship_pos_y) < threshold_dist:
            return True, None, None
        
        # compute horizontal drift
        horizontal_drift = self.compute_horizontal_drift(node_path=node_path, ship_pos=ship_pos)
        horizontal_drift = horizontal_drift / costmap_scale         # convert drift to meter scale
        if horizontal_drift > drift_threshold:
            return False, None, None
        
        # compute cost for new path from ship pose to old path max y
        old_path_max_y = int(self.path[1][-1])

        # print("ship pose y: ", ship_pos_y, "; old path max y: ", old_path_max_y, "; node path shape: ", node_path.shape)
        new_cost = self.compute_subpath_cost(node_path=node_path, swath_costs=swath_costs, start_y=ship_pos_y, end_y=old_path_max_y)

        # compute cost for old path from ship pose to old path max y
        old_cost = self.compute_subpath_cost(node_path=self.node_path.T, swath_costs=old_swath_costs, start_y=ship_pos_y, end_y=old_path_max_y)

        # check if the new path is better than the old path by some threshold
        if new_cost < old_cost * threshold_cost:
            return True, old_cost, new_cost

        return False, old_cost, new_cost
    

    def compute_horizontal_drift(self, node_path, ship_pos):
        """
        This function return the closest distance between the ship pos and the node path (i.e. the new path)
        :param node_path: node path of the NEW path;  requires shape: (n_nodes, 3)
        :param ship_pos: current ship position in cost map scale (x, y)
        """

        for i in range(node_path.shape[0] - 1):
            node_src = node_path[i]
            node_target = node_path[i + 1]

            # find the segment in which the current ship pose is in between
            if ship_pos[1] >= node_src[1] and ship_pos[1] <= node_target[1]:
                return self.closest_distance(node_src=node_src, node_target=node_target, ship_pos=ship_pos)


    def closest_distance(self, node_src, node_target, ship_pos):
        # Convert points to numpy arrays for vector operations
        p1 = np.array(node_src)
        p2 = np.array(node_target)
        p3 = np.array(ship_pos)

        # Vector from p1 to p2
        line_vec = p2 - p1
        
        # Vector from p1 to p3
        p1_to_p3_vec = p3 - p1
        
        # Project vector from p1 to p3 onto the line vector
        line_len_squared = np.dot(line_vec, line_vec)
        if line_len_squared == 0:
            # p1 and p2 are the same point, so just return the distance from p1 to p3
            return np.linalg.norm(p1_to_p3_vec)
        
        # Calculate projection scalar
        t = np.dot(p1_to_p3_vec, line_vec) / line_len_squared
        
        # Clamp t to the [0, 1] range to ensure the closest point is on the segment
        t = max(0, min(1, t))
        
        # Find the closest point on the line segment
        closest_point = p1 + t * line_vec
        
        # Compute the distance from ship_pos to the closest point on the line segment
        distance = np.linalg.norm(p3 - closest_point)
        
        return distance


    
    def compute_subpath_cost(self, node_path, swath_costs, start_y, end_y):
        """
        Given a path as node_path, and a list of swath costs corrsponding to each swath in the path, 
        compute the cost of the subpath from start_y to end_y
        :param node_path: requires shape: (n_nodes, 3)
        """
        cost = 0
        for i in range(node_path.shape[0] - 1):
            node_src = node_path[i]
            node_target = node_path[i + 1]

            # entire swath below ship pose
            if start_y > node_target[1]:
                continue

            # entire swath above ship pose max y
            if end_y < node_src[1]:
                continue

            # handle start y in between this swath
            if start_y >= node_src[1] and start_y <= node_target[1]:
                src_mid_cost, mid_target_cost = self.interpolate_cost(node_src=node_src, node_target=node_target, y_mid=start_y, swath_cost=swath_costs[i])
                swath_cost = mid_target_cost

            # handle end y in between this swath
            elif end_y >= node_src[1] and end_y <= node_target[1]:
                src_mid_cost, mid_target_cost = self.interpolate_cost(node_src=node_src, node_target=node_target, y_mid=end_y, swath_cost=swath_costs[i])
                swath_cost = src_mid_cost

            else:
                swath_cost = swath_costs[i]

            cost += swath_cost
        
        return cost


    def interpolate_cost(self, node_src, node_target, y_mid, swath_cost):
        """
        Takes as input a source node, a target node, the cost from source to target, and an intermediate y coordinate
        Compute by linear interpolation the cost from source to mid y, and cost from mid y to target node
        NOTE: assume y_mid is between node_src[1] and node_target[1]
        """
        assert len(node_src) == 3, print("invalid node_src: ", node_src)
        assert len(node_target) == 3, print("invalid node_target: ", node_target)

        x_src, y_src, _ = node_src
        x_target, y_target, _ = node_target

        # NOTE now only interpolate on y-axis. Could consider interpolate on distance
        dist_src_target = abs(y_target - y_src)
        dist_src_mid = abs(y_mid - y_src)
        dist_mid_target = abs(y_target - y_mid)

        src_mid_cost = (dist_src_mid / dist_src_target) * swath_cost
        mid_target_cost = (dist_mid_target / dist_src_target) * swath_cost

        return src_mid_cost, mid_target_cost
    

    @staticmethod
    def clip_path(path, ship_pos_y: float, op: Union[ge, le] = ge):
        # clip points along path that are less/greater than ship y position
        return path[..., op(path[1], ship_pos_y)]


def rotation_matrix(theta) -> np.ndarray:
    return np.asarray([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    Class used to store configuration parameters
    """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, attr):
        if attr not in self.keys():
            raise AttributeError

        return self.get(attr)

    @staticmethod
    def to_dict(d):
        return {
            k: DotDict.to_dict(d[k]) if type(v) is DotDict else v
            for k, v in d.items()
        }

    @staticmethod
    def to_dot_dict(d):
        return DotDict({
            k: DotDict.to_dot_dict(d[k]) if type(v) is dict else v
            for k, v in d.items()
        })

    @staticmethod
    def load_from_file(fp):
        with open(fp, 'r') as fd:
            cfg = yaml.load(fd, Loader=Loader)

        # convert to a DotDict
        return DotDict.to_dot_dict(cfg)


def setup_logger(output=None, console=True, name="", level=logging.DEBUG, prefix=""):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create formatter
    formatter = logging.Formatter(
        prefix + "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if output:
        fh = logging.StreamHandler(open(os.path.join(output, "log.txt"), 'w'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def scale_axis_labels(axes, scale):
    # divide axis labels by scale
    # thank you, https://stackoverflow.com/a/27575514/13937378
    def numfmt(x, pos):
        s = '{}'.format(x / scale)
        return s

    yfmt = tkr.FuncFormatter(numfmt)  # create your custom formatter function

    for ax in axes:
        ax.yaxis.set_major_formatter(yfmt)
        ax.xaxis.set_major_formatter(yfmt)
