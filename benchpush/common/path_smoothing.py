from typing import List

import dubins
import numpy as np
from skimage import draw

from benchpush.common.evaluation.metrics import euclid_dist


def path_smoothing(node_path: List, path_lengths: List, full_path, cost_map: np.ndarray,
                   step_size, ship_vertices, turning_radius: float,
                   add_nodes: int, max_dist: float, eps=1e-1):
    """
    Path smoothing algorithm described in the paper
    'Spatio-Temporal Lattice Planning Using Optimal Motion Primitives'
    """

    # generate a list where each value is an index corresponding to a segment between two nodes on the path
    # this will determine where to add new nodes, we use the lengths of the segments to bias the choice
    prob = np.asarray(path_lengths) / sum(path_lengths)
    segments = np.sort(np.random.choice(np.arange(len(node_path)), add_nodes, p=prob, replace=True))

    added_nodes = {}
    prev_idx = 0
    # get these new nodes
    for seg, count in zip(*np.unique(segments, return_counts=True)):
        # add new nodes
        l = sum(path_lengths[:seg + 1])
        idx = np.linspace(prev_idx, l / step_size, num=2 + count)
        prev_idx = l / step_size
        added_nodes[seg] = [full_path[int(i)] for i in idx[1:-1]]

    # insert nodes into node path
    new_node_path = []
    for idx, item in enumerate(node_path):
        new_node_path.append(item)
        if idx + 1 in added_nodes:
            new_node_path.extend(added_nodes[idx + 1])

    # search for best pairs of nodes
    prev_node = {}
    cost = {}
    for i, vi in enumerate(new_node_path):
        vi = tuple(vi)
        cost[vi] = 0 if i == 0 else np.inf
        prev_node[vi] = None

    for i, vi in enumerate(new_node_path[:-1]):
        flag = False  # this flag makes sure our sequence of nodes has no Nones
        for j, vj in enumerate(new_node_path[i + 1:]):
            # skip if too close or too far away
            dist = euclid_dist(vi, vj)
            if dist > max_dist and flag:
                break

            vi, vj = tuple(vi), tuple(vj)
            # determine cost between node vi and vj
            # epsilon handles small error from dubins package
            dubins_path = dubins.shortest_path(vi, vj, turning_radius - eps)
            path_len = dubins_path.path_length()

            swath_cost = 0
            configurations, _ = dubins_path.sample_many(1)  # rough paths here are fine
            for (x, y, theta) in configurations:
                R = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                rot_vi = np.array([[x], [y]]) + R @ ship_vertices.T
                rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :], shape=cost_map.shape)
                swath_cost += cost_map[rr, cc].sum()

            # compare cost of new pair
            curr_cost = swath_cost + path_len
            if cost[vi] + curr_cost < cost[vj]:
                cost[vj] = cost[vi] + curr_cost
                prev_node[vj] = vi
                flag = True

    # reconstruct path
    start, goal = tuple(node_path[0]), tuple(node_path[-1])
    smooth_node_path = [goal]
    node = goal
    while node != start:
        p_node, node = node, prev_node[node]
        smooth_node_path.append(node)
    smooth_node_path.reverse()  # we want start ~> goal

    # for plotting/debugging purposes
    added_x, added_y = [], []
    for v in added_nodes.values():
        added_x.extend([i[0] for i in v])
        added_y.extend([i[1] for i in v])

    # sample points along new path
    p_x, p_y, p_theta = [], [], []
    length = 0
    for p1, p2 in zip(smooth_node_path[:-1], smooth_node_path[1:]):
        dubins_path = dubins.shortest_path(p1, p2, turning_radius - eps)
        length += dubins_path.path_length()
        configurations, _ = dubins_path.sample_many(step_size)
        x = [item[0] for item in configurations]
        y = [item[1] for item in configurations]
        theta = [item[2] for item in configurations]
        p_x.extend(x)
        p_y.extend(y)
        p_theta.extend(theta)

    return np.asarray([p_x, p_y, p_theta]), smooth_node_path, (added_x, added_y), length

# TODO: fix weird loop in path bug
