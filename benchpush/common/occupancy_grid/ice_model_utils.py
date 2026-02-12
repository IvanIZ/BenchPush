import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.draw import draw

# define an arbitrary max cost applied to a cell in the costmap
MAX_COST = 1e10


def compute_ship_footprint_planner(node, theta_0, ship_vertices, occ_map_height, occ_map_width, scale, padding=0.25):
    """
    NOTE this function computes current ship footprint similarily to self.compute_ship_footprint()
    but is intended for generating observations for planners 
    :param node: a lattice node (x, y, theta) where x, y are in grid unit and theta in [0, 7]
    :param ship_vertices: original unscaled, unpadded ship vertices
    """
    footprint = np.zeros((occ_map_height, occ_map_width))
    angle = node[2] * (np.pi / 4)
    angle += theta_0
    # # apply padding as in a* search
    # ship_vertices = np.asarray(
    #     [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in ship_vertices]
    # )

    # ship vertices in meter
    heading = angle
    R = np.asarray([
        [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
    ])
    vertices = np.asarray(ship_vertices) @ R.T

    r = []
    c = []
    for x, y in vertices:
        grid_x = x * scale + node[0]
        grid_y = y * scale + node[1]
        if grid_y < 0 or grid_y >= occ_map_height or grid_x < 0 or grid_x >= occ_map_width:
            continue
        r.append(grid_y)
        c.append(grid_x)

    if len(r) == 0 or len(c) == 0:
        return footprint

    rr, cc = draw.polygon(r=r, c=c, shape=footprint.shape)

    # it is possible that the ship state is outside of the grid map
    # assert len(rr) > 0 and len(cc) > 0, print("Invalid ship position! Likely outside of cost map: ", node)
    
    if len(rr) > 0 and len(cc) > 0:
        footprint[rr, cc] = 1.0
    return footprint


def crop_window(grid_map, node, win_width, win_height, horizontal_shift, vertical_shift):
    """
    NOTE: this is a helper function for planners
    :param horizontal_shift: shift in grid unit
    :param node: ship state (x, y, theta) where x, y is in cost map scale, NOT METER!
    :param vertical_shift: number of grids the lower bound below node[1]
    """
    assert horizontal_shift < win_width, print("Invalid horizontal shift value!", horizontal_shift, win_width, node, grid_map.shape)

    x = int(node[0])
    # x = round(node[0])
    y = int(node[1])
    cropped_window = np.zeros((win_height, win_width))

    y_low_map = y - vertical_shift

    x_low_map = x - horizontal_shift
    x_high_map = x_low_map + win_width
    assert x_low_map >= 0, print("Invalid horizontal shift x_low: ", x, x_low_map, horizontal_shift)
    assert x_high_map <= grid_map.shape[1], print("Invalid horizontal shift x_high: ", x, x_high_map, horizontal_shift, grid_map.shape[1])
    x_low_win = 0
    x_high_win = win_width

    y_low_win = 0
    assert y_low_map >= 0, print("cropping y low bound negative! ", y_low_map)

    y_high_map = y_low_map + win_height
    if y_high_map > grid_map.shape[0]:
        y_gap = y_high_map - grid_map.shape[0]
        y_high_win = win_height - y_gap
        y_high_map = grid_map.shape[0]
    else:
        y_high_win = win_height

    assert (y_high_map - y_low_map) == (y_high_win - y_low_win), print("y-dim not same size!", y_low_map, y_high_map, y_low_win, y_high_win)
    assert (x_high_map - x_low_map) == (x_high_win - x_low_win), print("x-dim not same size!", x_low_map, x_high_map, x_low_win, x_high_win)

    assert x_low_map <= grid_map.shape[1] - 1, print("Something wrong with x_low_map", x_low_map)

    if x_low_map > grid_map.shape[1] - 1:
        return cropped_window, None, None, None, None, None, None, None, None

    cropped_window[y_low_win:y_high_win, x_low_win:x_high_win] = grid_map[y_low_map:y_high_map, x_low_map:x_high_map]
    return cropped_window, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win


def stitch_window(grid_map, window, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win):
    """
    NOTE: this is a helper function for planners
    """

    if x_low_map is None:
        return grid_map
    
    grid_map_new = np.copy(grid_map)
    grid_map_new[y_low_map:y_high_map, x_low_map:x_high_map] = window[y_low_win:y_high_win, x_low_win:x_high_win]
    return grid_map_new



def encode_swath(swath_input, grid_map, node, win_width=36, win_height=36, max_val=29, vertical_shift=0):
    """
    :param swath_input: numpy array of swath points of shape (n_points, 2)
    :param vertical_shift: number of grids the lower bound below node[1], In here, this means number of grids to shift up the swath
    """

    x = int(node[0])
    # x = round(node[0])
    y = int(node[1])

    win_mid = win_width // 2

    swath = np.copy(swath_input)

    # shift down and left s.t. swath starts at (0, 0)
    swath[:, 0] = swath[:, 0] - max_val + vertical_shift
    swath[:, 1] = swath[:, 1] - max_val

    # get swath horizontal middle point and compute horizontal shift
    horizontal_mid = np.mean(swath[:, 1])
    horizontal_shift = int(win_mid - horizontal_mid)
    original_shift = horizontal_shift

    # modify horizontal shift s.t. the shifted window stays within the environment
    x_low_map = x - horizontal_shift
    x_high_map = x_low_map + win_width
    if x_low_map < 0:           # too much to the left --> reduce h.s.
        x_gap = abs(x_low_map)
        horizontal_shift = horizontal_shift - x_gap

    elif x_high_map > grid_map.shape[1]:        # too much to the right --> increase h.s.
        x_gap = x_high_map - grid_map.shape[1]
        horizontal_shift = horizontal_shift + x_gap

    assert horizontal_shift >= 0, print("horizontal shift negative: ", horizontal_shift, "end\n")

    assert horizontal_shift < win_width, print("Invalid horizontal shift value!", horizontal_shift, original_shift, win_width, node, 
                                               (grid_map.shape[1], grid_map.shape[0]))


    # apply shift s.t. swath is horizontally centerd
    swath[:, 1] = swath[:, 1] + horizontal_shift

    # encode swath
    swath_win = np.zeros((win_height, win_width))
    for y, x in swath:
        if y < 0 or y > swath_win.shape[0] - 1:
            continue
        
        # it's possible a that part of the swath is outside of the environment, if the node is on the environment edge, ship head sticks out
        if x < 0 or x > swath_win.shape[1] - 1:
            continue

        swath_win[y, x] = 1.0

    return swath_win, horizontal_shift


def view_swath(swath_input, max_val):
    """
    View the entire swath before encode_swath
    :param swath_input: numpy array of swath points of shape (n_points, 2)
    """

    swath_vis = np.zeros((max_val * 2, max_val * 2))
    for y, x in swath_input:
        assert y >= 0 and y <= max_val * 2 - 1, print("Swath input y value invalid, ", y, max_val)
        assert x >= 0 and x <= max_val * 2 - 1, print("Swath input x value invalid, ", x, max_val)

        swath_vis[y, x] = 1.0

    return swath_vis



def update_costmap(original_shape, occ_map, lower_lim, upper_lim):
    costmap = np.zeros(original_shape)
    costmap[:occ_map.shape[0]] = occ_map
    costmap = costmap[lower_lim:upper_lim]
    return costmap


def boundary_cost(cost_map, margin):
    cost_map[:, :margin] = MAX_COST
    cost_map[:, -margin:] = MAX_COST
    return cost_map


def get_boundary_map(original_shape, margin):
    """
    A boundary cost map where at the boundary is a high cost, elsewhere is uniformly 0
    """
    boundary_map = np.zeros(original_shape)
    boundary_map = boundary_cost(cost_map=boundary_map, margin=margin)
    return boundary_map
