""" Analytical heuristic for the dubins path from a configuration to a line segment. """
from typing import Tuple

import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def dubins_heuristic(q: Tuple[float, float, float], goal: float, r_min: float, boundary: Tuple[float, float] = None):
    """
    Return the length of the dubins path from initial configuration to a line segment.
    The line segment is characterized by a y-coord meaning it is parallel to the x-axis.

    A configuration is (x, y, theta), where theta is in radians, with zero
    along the line x = 0, and counter-clockwise is positive

    Possible cases when goal is below centre of tangent circle
    theta in [0, pi/2]    : theta - 0   : m = 1  : output [0, pi/2]
    theta in [pi/2, pi]   : theta - pi  : m = -1 : output [-pi/2, 0]
    theta in [pi, 3pi/2]  : theta - pi  : m = -1 : output [0, pi/2]
    theta in [3pi/2, 2pi) : theta - 2pi : m = 1  : output [-pi/2, 0]

    :param q: a configuration (x, y, theta)
    :param goal: y-coord of the goal line segment
    :param r_min: turning radius of the vehicle
    :param boundary: boundaries of the environment along the x-axis
    """
    (b0, b1) = (-1, -1) if boundary is None else boundary

    # trivial case - node is pas goal line segment
    if q[1] >= goal:
        h = 0
        x = q[0]
        theta = q[2]

    else:
        if q[2] <= np.pi / 2 or q[2] >= 3 * np.pi / 2:
            m = 1
        else:
            m = -1
        omega_y = q[1] + m * r_min * np.cos(q[2])

        if omega_y >= goal:
            if q[2] <= np.pi / 2:
                n = 0
            elif q[2] <= 3 * np.pi / 2:
                n = np.pi
            else:
                n = 2 * np.pi
            theta = m * np.arccos((omega_y - goal) / r_min) + n
            h = r_min * abs(q[2] - theta)
            x = q[0] - m * r_min * np.sin(q[2]) + m * np.sqrt(r_min ** 2 - (omega_y - goal) ** 2)
        else:
            theta = np.pi / 2
            h = r_min * min(abs(np.pi / 2 - q[2]), abs(5 * np.pi / 2 - q[2])) + goal - omega_y
            x = m * r_min * (1 - np.sin(q[2])) + q[0]

        if b0 != -1 and (b0 > x or x > b1):
            if 0 <= q[2] <= np.pi:
                h = np.inf
            else:
                omega_y = q[1] - (omega_y - q[1])
                omega_x = q[0] + m * r_min * np.sin(q[2])
                if b0 > omega_x or omega_x > b1:
                    h = np.inf
                else:
                    theta = np.pi / 2
                    h = r_min * max(abs(np.pi / 2 - q[2]), abs(5 * np.pi / 2 - q[2])) + goal - omega_y
                    x = -m * r_min * (1 - np.sin(q[2])) + q[0]
                    if b0 > x or x > b1:
                        h = np.inf

    return h, (x, goal, theta)
