from typing import Tuple

import dubins
import numpy as np

from benchpush.common.utils.utils import heading_to_world_frame


def compute_turn_rate_control(path, speed):
    assert path.shape[0] == 3

    # compute the time derivative of theta
    arc_length = np.linalg.norm(np.diff(path[:2, :], axis=1), axis=0)
    theta = np.unwrap(path[2, :])
    theta_dot = np.diff(theta) / arc_length * speed

    return theta_dot


def apply_control(start, V, U, dt, rk4=False):
    X = np.zeros([3, len(U) + 1])
    X[:, 0] = start

    f = lambda x, u: np.asarray([V * np.cos(x[2]), V * np.sin(x[2]), u])  # dx/dt = f(x,u)

    for i in range(len(U)):
        if rk4:
            k1 = f(X[:, i], U[i])
            k2 = f(X[:, i] + dt / 2 * k1, U[i])
            k3 = f(X[:, i] + dt / 2 * k2, U[i])
            k4 = f(X[:, i] + dt * k3, U[i])
            x_next = X[:, i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            x_next = X[:, i] + dt * f(X[:, i], U[i])

        X[:, i + 1] = x_next

    return X


def get_points_on_dubins_path(p1: Tuple, p2: Tuple, num_headings: int, initial_heading: float,
                              turning_radius: float, eps: float = 0., step_size: float = 0.5):
    theta_0 = heading_to_world_frame(p1[2], initial_heading, num_headings)
    theta_1 = heading_to_world_frame(p2[2], initial_heading, num_headings)
    dubins_path = dubins.shortest_path((p1[0], p1[1], theta_0),
                                       (p2[0], p2[1], theta_1),
                                       turning_radius - eps)
    configurations, _ = dubins_path.sample_many(step_size)
    x = [item[0] for item in configurations]
    y = [item[1] for item in configurations]
    theta = [item[2] for item in configurations]

    return x, y, theta, dubins_path.path_length()
