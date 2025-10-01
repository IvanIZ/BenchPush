from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib import patches


class Ship:
    def __init__(self, scale: float, vertices: List, padding: float = 0, mass: float = 1):
        """
        :param scale: scale factor for vertices
        :param vertices: list of vertices that define the ship. Assumes vertices are defined symmetrically
                         about the origin (0, 0) and the ship is facing right (i.e. along the x-axis)
        :param padding: padding to add to the vertices
        :param mass: mass of the ship
        """
        self.vertices = np.asarray(
            [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in vertices]
        ) * scale
        dist = lambda a, b: np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        self.max_ship_length = np.ceil(max(dist(a, b) for a in self.vertices for b in self.vertices)).astype(int)
        assert self.max_ship_length != 0, 'ship length cannot be 0'
        self.mass = mass
        self.width = self.vertices[:, 1].max() - self.vertices[:, 1].min()
        self.right_half, self.left_half = self.split_vertices()  # mainly used in the swath generation step

    def plot(self, turning_radius, heading=np.pi/2):
        start_pos = (0, 0, heading)

        x, y, theta = start_pos
        R = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        f, ax = plt.subplots()
        # plot ship as polygon
        rot_vertices = self.vertices @ R.T + start_pos[:2]
        ax.add_patch(patches.Polygon(rot_vertices, True, fill=True, zorder=10, alpha=0.5))

        # plot the vertices
        for v in rot_vertices:
            ax.plot(v[0], v[1], 'k.')

        # plot line to show initial heading
        ax.plot([x, x + self.max_ship_length * np.cos(theta)],
                [y, y + self.max_ship_length * np.sin(theta)], 'r')

        # plot lines to show minimal turning radius
        a = np.arange(0, 2 * np.pi, 0.01)
        ax.plot((x - turning_radius * np.sin(theta) + turning_radius * np.cos(a)).tolist(),
                (y + turning_radius * np.cos(theta) + turning_radius * np.sin(a)).tolist(), 'g')
        ax.plot((x + turning_radius * np.sin(theta) + turning_radius * np.cos(a)).tolist(),
                (y - turning_radius * np.cos(theta) + turning_radius * np.sin(a)).tolist(), 'g')

        # plot the radius
        ax.plot([x, x - turning_radius * np.sin(theta)],
                [y, y + turning_radius * np.cos(theta)], 'b')

        ax.plot([x, x + turning_radius * np.sin(theta)],
                [y, y - turning_radius * np.cos(theta)], 'b')

        # plot the width
        ax.plot([x - self.width / 2 * np.cos(theta + np.pi / 2), x + self.width / 2 * np.cos(theta + np.pi / 2)],
                [y - self.width / 2 * np.sin(theta + np.pi / 2), y + self.width / 2 * np.sin(theta + np.pi / 2)],
                'k--', linewidth=2, zorder=20)

        # plot circle of radius width / 2
        ax.plot(x + self.width / 2 * np.cos(a + np.pi / 2),
                y + self.width / 2 * np.sin(a + np.pi / 2), 'k--', linewidth=2, zorder=20)

        ax.set_aspect('equal')

        plt.show()

    @staticmethod
    def sim(vertices: List, start_pos: Tuple[float, float, float], body_type=None, velocity=(0, 0), color=None):
        from pymunk import Vec2d, Body, Poly, moment_for_box

        x, y, theta = start_pos
        # setup for pymunk
        if body_type is None:
            body_type = Body.KINEMATIC  # Body.DYNAMIC
        body = Body(body_type=body_type)  # mass and moment ignored when kinematic body type
        body.position = (x, y)
        body.velocity = Vec2d(*velocity)
        body.angle = theta  # Rotation of the body in radians
        shape = Poly(body, [tuple(item) for item in vertices], radius=0.02)
        # shape.mass = 100
        shape.mass = 10.0
        shape.elasticity = 0.01
        shape.friction = 1.0

        if color is not None:
            shape.color = color

        return body, shape

    @staticmethod
    def calc_turn_radius(rate, speed):
        """
        rate: deg/min
        speed: knots, 1 knot = 30.8667 metre / minute
        """
        theta = rate * np.pi / 180  # convert to rads
        s = speed * 30.8667  # convert to m
        return s / theta

    def split_vertices(self, debug=False):
        new_vertices = [[0, self.width / 2], [0, -self.width / 2]]
        right_half = np.concatenate((self.vertices[self.vertices[:, 0] >= 0], new_vertices))
        left_half = np.concatenate((self.vertices[self.vertices[:, 0] <= 0], new_vertices))

        # ensure ordering of vertices is correct
        right_half = right_half[ConvexHull(right_half).vertices]
        left_half = left_half[ConvexHull(left_half).vertices]

        if debug:
            f, ax = plt.subplots()
            ax.add_patch(patches.Polygon(right_half, True, fill=True, color='r', alpha=0.5))
            ax.add_patch(patches.Polygon(left_half, True, fill=True, color='b', alpha=0.5))
            # plot the vertices
            for v in right_half:
                ax.plot(v[0], v[1], 'r.')
            for v in left_half:
                ax.plot(v[0], v[1], 'b.')
            ax.set_aspect('equal')
            plt.show()

        return right_half, left_half


if __name__ == '__main__':
    # run main to test ship config
    ship = Ship(
        padding=0.15,
        scale=1,
        vertices=[[1., -0.],
                  [0.9, 0.10],
                  [0.5, 0.25],
                  [-1., 0.25],
                  [-1., -0.25],
                  [0.5, -0.25],
                  [0.9, -0.10]]
    )
    ship.plot(turning_radius=2)
