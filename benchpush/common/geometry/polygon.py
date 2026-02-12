import random
from typing import Union, List

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LinearRing
from skimage import draw
from shapely.geometry import Polygon, Point, LineString

__all__ = [
    'poly_radius',
    'poly_area',
    'poly_centroid',
    'generate_polygon',
    'radius_of_gyration_squared',
    'generate_body_points_polygon'
]


def poly_radius(vertices, centre_pos) -> float:
    dist = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return max(dist(v, centre_pos) for v in vertices)


def poly_area(vertices):
    # area of solid simple polygon given set of vertices (shape is n x 2)
    # https://en.wikipedia.org/wiki/Polygon#Simple_polygons
    x, y = vertices.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def poly_centroid(vertices):
    # centroid for solid simple polygon given set of vertices (shape is n x 2)
    # https://en.wikipedia.org/wiki/Polygon#Centroid
    x, y = vertices.T
    A = poly_area(vertices)
    u = x * np.roll(y, 1) - np.roll(x, 1) * y
    return np.abs((
        1 / (6 * A) * np.dot(x + np.roll(x, 1), u),
        1 / (6 * A) * np.dot(y + np.roll(y, 1), u)
    ))


def radius_of_gyration_squared(vertices, scale=100):
    # radius of gyration squared about the axis of rotation
    # https://en.wikipedia.org/wiki/Radius_of_gyration
    rr, cc = draw.polygon(vertices[:, 1] * scale, vertices[:, 0] * scale)
    x, y = poly_centroid(vertices)

    return sum((rr / scale - y) ** 2 + (cc / scale - x) ** 2) / len(rr)


def generate_polygon(diameter, origin=(0, 0), num_vertices_range=(10, 20), circular=False) -> Union[None, np.ndarray]:
    """
    Implements the random convex polygon generation algorithm from
    https://cglab.ca/~sander/misc/ConvexGeneration/convex.html

    :param diameter: diameter of the circle that the polygon is inscribed in
    :param origin: origin of the polygon
    :param num_vertices_range: range of number of vertices in the polygon
    :param circular: if True, generates a circular-like polygon
    """
    if circular:
        # for more circular-like obstacles
        x = np.cos(np.arange(0, 2 * np.pi, 0.1)) * diameter / 2
        y = np.sin(np.arange(0, 2 * np.pi, 0.1)) * diameter / 2
        num_vertices = len(x)

    else:
        # generate two lists of x and y of N random integers between 0 and n
        # sample a random number for number of vertices
        num_vertices = random.randint(*num_vertices_range)
        x = [random.uniform(0, diameter) for _ in range(num_vertices)]
        y = [random.uniform(0, diameter) for _ in range(num_vertices)]

    # sort both lists
    x.sort()
    y.sort()

    x_max = x[-1]
    y_max = y[-1]
    x_min = x[0]
    y_min = y[0]

    last_top = x_min
    lastBot = x_min
    x_vec = []

    for i in range(1, num_vertices - 1):
        val = x[i]
        if bool(random.getrandbits(1)):
            x_vec.append(val - last_top)
            last_top = val
        else:
            x_vec.append(lastBot - val)
            lastBot = val

    x_vec.append(x_max - last_top)
    x_vec.append(lastBot - x_max)

    last_left = y_min
    last_right = y_min
    y_vec = []

    for i in range(1, num_vertices - 1):
        val = y[i]
        if bool(random.getrandbits(1)):
            y_vec.append(val - last_left)
            last_left = val
        else:
            y_vec.append(last_right - val)
            last_right = val

    y_vec.append(y_max - last_left)
    y_vec.append(last_right - y_max)
    random.shuffle(y_vec)

    pairs = zip(x_vec, y_vec)
    sorted_pairs = sorted(pairs, key=lambda pair: np.arctan2(pair[0], pair[1]))

    min_polygon_x = 0
    min_polygon_y = 0
    x = 0
    y = 0
    points = []

    for pair in sorted_pairs:
        points.append((x, y))
        x += pair[0]
        y += pair[1]
        min_polygon_x = min(min_polygon_x, x)
        min_polygon_y = min(min_polygon_y, y)

    x_shift = x_min - min_polygon_x
    y_shift = y_min - min_polygon_y

    points = np.asarray(points) + np.array([x_shift, y_shift]).T

    # find the centroid of polygon
    centre_pos = poly_centroid(points)

    # make centre of polygon at the origin
    points -= centre_pos - np.asarray(origin)  # assumes origin is in the 1st quadrant (positive x,y)

    # n x 2 array where each element is a vertex (x, y)
    return points


def generate_body_points_polygon(vertices: Union[List, np.ndarray], spacing, plot=False):
    """
    Generates equally spaced points on the boundary of a polygon.
    Body points are described in the paper "CHOMP: Covariant Hamiltonian optimization for motion planning"
    """
    body_points = []

    linear_ring = LinearRing(vertices)  # Convert polygon to linear ring
    perimeter = linear_ring.length

    t = np.linspace(0, perimeter, int(perimeter / spacing))

    for ti in t:
        point = linear_ring.interpolate(ti)  # Interpolate points on the linear ring
        x, y = point.xy
        body_points.append((x[0], y[0]))

    if plot:
        print('Number of boundary points:', len(body_points))
        plt.plot(*np.asarray(body_points).T, 'r.')
        plt.gca().add_patch(plt.Polygon(vertices, True, fill=False, linewidth=2))
        plt.gca().set_aspect('equal')
        plt.show()

    return np.asarray(body_points)

def shrink_or_swell_polygon(vertices, factor=0.10, swell=False, plot=False):
    """
    returns the shapely polygon which is smaller or bigger by passed factor.
    If swell = True , then it returns bigger polygon, else smaller

    Original code from https://stackoverflow.com/a/67205583
    """
    polygon = Polygon(vertices)
    polygon_resized = _shrink_swell_helper(polygon, factor, swell)

    if polygon_resized.area <= 0:
        return vertices  # return original

    if type(polygon_resized) is not Polygon:
        return vertices  # return original

    if plot:
        # visualize for debugging
        x, y = polygon.exterior.xy
        plt.plot(x, y)
        x, y = polygon_resized.exterior.xy
        plt.plot(x, y)
        plt.axis('equal')
        plt.show()

    return np.asarray(polygon_resized.exterior.xy).T

def create_polygon_from_line(line_vertices, width=0.2):
    """
    Create a polygon from a line with a given width.
    """
    line = LineString(line_vertices)
    polygon = line.buffer(width / 2, cap_style=2, join_style=2)
    return np.asarray(polygon.exterior.coords)


def _shrink_swell_helper(polygon: Polygon, factor: float, swell: bool = False) -> Polygon:
    xs = list(polygon.exterior.coords.xy[0])
    ys = list(polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * factor

    if swell:
        polygon_resized = polygon.buffer(shrink_distance)  # expand
    else:
        polygon_resized = polygon.buffer(-shrink_distance)  # shrink

    return polygon_resized
