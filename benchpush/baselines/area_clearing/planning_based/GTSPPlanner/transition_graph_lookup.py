from shapely.geometry import Polygon, LineString, Point

import numpy as np

LIN_VEL  = 0.5
ANG_VEL = np.pi/4

def create_linestring_from_point(origin: tuple, angle_rad: float, distance = 0.1) -> LineString:
    """Create a small LineString from a point at a given angle and distance."""
    x1, y1 = origin

    # Compute endpoint using trigonometry
    x2 = x1 + distance * np.cos(angle_rad)
    y2 = y1 + distance * np.sin(angle_rad)

    return LineString([(x1, y1), (x2, y2)])

def compute_angle_between_lines(line1: LineString, line2: LineString) -> float:
    """Compute the angle (in degrees) between two LineStrings."""
    if len(line1.coords) < 2 or len(line2.coords) < 2:
        raise ValueError("Each LineString must have at least two points")

    # Get direction vectors (last two points)
    vec1 = np.array(line1.coords[-1]) - np.array(line1.coords[-2])
    vec2 = np.array(line2.coords[-1]) - np.array(line2.coords[-2])

    # Compute dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Compute angle in radians and convert to degrees
    angle_rad = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))

    return angle_rad

class TransitionGraphLookup:
    def __init__(self, robot_position, obs_to_push, obs_pushing_paths):
        self.robot_position = robot_position
        self.obs_to_push = obs_to_push
        self.obs_pushing_paths = obs_pushing_paths

        self.path_lookup = dict()
        self.length_lookup = dict()
        self.angle_lookup = dict()
        self.cost_lookup = dict()

    def _compute_transition_graph(self):
        # all paths from robot to obs to push
        robot_position = (self.robot_position[0], self.robot_position[1])
        robot_pose_line = create_linestring_from_point(robot_position, self.robot_position[2])
        for i in range(len(self.obs_to_push)):
            for push_path in self.obs_pushing_paths[i]:
                # compute the start path from robot to obs
                start = push_path.coords[0]
                path = LineString([robot_position, start])
                path_length = path.length
                path_angle = compute_angle_between_lines(robot_pose_line, path) + compute_angle_between_lines(path, push_path)
                
                self.path_lookup[(robot_position, start)] = path
                self.length_lookup[(robot_position, start)] = path_length
                self.angle_lookup[(robot_position, start)] = path_angle
                self.cost_lookup[(robot_position, start)] = LIN_VEL * path_length + ANG_VEL * path_angle

                # compute the end path from obs to robot
                end = push_path.coords[-1]
                path = LineString([end, robot_position])
                path_length = path.length
                path_angle = compute_angle_between_lines(push_path, path) + compute_angle_between_lines(path, robot_pose_line)

                self.path_lookup[(end, robot_position)] = path
                self.length_lookup[(end, robot_position)] = path_length
                self.angle_lookup[(end, robot_position)] = path_angle
                self.cost_lookup[(end, robot_position)] = LIN_VEL * path_length + ANG_VEL * path_angle

        # all paths from obs to push to obs to push
        for i in range(len(self.obs_to_push)):
            for j in range(len(self.obs_to_push)):
                if i != j:
                    for push_path in self.obs_pushing_paths[i]:
                        for next_push_path in self.obs_pushing_paths[j]:
                            # shortest line between end of path and start of next path
                            end = push_path.coords[-1]
                            start = next_push_path.coords[0]
                            path = LineString([end, start])
                            path_length = path.length
                            path_angle = compute_angle_between_lines(push_path, path) + compute_angle_between_lines(path, next_push_path)
                            
                            self.path_lookup[(end, start)] = path
                            self.length_lookup[(end, start)] = path_length
                            self.angle_lookup[(end, start)] = path_angle
                            self.cost_lookup[(end, start)] = LIN_VEL * path_length + ANG_VEL * path_angle

    @staticmethod
    def compute_transition_graph(robot_pose, obs_to_push, obs_pushing_paths):
        transition_graph = TransitionGraphLookup(robot_pose, obs_to_push, obs_pushing_paths)
        transition_graph._compute_transition_graph()

        return transition_graph