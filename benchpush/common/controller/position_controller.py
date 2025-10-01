import numpy as np

# from benchpush.environments.area_clearing.utils import position_to_pixel_indices, pixel_indices_to_position

from skimage.draw import line
from skimage.measure import approximate_polygon

import spfa

# Helper functions

def restrict_heading_range(heading):
    return np.mod(heading + np.pi, 2 * np.pi) - np.pi

def distance(position1, position2):
    return np.linalg.norm(np.asarray(position1)[:2] - np.asarray(position2)[:2])

def heading_difference(heading1, heading2):
    return restrict_heading_range(heading1 - heading2)

def position_to_pixel_indices(x, y, image_shape, local_map_pixels_per_meter):
    pixel_i = np.floor(image_shape[0] / 2 - y * local_map_pixels_per_meter).astype(np.int32)
    pixel_j = np.floor(image_shape[1] / 2 + x * local_map_pixels_per_meter).astype(np.int32)
    pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
    pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
    return pixel_i, pixel_j

def pixel_indices_to_position(pixel_i, pixel_j, image_shape, local_map_pixels_per_meter):
    x = (pixel_j - image_shape[1] / 2) / local_map_pixels_per_meter
    y = (image_shape[0] / 2 - pixel_i) / local_map_pixels_per_meter
    return x, y

class PositionController:
    def __init__(self, cfg, robot_radius, map_width, map_height, 
                 configuration_space, configuration_space_thin, closest_cspace_indices,
                 local_map_pixel_width, local_map_width, local_map_pixels_per_meter,
                 turn_step_size, move_step_size, waypoint_moving_threshold, waypoint_turning_threshold):
        self.cfg = cfg
        self.robot_radius = robot_radius

        self.map_width = map_width
        self.map_height = map_height

        self.configuration_space = configuration_space
        self.configuration_space_thin = configuration_space_thin
        self.closest_cspace_indices = closest_cspace_indices

        self.local_map_pixel_width = local_map_pixel_width
        self.local_map_width = local_map_width
        self.local_map_pixels_per_meter = local_map_pixels_per_meter
        self.turn_step_size = turn_step_size
        self.move_step_size = move_step_size
        self.waypoint_moving_threshold = waypoint_moving_threshold
        self.waypoint_turning_threshold = waypoint_turning_threshold

    def get_waypoints_to_spatial_action(self, robot_initial_position, robot_initial_heading, spatial_action):
        ################################ Position Control ################################
        robot_action = np.unravel_index(spatial_action, (self.local_map_pixel_width, self.local_map_pixel_width))

        # compute target position for front of robot:
        # computes distance from front of robot (not center), which is used to find the
        # robot position and heading needed in order to place front over specified location
        x_movement = -self.local_map_width / 2 + float(robot_action[1]) / self.local_map_pixels_per_meter
        y_movement = self.local_map_width / 2 - float(robot_action[0]) / self.local_map_pixels_per_meter

        straight_line_dist = np.sqrt(x_movement**2 + y_movement**2)
        turn_angle = np.arctan2(-x_movement, y_movement)
        straight_line_heading = restrict_heading_range(robot_initial_heading + turn_angle)

        robot_target_front_position = [
            robot_initial_position[0] + straight_line_dist * np.cos(straight_line_heading),
            robot_initial_position[1] + straight_line_dist * np.sin(straight_line_heading)
        ]

        # bound the robot to the room
        diff = np.asarray(robot_target_front_position) - np.asarray(robot_initial_position)
        ratio_x, ratio_y = (1, 1)
        bound_x = np.sign(robot_target_front_position[0]) * self.map_height / 2
        bound_y = np.sign(robot_target_front_position[1]) * self.map_width / 2
        if abs(robot_target_front_position[0]) > abs(bound_x):
            ratio_x = (bound_x - robot_initial_position[0]) / (robot_target_front_position[0] - robot_initial_position[0])
        if abs(robot_target_front_position[1]) > abs(bound_y):
            ratio_y = (bound_y - robot_initial_position[1]) / (robot_target_front_position[1] - robot_initial_position[1])
        ratio = min(ratio_x, ratio_y)
        robot_target_front_position = (np.asarray(robot_initial_position) + ratio * diff).tolist()
        # compute waypoint positions
        robot_waypoint_positions = self.shortest_path(robot_initial_position, robot_target_front_position, check_straight=True)

        # compute waypoint headings
        robot_waypoint_headings = [None]
        for i in range(1, len(robot_waypoint_positions)):
            x_diff = robot_waypoint_positions[i][0] - robot_waypoint_positions[i - 1][0]
            y_diff = robot_waypoint_positions[i][1] - robot_waypoint_positions[i - 1][1]
            waypoint_headings = restrict_heading_range(np.arctan2(y_diff, x_diff))
            robot_waypoint_headings.append(waypoint_headings)
        
        # compute movement from final waypoint to the target and apply robot radius offset to the final waypoint
        # robot_pixel_width = int(np.ceil(self.robot_info.length * self.local_map_pixels_per_meter))
        dist_to_target_end_effector_position = distance(robot_waypoint_positions[-2], robot_waypoint_positions[-1])
        signed_dist = dist_to_target_end_effector_position - self.robot_radius

        robot_move_sign = np.sign(signed_dist) # might have to move backwards to get to final position
        robot_target_heading = robot_waypoint_headings[-1]
        robot_target_position = [
            robot_waypoint_positions[-2][0] + signed_dist * np.cos(robot_target_heading),
            robot_waypoint_positions[-2][1] + signed_dist * np.sin(robot_target_heading)
        ]
        # robot_waypoint_positions[-1] = robot_target_position

        # avoid awkward backing up to reach the last waypoint
        if len(robot_waypoint_positions) > 2 and signed_dist < 0:
            robot_waypoint_positions[-2] = robot_waypoint_positions[-1]
            x_diff = robot_waypoint_positions[-2][0] - robot_waypoint_positions[-3][0]
            y_diff = robot_waypoint_positions[-2][1] - robot_waypoint_positions[-3][1]
            waypoint_heading = restrict_heading_range(np.arctan2(y_diff, x_diff))
            robot_waypoint_headings[-2] = waypoint_heading
            robot_move_sign = 1
        
        path = []
        for position, heading in zip(robot_waypoint_positions, robot_waypoint_headings):
            path.append([position[0], position[1], heading])
        path = np.array(path)

        return path, robot_move_sign
    
    def shortest_path(self, source_position, target_position, check_straight=False, configuration_space=None):
        if configuration_space is None:
            configuration_space = self.configuration_space

        # convert positions to pixel indices
        source_i, source_j = position_to_pixel_indices(source_position[0], source_position[1], configuration_space.shape, self.local_map_pixels_per_meter)
        target_i, target_j = position_to_pixel_indices(target_position[0], target_position[1], configuration_space.shape, self.local_map_pixels_per_meter)

        # check if there is a straight line path
        if check_straight:
            rr, cc = line(source_i, source_j, target_i, target_j)
            if (1 - self.configuration_space_thin[rr, cc]).sum() == 0:
                return [source_position, target_position]

        # run SPFA
        source_i, source_j = self.closest_valid_cspace_indices(source_i, source_j) # NOTE does not use the cspace passed into this method
        target_i, target_j = self.closest_valid_cspace_indices(target_i, target_j)
        _, parents = spfa.spfa(configuration_space, (source_i, source_j))

        # recover shortest path
        parents_ij = np.stack((parents // parents.shape[1], parents % parents.shape[1]), axis=2)
        parents_ij[parents < 0, :] = [-1, -1]
        i, j = target_i, target_j
        coords = [[i, j]]
        while not (i == source_i and j == source_j):
            i, j = parents_ij[i, j]
            if i + j < 0:
                break
            coords.append([i, j])

        # convert dense path to sparse path (waypoints)
        coords = approximate_polygon(np.asarray(coords), tolerance=1)

        # remove unnecessary waypoints
        new_coords = [coords[0]]
        for i in range(1, len(coords) - 1):
            rr, cc = line(*new_coords[-1], *coords[i+1])
            if (1 - configuration_space[rr, cc]).sum() > 0:
                new_coords.append(coords[i])
        if len(coords) > 1:
            new_coords.append(coords[-1])
        coords = new_coords

        # convert pixel indices back to positions
        path = []
        for coord in coords[::-1]:
            position_x, position_y = pixel_indices_to_position(coord[0], coord[1], configuration_space.shape, self.local_map_pixels_per_meter)
            path.append([position_x, position_y])
        
        if len(path) < 2:
            path = [source_position, target_position]
        else:
            path[0] = source_position
            path[-1] = target_position
        
        return path
    
    def closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]
