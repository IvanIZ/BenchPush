import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

import pymunk
from pymunk import Vec2d
from matplotlib import pyplot as plt
from shapely.geometry import Point

# Bench-NPIN related imports
from benchnpin.common.cost_map import CostMap
from benchnpin.common.controller.position_controller import PositionController
from benchnpin.common.evaluation.metrics import total_work_done
from benchnpin.common.geometry.polygon import poly_area
from benchnpin.common.utils.plot_pushing import Plot
from benchnpin.common.utils.renderer import Renderer
from benchnpin.common.utils.sim_utils import generate_sim_boxes, generate_sim_bounds, generate_sim_agent, get_color
from benchnpin.common.geometry.polygon import poly_centroid
from benchnpin.common.utils.utils import DotDict
from benchnpin.common.occupancy_grid.occupancy_map import OccupancyGrid
from benchnpin.common.controller.dp import DP

# SAM imports
from scipy.ndimage import distance_transform_edt, rotate as rotate_image
from cv2 import fillPoly
from skimage.draw import line
from skimage.measure import approximate_polygon
from skimage.morphology import disk, binary_dilation
import spfa

R = lambda theta: np.asarray([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

FORWARD = 0
STOP_TURNING = 1
LEFT = 2
RIGHT = 3
STOP = 4
BACKWARD = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7

OBSTACLE_SEG_INDEX = 0
FLOOR_SEG_INDEX = 1
RECEPTACLE_SEG_INDEX = 3
BOX_SEG_INDEX = 4
ROBOT_SEG_INDEX = 5
MAX_SEG_INDEX = 8

MOVE_STEP_SIZE = 0.05
TURN_STEP_SIZE = np.radians(15)

WAYPOINT_MOVING_THRESHOLD = 0.6
WAYPOINT_TURNING_THRESHOLD = np.radians(10)
NOT_MOVING_THRESHOLD = 0.005
NOT_TURNING_THRESHOLD = np.radians(0.05)
NONMOVEMENT_DIST_THRESHOLD = 0.05
NONMOVEMENT_TURN_THRESHOLD = np.radians(0.05)
STEP_LIMIT = 5000

class BoxDeliveryEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, cfg: dict = None):
        super(BoxDeliveryEnv, self).__init__()

        # get current directory of this script
        self.current_dir = os.path.dirname(__file__)

        # construct absolute path to the env_config folder
        base_cfg_path = os.path.join(self.current_dir, 'config.yaml')
        self.cfg = DotDict.load_from_file(base_cfg_path)

        if cfg is not None:
            # Update the base configuration with the user provided configuration
            for cfg_type in cfg:
                if type(cfg[cfg_type]) is DotDict or type(cfg[cfg_type]) is dict:
                    if cfg_type not in self.cfg:
                        self.cfg[cfg_type] = DotDict()
                    for param in cfg[cfg_type]:
                        self.cfg[cfg_type][param] = cfg[cfg_type][param]
                else:
                    self.cfg[cfg_type] = cfg[cfg_type]

        # environment
        self.local_map_pixel_width = self.cfg.env.local_map_pixel_width if self.cfg.train.job_type != 'sam' else self.cfg.env.local_map_pixel_width_sam
        self.local_map_width = self.cfg.env.local_map_width
        self.local_map_pixels_per_meter = self.local_map_pixel_width / self.local_map_width
        self.room_length = self.cfg.env.room_length
        self.wall_thickness = self.cfg.env.wall_thickness
        env_size = self.cfg.env.obstacle_config.split('_')[0]
        if env_size == 'small':
            self.num_boxes = self.cfg.boxes.num_boxes_small
            self.room_width = self.cfg.env.room_width_small
        else:
            self.num_boxes = self.cfg.boxes.num_boxes_large
            self.room_width = self.cfg.env.room_width_large
        
        # state
        self.num_channels = 4
        self.observation = None
        self.global_overhead_map = None
        self.small_obstacle_map = None
        self.configuration_space = None
        self.configuration_space_thin = None
        self.closest_cspace_indices = None

        # stats
        self.inactivity_counter = None
        self.robot_cumulative_distance = None
        self.robot_cumulative_boxes = None
        self.robot_cumulative_reward = None
        
        # robot
        self.robot_hit_obstacle = False
        self.robot_info = self.cfg.agent
        self.robot_info['color'] = get_color('agent')
        self.robot_radius = ((self.robot_info.length**2 + self.robot_info.width**2)**0.5 / 2) * 1.2
        self.robot_half_width = max(self.robot_info.length, self.robot_info.width) / 2
        robot_pixel_width = int(2 * self.robot_radius * self.local_map_pixels_per_meter)
        self.robot_state_channel = np.zeros((self.local_map_pixel_width, self.local_map_pixel_width), dtype=np.float32)
        start = int(np.floor(self.local_map_pixel_width / 2 - robot_pixel_width / 2))
        for i in range(start, start + robot_pixel_width):
            for j in range(start, start + robot_pixel_width):
                # Circular robot mask
                if (((i + 0.5) - self.local_map_pixel_width / 2)**2 + ((j + 0.5) - self.local_map_pixel_width / 2)**2)**0.5 < robot_pixel_width / 2:
                    self.robot_state_channel[i, j] = 1
        
        # rewards
        if self.cfg.train.job_type == 'sam':
            rewards = self.cfg.rewards_sam
        else:
            rewards = self.cfg.rewards
        self.partial_rewards_scale = rewards.partial_rewards_scale
        self.goal_reward = rewards.goal_reward
        self.collision_penalty = rewards.collision_penalty
        self.non_movement_penalty = rewards.non_movement_penalty
        self.correct_direction_reward_scale = rewards.correct_direction_reward_scale

        # misc
        self.ministep_size = self.cfg.misc.ministep_size
        self.inactivity_cutoff = self.cfg.misc.inactivity_cutoff if self.cfg.train.job_type != 'sam' else self.cfg.misc.inactivity_cutoff_sam
        self.random_seed = self.cfg.misc.random_seed

        self.random_state = np.random.RandomState(self.random_seed)

        self.episode_idx = None

        self.path = None
        self.scatter = False

        # Define action space
        if self.cfg.agent.action_type == 'velocity':
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'heading':
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'position':
            self.action_space = spaces.Box(low=0, high=self.local_map_pixel_width * self.local_map_pixel_width, dtype=np.float32)

        # Define observation space
        self.show_observation = False
        self.low_dim_state = self.cfg.low_dim_state
        if self.low_dim_state:
            self.fixed_trial_idx = self.cfg.fixed_trial_idx
            self.observation_space = spaces.Box(low=-10, high=30, shape=(self.num_boxes * 2,), dtype=np.float32)

        else:
            self.observation_shape = (self.local_map_pixel_width, self.local_map_pixel_width, self.num_channels)
            self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        self.plot = None
        self.renderer = None

        # used for teleoperation
        self.angular_speed = 0.0
        self.angular_speed_increment = 0.005
        self.linear_speed = 0.0
        self.linear_speed_increment = 0.02

        if self.cfg.render.show_obs or self.cfg.render.show:
            # show state
            num_plots = self.num_channels
            self.state_plot = plt
            self.state_fig, self.state_ax = self.state_plot.subplots(1, num_plots, figsize=(4 * num_plots, 6))
            self.colorbars = [None] * num_plots
            if self.cfg.render.show_obs:
                self.state_plot.ion()  # Interactive mode on        

    def init_box_delivery_sim(self):

        self.steps = self.cfg.sim.steps
        self.dp = None
        self.dt = self.cfg.controller.dt
        self.target_speed = self.cfg.controller.target_speed

        # setup pymunk environment
        self.space = pymunk.Space()  # threaded=True causes some issues
        self.space.iterations = self.cfg.sim.iterations
        self.space.gravity = self.cfg.sim.gravity
        self.space.damping = self.cfg.sim.damping

        self.total_work = [0, []]

        def robot_boundary_pre_solve(arbiter, space, data):
            self.robot_hit_obstacle = self.prevent_boundary_intersection(arbiter)
            return True
        
        def box_boundary_pre_solve(arbiter, space, data):
            self.prevent_boundary_intersection(arbiter)
            return True
        
        def recept_collision_begin(arbiter, space, data):
            return False

        self.robot_boundary_handler = self.space.add_collision_handler(1, 3)
        self.robot_boundary_handler.pre_solve = robot_boundary_pre_solve
        
        self.box_boundary_handler = self.space.add_collision_handler(2, 3)
        self.box_boundary_handler.pre_solve = box_boundary_pre_solve

        self.robot_recept_handler = self.space.add_collision_handler(1, 4)
        self.robot_recept_handler.begin = recept_collision_begin

        self.box_recept_handler = self.space.add_collision_handler(2, 4)
        self.box_recept_handler.pre_solve = recept_collision_begin

        if self.cfg.render.show:
            if self.renderer is None:
                self.renderer = Renderer(self.space, env_width=self.room_length + self.wall_thickness / 2,
                                         env_height=self.room_width + self.wall_thickness / 2,
                                         render_scale=self.cfg.render_scale, background_color=(234, 234, 234), caption='Box Delivery', centered=True)
            else:
                self.renderer.reset(new_space=self.space)

    def init_box_delivery_env(self):
        
        self.receptacle_position, self.receptacle_size = self.get_receptacle_position_and_size()
        self.goal_points = [Point(self.receptacle_position)]

        # generate random start point, if specified
        if self.cfg.agent.random_start:
            self.start = self.get_random_robot_start()
        else:
            self.start = (5, 1.5, np.pi*3/2)
        self.robot_info['start_pos'] = self.start

        self.boundary_dicts = self.generate_boundary()
        self.boxes_dicts = self.generate_boxes()

        # initialize sim objects
        self.robot = generate_sim_agent(self.space, self.robot_info, label='robot',
                                        body_type=pymunk.Body.KINEMATIC, wheel_vertices_list=self.robot_info['wheel_vertices'])
        self.boxes = generate_sim_boxes(self.space, self.boxes_dicts, self.cfg.boxes.box_density)
        self.boundaries = generate_sim_bounds(self.space, self.boundary_dicts)
        self.robot.collision_type = 1
        for p in self.boxes:
            p.collision_type = 2
        for b in self.boundaries:
            b.collision_type = 3
            if b.label == 'receptacle':
                b.collision_type = 4

        # Get vertices of corners (after they have been moved to proper spots)
        corner_dicts = [obstacle for obstacle in self.boundary_dicts if obstacle['type'] == 'corner']
        corner_polys = [shape for shape in self.boundaries if getattr(shape, 'label', None) == 'corner']
        for dict in corner_dicts:
            dict['vertices'] = []
            for _ in range(3):
                vs = corner_polys[0].get_vertices()
                transformed_vertices = [corner_polys[0].body.local_to_world(v) for v in vs]
                dict['vertices'].append(np.array([[v.x, v.y] for v in transformed_vertices]))
                corner_polys.pop(0)

        # Initialize configuration space (only need to compute once)
        self.update_configuration_space()

        self.box_clearance_statuses = [False for i in range(len(self.boxes))]

        # run initial simulation steps to let environment settle
        for _ in range(1000):
            self.space.step(self.dt / self.steps)
        self.prev_boxes = CostMap.get_obs_from_poly(self.boxes)

        self.position_controller = PositionController(self.cfg, self.robot_radius, self.room_width, self.room_length, 
                                                      self.configuration_space, self.configuration_space_thin, self.closest_cspace_indices,
                                                      self.local_map_pixel_width, self.local_map_width, self.local_map_pixels_per_meter, 
                                                      TURN_STEP_SIZE, MOVE_STEP_SIZE, WAYPOINT_MOVING_THRESHOLD, WAYPOINT_TURNING_THRESHOLD)
        
    
    def prevent_boundary_intersection(self, arbiter):
        collision = False
        normal = arbiter.contact_point_set.normal
        current_velocity = arbiter.shapes[0].body.velocity
        reflection = current_velocity - 2 * current_velocity.dot(normal) * normal

        elasticity = 0.5
        new_velocity = reflection * elasticity

        penetration_depth = arbiter.contact_point_set.points[0].distance
        if penetration_depth < 0:
            collision = True
        correction_vector = normal * penetration_depth
        arbiter.shapes[0].body.position += correction_vector

        arbiter.shapes[0].body.velocity = new_velocity

        return collision
    
    def get_random_robot_start(self):
        length = self.robot_info.length
        width = self.robot_info.width
        size = max(length, width)
        x_start = self.random_state.uniform(-self.room_length / 2 + size, self.room_length / 2 - size)
        y_start = self.random_state.uniform(-self.room_width / 2 + size, self.room_width / 2 - size)
        heading = self.random_state.uniform(0, 2 * np.pi)
        return (x_start, y_start, heading)
    
    def get_receptacle_position_and_size(self):
        size = self.cfg.env.receptacle_width
        return [(self.room_length / 2 - size / 2, self.room_width / 2 - size / 2), size]

    def generate_boundary(self):
        boundary_dicts = []
        # generate receptacle
        (x, y), size = self.receptacle_position, self.receptacle_size
        boundary_dicts.append(
            {'type': 'receptacle',
             'position': (x, y),
             'vertices': np.array([
                [x - size / 2, y - size / 2],  # bottom-left
                [x + size / 2, y - size / 2],  # bottom-right
                [x + size / 2, y + size / 2],  # top-right
                [x - size / 2, y + size / 2],  # top-left
            ]),
            'length': size,
            'width': size,
            'color': get_color('green')
        })
        
        # generate walls
        for x, y, length, width in [
            (-self.room_length / 2 - self.wall_thickness / 2, 0, self.wall_thickness, self.room_width),
            (self.room_length / 2 + self.wall_thickness / 2, 0, self.wall_thickness, self.room_width),
            (0, -self.room_width / 2 - self.wall_thickness / 2, self.room_length + 2 * self.wall_thickness, self.wall_thickness),
            (0, self.room_width / 2 + self.wall_thickness / 2, self.room_length + 2 * self.wall_thickness, self.wall_thickness),
            ]:

            boundary_dicts.append(
                {'type': 'wall',
                 'position': (x, y),
                 'vertices': np.array([
                    [x - length / 2, y - width / 2],  # bottom-left
                    [x + length / 2, y - width / 2],  # bottom-right
                    [x + length / 2, y + width / 2],  # top-right
                    [x - length / 2, y + width / 2],  # top-left
                ]),
                'color': get_color('boundary')
            })
        
        def add_random_columns(obstacles, max_num_columns):
            num_columns = self.random_state.randint(1, max_num_columns)
            column_length = 1
            column_width = 1
            buffer_width = 0.8
            col_min_dist = 2
            cols_dict = []

            new_cols = []
            for _ in range(num_columns):
                for _ in range(100): # try 100 times to generate a column that doesn't overlap with existing polygons
                    x = self.random_state.uniform(-self.room_length / 2 + 2 * buffer_width + column_length / 2,
                                        self.room_length / 2 - 2 * buffer_width - column_length / 2)
                    y = self.random_state.uniform(-self.room_width / 2 + 2 * buffer_width + column_width / 2,
                                        self.room_width / 2 - 2 * buffer_width - column_width / 2)
                    
                    overlapped = False
                    # check if column overlaps with receptacle
                    (rx, ry), size = self.receptacle_position, self.receptacle_size
                    if ((x - rx)**2 + (y - ry)**2)**(0.5) <= col_min_dist / 2 + size / 2:
                        overlapped = True
                        break

                    # check if column overlaps with robot
                    rob_x, rob_y, _ = self.robot_info['start_pos']
                    if ((x - rob_x)**2 + (y - rob_y)**2)**(0.5) <= col_min_dist / 2 + self.robot_radius:
                        overlapped = True
                        break

                    # check if column overlaps with other columns
                    for prev_col in new_cols:
                        if ((x - prev_col[0])**2 + (y - prev_col[1])**2)**(0.5) <= col_min_dist:
                            overlapped = True
                            break

                    if not overlapped:
                        new_cols.append([x, y])
                        break

            for x, y in new_cols:
                cols_dict.append({'type': 'column',
                                  'position': (x, y),
                                  'vertices': np.array([
                                      [x - column_length / 2, y - column_width / 2],  # bottom-left
                                      [x + column_length / 2, y - column_width / 2],  # bottom-right
                                      [x + column_length / 2, y + column_width / 2],  # top-right
                                      [x - column_length / 2, y + column_width / 2],  # top-left
                                      ]),
                                  'length': column_length,
                                  'width': column_width,
                                  'color': get_color('boundary')
                                })
            return cols_dict
        
        def add_random_horiz_divider():
            divider_length = 8
            divider_width = 0.5
            buffer_width = 3.5

            new_divider = []
            # while len(new_divider) == 0:
            for _ in range(100): # try 100x100 times to generate a divider that doesn't overlap with existing obstacles
                for _ in range(100):
                    overlapped = False
                    x = self.room_length / 2 - divider_length / 2
                    y = self.random_state.uniform(-self.room_width / 2 + buffer_width + divider_width / 2,
                                        self.room_width / 2 - buffer_width - divider_width / 2)
                    
                    # check if divider overlaps with robot
                    rob_x, rob_y, _ = self.robot_info['start_pos']
                    if ((x - rob_x)**2 + (y - rob_y)**2)**(0.5) <= 3 * self.robot_radius:
                        overlapped = True
                        # break
                    
                    if not overlapped:
                        new_divider.append([x, y])
                        break
                if len(new_divider) == 1:
                        break
                else:
                    self.robot_info['start_pos'] = self.get_random_robot_start()

            divider_dicts = []
            for x, y in new_divider:
                divider_dicts.append({'type': 'divider',
                                      'position': (x, y),
                                      'vertices': np.array([
                                          [x - divider_length / 2, y - divider_width / 2],  # bottom-left
                                          [x + divider_length / 2, y - divider_width / 2],  # bottom-right
                                          [x + divider_length / 2, y + divider_width / 2],  # top-right
                                          [x - divider_length / 2, y + divider_width / 2],  # top-left
                                          ]),
                                        'length': divider_length,
                                        'width': divider_width,
                                        'color': get_color('boundary')
                                    })
            return divider_dicts
                    
        
        # generate obstacles
        if self.cfg.env.obstacle_config == 'small_empty':
            pass
        elif self.cfg.env.obstacle_config == 'small_columns':
            boundary_dicts.extend(add_random_columns(boundary_dicts, 3))
        elif self.cfg.env.obstacle_config == 'large_columns':
            boundary_dicts.extend(add_random_columns(boundary_dicts, 8))
        elif self.cfg.env.obstacle_config == 'large_divider':
            boundary_dicts.extend(add_random_horiz_divider())
        else:
            raise ValueError(f'Invalid obstacle config: {self.cfg.env.obstacle_config}')
        
        # generate corners
        for i, (x, y) in enumerate([
            (-self.room_length / 2, self.room_width / 2),
            (self.room_length / 2, self.room_width / 2),
            (self.room_length / 2, -self.room_width / 2),
            (-self.room_length / 2, -self.room_width / 2),
            ]):
            if i == 1: # Skip the receptacle corner
                continue
            heading = -np.radians(i * 90)
            boundary_dicts.append(
                {'type': 'corner',
                 'position': (x, y),
                 'heading': heading,
                 'color': get_color('boundary')
                })
            
        # generate corners for divider
        for obstacle in boundary_dicts:
            if obstacle['type'] == 'divider':
                (x, y), length, width = obstacle['position'], obstacle['length'], obstacle['width']
                corner_positions = [(self.room_length / 2, y - width / 2), (self.room_length / 2, y + width / 2)]
                corner_headings = [-90, 180]
                for position, heading in zip(corner_positions, corner_headings):
                    heading = np.radians(heading)
                    boundary_dicts.append(
                        {'type': 'corner',
                        'position': position,
                        'heading': heading,
                        'color': get_color('boundary')
                        })

        return boundary_dicts

    def generate_boxes(self):
        box_size = self.cfg.boxes.box_size / 2
        boxes = []          # a list storing non-overlapping box centers

        total_boxes_required = self.num_boxes
        box_min_dist = self.cfg.boxes.min_box_dist
        min_x = -self.room_length / 2 + box_size
        max_x = self.room_length / 2 - box_size
        min_y = -self.room_width / 2 + box_size
        max_y = self.room_width / 2 - box_size

        box_count = 0
        while box_count < total_boxes_required:
            center_x = self.random_state.uniform(min_x, max_x)
            center_y = self.random_state.uniform(min_y, max_y)
            heading = self.random_state.uniform(0, 2 * np.pi)

            # loop through previous boxes to check for overlap
            overlapped = False
            for obstacle in self.boundary_dicts:
                if obstacle['type'] == 'corner' or obstacle['type'] == 'wall':
                    continue
                elif obstacle['type'] == 'divider':
                    # just check y distance
                    if abs(center_y - obstacle['position'][1]) <= (box_min_dist / 2 + obstacle['width'] / 2) * 1.2:
                        overlapped = True
                        break
                elif ((center_x - obstacle['position'][0])**2 + (center_y - obstacle['position'][1])**2)**(0.5) <= (box_min_dist / 2 + obstacle['width'] / 2) * 1.2:
                    overlapped = True
                    break
            for prev_box_x, prev_box_y, _ in boxes:
                if ((center_x - prev_box_x)**2 + (center_y - prev_box_y)**2)**(0.5) <= box_min_dist:
                    overlapped = True
                    break
            
            if not overlapped:
                boxes.append([center_x, center_y, heading])
                box_count += 1
        
        # convert to boxes dict
        boxes_dict = []
        for i, [box_x, boxes, box_heading] in enumerate(boxes):
            boxes_info = {}
            boxes_info['type'] = 'box'
            boxes_info['position'] = np.array([box_x, boxes])
            boxes_info['vertices'] = np.array([[box_x + box_size, boxes + box_size], 
                                    [box_x - box_size, boxes + box_size], 
                                    [box_x - box_size, boxes - box_size], 
                                    [box_x + box_size, boxes - box_size]])
            boxes_info['heading'] = box_heading
            boxes_info['idx'] = i
            boxes_info['color'] = get_color('box')
            boxes_dict.append(boxes_info)
        return boxes_dict

    def box_position_in_receptacle(self, box_vertices):
        for vertex in box_vertices:
            query_info = self.space.point_query(vertex, 0, pymunk.ShapeFilter())
            if not any(query.shape.label == 'receptacle' for query in query_info):
                return False
        return True
    
    def box_position_in_obstacle(self, box_vertices):
        for vertex in box_vertices:
            query_info = self.space.point_query(vertex, 0, pymunk.ShapeFilter())
            if any(query.shape.label in ['wall', 'divider', 'column', 'corner'] for query in query_info):
                return True
        return False

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state and returns the initial observation."""

        if self.episode_idx is None:
            self.episode_idx = 0
        else:
            self.episode_idx += 1

        self.init_box_delivery_sim()
        self.init_box_delivery_env()

        # reset map
        self.global_overhead_map = self.create_padded_room_zeros()
        self.update_global_overhead_map()

        self.t = 0

        # get updated boxes
        updated_boxes = CostMap.get_obs_from_poly(self.boxes)

        # reset stats
        self.inactivity_counter = 0
        self.boxes_cumulative_distance = 0
        self.robot_cumulative_distance = 0
        self.robot_cumulative_boxes = 0
        self.robot_cumulative_reward = 0

        updated_boxes = CostMap.get_obs_from_poly(self.boxes)
        if self.low_dim_state:
            self.observation = self.generate_observation_low_dim(updated_boxes=updated_boxes)

        else:
            self.observation = self.generate_observation()

        if self.cfg.render.show:
            self.show_observation = True
            self.render()
        
        info = {
            'state': (round(self.robot.body.position.x, 2),
                      round(self.robot.body.position.y, 2),
                      round(self.robot.body.angle, 2)),
            'cumulative_distance': self.robot_cumulative_distance,
            'cumulative_boxes': self.robot_cumulative_boxes,
            'cumulative_reward': self.robot_cumulative_reward,
            'total_work': self.total_work[0],
            'obs': updated_boxes,
            'box_completed_statuses': self.box_clearance_statuses,
            'goal_positions': self.goal_points,
            'ministeps': 0,
            'inactivity': self.inactivity_counter,
        }

        return self.observation, info
    

    def step(self, action):
        """Executes one time step in the environment and returns the result."""
        # print("Action: ", action)
        self.t += 1
        self.dp = None

        self.robot_hit_obstacle = False
        robot_boxes = 0
        robot_reward = 0

        # get initial state

        # initial pose
        robot_initial_position, robot_initial_heading = self.robot.body.position, self.restrict_heading_range(self.robot.body.angle)
        robot_initial_position = list(robot_initial_position)  

        # store initial box distances for partial reward calculation
        initial_box_distances = {}
        for box in self.boxes:
            box_position = box.body.position
            dist = self.shortest_path_distance(box_position, self.receptacle_position)
            initial_box_distances[box.idx] = dist

        if self.cfg.teleop_mode:
            self.demo_control(action)

            # move simulation forward
            for _ in range(self.steps):
                self.space.step(self.dt / self.steps)
                self.render()

            # get new robot pose
            robot_position, robot_heading = self.robot.body.position, self.restrict_heading_range(self.robot.body.angle)
            robot_position = list(robot_position)
            
            # update distance moved
            robot_distance = self.distance(robot_initial_position, robot_position)
        
        elif self.cfg.agent.action_type == 'velocity':
            ################################ Velocity Control ################################
            linear_speed = action[0]
            angular_speed = action[1]
            if abs(linear_speed) >= self.target_speed:
                linear_speed = self.target_speed*np.sign(linear_speed)

            # apply linear and angular speeds
            global_velocity = R(self.robot.body.angle) @ [linear_speed, 0]

            # apply velocity controller
            self.robot.body.angular_velocity = angular_speed
            self.robot.body.velocity = Vec2d(global_velocity[0], global_velocity[1])
            
            # move simulation forward
            for _ in range(self.steps):
                # ensure robot moving in the direction it is facing
                global_velocity = R(self.robot.body.angle) @ [linear_speed, 0]
                self.robot.body.velocity = Vec2d(global_velocity[0], global_velocity[1])

                self.space.step(self.dt / self.steps)
                # self.render()

                if self.robot_hit_obstacle:
                    break
            
            # get new robot pose
            robot_position, robot_heading = self.robot.body.position, self.restrict_heading_range(self.robot.body.angle)
            robot_position = list(robot_position)
            
            # update distance moved
            robot_distance = self.distance(robot_initial_position, robot_position)

        elif self.cfg.agent.action_type == 'position' or self.cfg.agent.action_type == 'heading':
            if self.cfg.agent.action_type == 'heading':
                ################################ Heading Control ################################
                # convert heading action to a pixel index in order to use the position control code

                # rescale heading action to be in range [0, 2*pi]
                angle = (action + 1) * np.pi + np.pi / 2
                step_size = self.cfg.agent.step_size

                # calculate target position
                x_movement = step_size * np.cos(angle)
                y_movement = step_size * np.sin(angle)

                # convert target position to pixel coordinates
                x_pixel = int(self.local_map_pixel_width / 2 + x_movement * self.local_map_pixels_per_meter)
                y_pixel = int(self.local_map_pixel_width / 2 - y_movement * self.local_map_pixels_per_meter)

                # convert pixel coordinates to a single index
                action = y_pixel * self.local_map_pixel_width + x_pixel

            ################################ Position Control ################################
            self.path, robot_move_sign = self.position_controller.get_waypoints_to_spatial_action(robot_initial_position, robot_initial_heading, action)
            if self.cfg.render.show:
                self.renderer.update_path(self.path)
                
            robot_distance, robot_turn_angle = self.execute_robot_path(robot_initial_position, robot_initial_heading, robot_move_sign)


        # step the simulation until everything is still
        self.step_simulation_until_still()
        # get new box positions
        final_box_distances = {}
        for box in self.boxes:
            box_position = box.body.position
            dist = self.shortest_path_distance(box_position, self.receptacle_position)
            final_box_distances[box.idx] = dist

        ############################################################################################################
        # Rewards

        # partial reward for moving boxes towards receptacle
        boxes_distance = 0
        to_remove = []
        for box in self.boxes:
            dist_moved = initial_box_distances[box.idx] - final_box_distances[box.idx]
            boxes_distance += abs(dist_moved)
            if self.cfg.train.use_correct_direction_reward and dist_moved > 0:
                dist_moved *= self.correct_direction_reward_scale
            robot_reward += self.partial_rewards_scale * dist_moved

            # reward for boxes in receptacle
            # to_remove = []
            # for box in self.boxes:
            box_vertices = [box.body.local_to_world(v) for v in box.get_vertices()]
            if self.box_position_in_receptacle(box_vertices):
                to_remove.append(box)
                self.box_clearance_statuses[box.idx] = True
                self.inactivity_counter = 0
                robot_boxes += 1
                robot_reward += self.goal_reward
        for box in to_remove:
            self.space.remove(box.body, box)
            self.boxes.remove(box)

        # penalty for hitting obstacles
        if self.robot_hit_obstacle:
            robot_reward -= self.collision_penalty
        
        # penalty for small movements
        robot_heading = self.restrict_heading_range(self.robot.body.angle)
        robot_turn_angle = self.heading_difference(robot_initial_heading, robot_heading)
        if robot_distance < NONMOVEMENT_DIST_THRESHOLD and abs(robot_turn_angle) < NONMOVEMENT_TURN_THRESHOLD:
            robot_reward -= self.non_movement_penalty
        
        ############################################################################################################
        # Compute stats
        self.robot_cumulative_distance += robot_distance
        self.robot_cumulative_boxes += robot_boxes
        self.robot_cumulative_reward += robot_reward

        # work
        updated_boxes = CostMap.get_obs_from_poly(self.boxes)
        work = total_work_done(self.prev_boxes, updated_boxes)
        self.total_work[0] += work
        self.total_work[1].append(work)
        self.prev_boxes = updated_boxes

        # increment inactivity counter, which measures steps elapsed since the previous box was stashed
        if robot_boxes == 0:
            self.inactivity_counter += 1
        
        # check if episode is done
        terminated = False
        if self.robot_cumulative_boxes == self.num_boxes:
            terminated = True
        
        truncated = False
        if self.inactivity_counter >= self.inactivity_cutoff:
            terminated = True
            truncated = True
        
        # items to return
        self.observation = self.generate_observation(done=terminated)
        reward = robot_reward
        ministeps = robot_distance / self.ministep_size
        info = {
            'state': (round(self.robot.body.position.x, 2),
                      round(self.robot.body.position.y, 2),
                      round(self.robot.body.angle, 2)),
            'cumulative_distance': self.robot_cumulative_distance,
            'cumulative_boxes': self.robot_cumulative_boxes,
            'cumulative_reward': self.robot_cumulative_reward,
            'total_work': self.total_work[0],
            'obs': updated_boxes,
            'box_completed_statuses': self.box_clearance_statuses,
            'goal_positions': self.goal_points,
            'ministeps': ministeps,
            'inactivity': self.inactivity_counter,
        }
        
        # render environment
        if self.cfg.render.show:
            self.show_observation = True
            self.render()

        return self.observation, reward, terminated, truncated, info

    def demo_control(self, action):
        if action == FORWARD:
            self.linear_speed = 0.01
        elif action == BACKWARD:
            self.linear_speed = -0.01
        elif action == STOP_TURNING:
            self.angular_speed = 0.0

        elif action == LEFT:
            self.angular_speed = 0.01
        elif action == RIGHT:
            self.angular_speed = -0.01

        elif action == SMALL_LEFT:
            self.angular_speed = 0.005
        elif action == SMALL_RIGHT:
            self.angular_speed = -0.005

        elif action == STOP:
            self.linear_speed = 0.0
            # self.angular_speed = 0.0

        # check speed boundary
        # if self.linear_speed <= 0:
        #     self.linear_speed = 0
        if abs(self.linear_speed) >= self.target_speed:
            self.linear_speed = self.target_speed*np.sign(self.linear_speed)

        # apply linear and angular speeds
        global_velocity = R(self.robot.body.angle) @ [self.linear_speed, 0]

        # apply velocity controller
        self.robot.body.angular_velocity = self.angular_speed * 100
        self.robot.body.velocity = Vec2d(global_velocity[0], global_velocity[1]) * 100
    
    def controller(self, curr_position, curr_heading):
        x = curr_position[0]
        y = curr_position[1]
        h = curr_heading

        if self.dp == None:
            cx = self.path.T[0][0:2]
            cy = self.path.T[1][0:2]
            ch = self.path.T[2][0:2]
            self.dp = DP(x=x, y=y, yaw=h, cx=cx, cy=cy, ch=ch, **self.cfg.controller)
        
        # call ideal controller to get angular and linear speeds
        omega, v = self.dp.ideal_control(x, y, h)

        # update setpoint
        x_s, y_s, h_s = self.dp.get_setpoint()
        # self.dp.setpoint = np.asarray([x_s, y_s, np.unwrap([self.dp.state.yaw, h_s])[1]])
        self.dp.setpoint = np.asarray([x_s, y_s, h_s])
        return omega, v
    
    def apply_controller(self, omega, v):
        self.robot.body.angular_velocity = omega / 2
        self.robot.body.velocity = (v*5).tolist()

    def execute_robot_path(self, robot_initial_position, robot_initial_heading, robot_move_sign):
        ############################################################################################################
        # Movement
        robot_position = robot_initial_position.copy()
        robot_heading = robot_initial_heading
        robot_is_moving = True
        robot_distance = 0
        robot_waypoint_index = 1

        robot_waypoint_positions = [(waypoint[0], waypoint[1]) for waypoint in self.path]
        robot_waypoint_headings = [waypoint[2] for waypoint in self.path]

        robot_prev_waypoint_position = robot_waypoint_positions[robot_waypoint_index - 1]
        robot_waypoint_position = robot_waypoint_positions[robot_waypoint_index]
        robot_waypoint_heading = robot_waypoint_headings[robot_waypoint_index]

        sim_steps = 0
        done_turning = False
        prev_heading_diff = 0
        while True:
            if not robot_is_moving:
                break

            # store pose to determine distance moved during simulation step
            robot_prev_position = robot_position.copy()
            robot_prev_heading = robot_heading

            # compute robot pose for new constraint
            robot_new_position = robot_position.copy()
            robot_new_heading = robot_heading
            heading_diff = self.heading_difference(robot_heading, robot_waypoint_heading)
            if np.abs(heading_diff) > TURN_STEP_SIZE and np.abs(heading_diff - prev_heading_diff) > 0.001:
                # turn towards next waypoint first
                robot_new_heading += np.sign(heading_diff) * TURN_STEP_SIZE
            else:
                done_turning = True
                dx = robot_waypoint_position[0] - robot_position[0]
                dy = robot_waypoint_position[1] - robot_position[1]
                if self.distance(robot_position, robot_waypoint_position) < MOVE_STEP_SIZE:
                    robot_new_position = robot_waypoint_position
                else:
                    if robot_waypoint_index == len(robot_waypoint_position) - 1:
                        move_sign = robot_move_sign
                    else:
                        move_sign = 1
                    robot_new_heading = np.arctan2(move_sign * dy, move_sign * dx)
                    robot_new_position[0] += move_sign * MOVE_STEP_SIZE * np.cos(robot_new_heading)
                    robot_new_position[1] += move_sign * MOVE_STEP_SIZE * np.sin(robot_new_heading)
            # change robot pose
            omega, v = self.controller(robot_prev_position, robot_prev_heading)
            if not done_turning:
                self.apply_controller(omega, v*0)
            else:
                self.apply_controller(omega, v)
            self.space.step(self.dt / self.steps)

            # get new robot pose
            robot_position, robot_heading = self.robot.body.position, self.restrict_heading_range(self.robot.body.angle)
            robot_position = list(robot_position)
            prev_heading_diff = heading_diff

            # stop moving if robot collided with obstacle
            if self.distance(robot_prev_waypoint_position, robot_position) > MOVE_STEP_SIZE:
                if self.robot_hit_obstacle:
                    # self.robot_hit_obstacle = False
                    robot_is_moving = False
                    break  # Note: self.robot_distance does not get not updated
            
            # stop if robot reached waypoint
            if (self.distance(robot_position, robot_waypoint_positions[robot_waypoint_index]) < WAYPOINT_MOVING_THRESHOLD
                and np.abs(robot_heading - robot_waypoint_headings[robot_waypoint_index]) < WAYPOINT_TURNING_THRESHOLD):
                
                # update distance moved
                robot_distance += self.distance(robot_prev_waypoint_position, robot_position)

                # increment waypoint index or stop moving if done
                if robot_waypoint_index == len(robot_waypoint_positions) - 1:
                    robot_is_moving = False
                else:
                    robot_waypoint_index += 1
                    robot_prev_waypoint_position = robot_waypoint_positions[robot_waypoint_index - 1]
                    robot_waypoint_position = robot_waypoint_positions[robot_waypoint_index]
                    robot_waypoint_heading = robot_waypoint_headings[robot_waypoint_index]
                    done_turning = False
                    self.dp = None
                    self.path = self.path[1:]

            sim_steps += 1
            if sim_steps % 5 == 0 and self.cfg.render.show:
                self.render()

            # break if robot is stuck
            if sim_steps > STEP_LIMIT:
                break

        robot_heading = self.restrict_heading_range(self.robot.body.angle)
        robot_turn_angle = self.heading_difference(robot_initial_heading, robot_heading)
        return robot_distance, robot_turn_angle

    def step_simulation_until_still(self):
        prev_positions = []
        sim_steps = 0
        done = False
        while not done:
            # check if any box is stuck in a wall
            for box in self.boxes:
                box_vertices = [box.body.local_to_world(v) for v in box.get_vertices()]
                if self.box_position_in_obstacle(box_vertices):
                    # move to nearest free space
                    box_position = box.body.position
                    pixel_i, pixel_j = self.position_to_pixel_indices(box_position.x, box_position.y, self.configuration_space.shape)
                    nearest_i, nearest_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
                    new_x, new_y = self.pixel_indices_to_position(nearest_i, nearest_j, self.configuration_space.shape)
                    box.body.position = (new_x, new_y)
                    box.body.velocity = (0, 0)
                    
            # check whether anything moved since last step
            positions = []
            for poly in self.boxes + [self.robot]:
                positions.append(poly.body.position)
            if len(prev_positions) > 0:
                done = True
                for i, position in enumerate(positions):
                    if np.linalg.norm(np.asarray(prev_positions[i]) - np.asarray(position)) > NOT_MOVING_THRESHOLD:
                        done = False
                        break
            prev_positions = positions

            self.space.step(self.dt / self.steps)

            sim_steps += 1
            if sim_steps > STEP_LIMIT:
                break

    def generate_observation_low_dim(self, updated_boxes):
        """
        The observation is a vector of shape (num_boxes * 2) specifying the 2d position of the boxes
        <obs1_x, obs1_y, obs2_x, obs2_y, ..., obsn_x, obsn_y>
        """
        observation = np.zeros((len(updated_boxes) * 2))
        for i in range(len(updated_boxes)):
            obs = updated_boxes[i]
            center = np.abs(poly_centroid(obs))
            observation[i * 2] = center[0]
            observation[i * 2 + 1] = center[1]
        return observation


    def update_path(self, new_path, scatter=False):
        if scatter:
            self.scatter = True
        self.path = new_path
    

    def generate_observation(self, done=False):
        self.update_global_overhead_map()

        if done and self.cfg.agent.action_type == 'position':
            return None
        
        # Overhead map
        channels = []
        channels.append(self.get_local_map(self.global_overhead_map, self.robot.body.position, self.robot.body.angle))
        channels.append(self.robot_state_channel)
        channels.append(self.get_local_distance_map(self.create_global_shortest_path_to_receptacle_map(), self.robot.body.position, self.robot.body.angle))
        channels.append(self.get_local_distance_map(self.create_global_shortest_path_map(self.robot.body.position), self.robot.body.position, self.robot.body.angle))
        observation = np.stack(channels, axis=2)
        observation = (observation * 255).astype(np.uint8)
        return observation
    
    def get_local_overhead_map(self):
        rotation_angle = -np.degrees(self.robot.body.angle) + 90
        pos_y = int(np.floor(self.global_overhead_map.shape[0] / 2 - self.robot.body.position.y * self.local_map_pixels_per_meter))
        pos_x = int(np.floor(self.global_overhead_map.shape[1] / 2 + self.robot.body.position.x * self.local_map_pixels_per_meter))
        mask = rotate_image(np.zeros((self.local_map_pixel_width, self.local_map_pixel_width), dtype=np.float32), rotation_angle, order=0)
        y_start = pos_y - int(mask.shape[0] / 2)
        y_end = y_start + mask.shape[0]
        x_start = pos_x - int(mask.shape[1] / 2)
        x_end = x_start + mask.shape[1]
        crop = self.global_overhead_map[y_start:y_end, x_start:x_end]
        crop = rotate_image(crop, rotation_angle, order=0)
        y_start = int(crop.shape[0] / 2 - self.local_map_pixel_width / 2)
        y_end = y_start + self.local_map_pixel_width
        x_start = int(crop.shape[1] / 2 - self.local_map_pixel_width / 2)
        x_end = x_start + self.local_map_pixel_width
        return crop[y_start:y_end, x_start:x_end]
    
    def get_local_map(self, global_map, robot_position, robot_heading):
        crop_width = self.round_up_to_even(self.local_map_pixel_width * np.sqrt(2))
        rotation_angle = 90 - np.degrees(robot_heading)
        pixel_i = int(np.floor(-robot_position[1] * self.local_map_pixels_per_meter + global_map.shape[0] / 2))
        pixel_j = int(np.floor(robot_position[0] * self.local_map_pixels_per_meter + global_map.shape[1] / 2))
        crop = global_map[pixel_i - crop_width // 2:pixel_i + crop_width // 2, pixel_j - crop_width // 2:pixel_j + crop_width // 2]
        rotated_crop = rotate_image(crop, rotation_angle, order=0)
        local_map = rotated_crop[
            rotated_crop.shape[0] // 2 - self.local_map_pixel_width // 2:rotated_crop.shape[0] // 2 + self.local_map_pixel_width // 2,
            rotated_crop.shape[1] // 2 - self.local_map_pixel_width // 2:rotated_crop.shape[1] // 2 + self.local_map_pixel_width // 2
        ]
        return local_map
    
    def get_local_distance_map(self, global_map, robot_position, robot_heading):
        local_map = self.get_local_map(global_map, robot_position, robot_heading)
        local_map -= local_map.min() # move the min to 0 to make invariant to size of environment
        return local_map

    def create_padded_room_zeros(self):
        return np.zeros((
            int(2 * np.ceil((self.room_width * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2)),
            int(2 * np.ceil((self.room_length * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2))
        ), dtype=np.float32)
    
    def create_padded_room_ones(self):
        return np.ones((
            int(2 * np.ceil((self.room_width * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2)),
            int(2 * np.ceil((self.room_length * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2))
        ), dtype=np.float32)

    def create_global_shortest_path_to_receptacle_map(self):
        global_map = self.create_padded_room_zeros() + np.inf
        (rx, ry) = self.receptacle_position
        pixel_i, pixel_j = self.position_to_pixel_indices(rx, ry, self.configuration_space.shape)
        pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
        shortest_path_image, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
        shortest_path_image /= self.local_map_pixels_per_meter
        global_map = np.minimum(global_map, shortest_path_image)
        global_map /= (np.sqrt(2) * self.local_map_pixel_width) / self.local_map_pixels_per_meter
        global_map *= self.cfg.env.shortest_path_channel_scale
        if self.cfg.env.invert_receptacle_map:
            global_map += 1-self.configuration_space
            global_map[global_map==(1-self.configuration_space)] = 1
        return global_map
    
    def create_global_shortest_path_map(self, robot_position):
        pixel_i, pixel_j = self.position_to_pixel_indices(robot_position[0], robot_position[1], self.configuration_space.shape)
        pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
        global_map, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
        global_map /= self.local_map_pixels_per_meter
        global_map /= (np.sqrt(2) * self.local_map_pixel_width) / self.local_map_pixels_per_meter
        global_map *= self.cfg.env.shortest_path_channel_scale
        return global_map
    
    def update_configuration_space(self):
        """
        Obstacles are dilated based on the robot's radius to define a collision-free space
        """

        # obstacle_map = self.create_padded_room_zeros()
        obstacle_map = self.create_padded_room_ones()
        small_obstacle_map = np.zeros((self.local_map_pixel_width+20, self.local_map_pixel_width+20), dtype=np.float32)

        for poly in self.boundaries:
            # get world coordinates of vertices
            vertices = [poly.body.local_to_world(v) for v in poly.get_vertices()]
            vertices_np = np.array([[v.x, v.y] for v in vertices])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * self.local_map_pixels_per_meter).astype(np.int32)
            vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] = small_obstacle_map.shape[0] - vertices_px[:, 1]

            # draw the boundary on the small_obstacle_map
            if poly.label in ['wall', 'divider', 'column', 'corner']:
                fillPoly(small_obstacle_map, [vertices_px], color=1)
        
        start_i, start_j = int(obstacle_map.shape[0] / 2 - small_obstacle_map.shape[0] / 2), int(obstacle_map.shape[1] / 2 - small_obstacle_map.shape[1] / 2)
        obstacle_map[start_i:start_i + small_obstacle_map.shape[0], start_j:start_j + small_obstacle_map.shape[1]] = small_obstacle_map

        # Dilate obstacles and walls based on robot size
        selem = disk(np.floor(self.robot_radius * self.local_map_pixels_per_meter))
        self.configuration_space = 1 - binary_dilation(obstacle_map, selem).astype(np.float32)
        
        selem_thin = disk(np.floor(self.robot_half_width * self.local_map_pixels_per_meter))
        self.configuration_space_thin = 1 - binary_dilation(obstacle_map, selem_thin).astype(np.float32)

        self.closest_cspace_indices = distance_transform_edt(1 - self.configuration_space, return_distances=False, return_indices=True)
        self.small_obstacle_map = 1 - small_obstacle_map

    def update_global_overhead_map(self):
        small_overhead_map = self.small_obstacle_map.copy()

        for poly in self.boundaries + self.boxes + [self.robot]:
            if poly.label in ['wall', 'divider', 'column', 'corner']:
                continue # precomputed in update_configuration_space

            # get world coordinates of vertices
            vertices = [poly.body.local_to_world(v) for v in poly.get_vertices()]
            vertices_np = np.array([[v.x, v.y] for v in vertices])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * self.local_map_pixels_per_meter).astype(np.int32)
            vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] = small_overhead_map.shape[0] - vertices_px[:, 1]

            # draw the boundary on the small_overhead_map
            small_overhead_map[small_overhead_map == 1] = FLOOR_SEG_INDEX / MAX_SEG_INDEX
            if poly.label == 'receptacle':
                fillPoly(small_overhead_map, [vertices_px], color=RECEPTACLE_SEG_INDEX/MAX_SEG_INDEX)
            elif poly.label == 'box':
                fillPoly(small_overhead_map, [vertices_px], color=BOX_SEG_INDEX/MAX_SEG_INDEX)
            elif poly.label == 'robot':
                fillPoly(small_overhead_map, [vertices_px], color=ROBOT_SEG_INDEX/MAX_SEG_INDEX)

        start_i, start_j = int(self.global_overhead_map.shape[0] / 2 - small_overhead_map.shape[0] / 2), int(self.global_overhead_map.shape[1] / 2 - small_overhead_map.shape[1] / 2)
        self.global_overhead_map[start_i:start_i + small_overhead_map.shape[0], start_j:start_j + small_overhead_map.shape[1]] = small_overhead_map
    
    def shortest_path(self, source_position, target_position, check_straight=False, configuration_space=None):
        if configuration_space is None:
            configuration_space = self.configuration_space

        # convert positions to pixel indices
        source_i, source_j = self.position_to_pixel_indices(source_position[0], source_position[1], configuration_space.shape)
        target_i, target_j = self.position_to_pixel_indices(target_position[0], target_position[1], configuration_space.shape)

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
            position_x, position_y = self.pixel_indices_to_position(coord[0], coord[1], configuration_space.shape)
            path.append([position_x, position_y])
        
        if len(path) < 2:
            path = [source_position, target_position]
        else:
            path[0] = source_position
            path[-1] = target_position
        
        return path

    def shortest_path_distance(self, source_position, target_position, configuration_space=None):
        path = self.shortest_path(source_position, target_position, configuration_space=configuration_space)
        return sum(self.distance(path[i - 1], path[i]) for i in range(1, len(path)))
    
    def closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]

    def render(self, mode='human', close=False):
        """Renders the environment."""
        if self.t % self.cfg.anim.plot_steps == 0 and False:
            direc = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + '.png')
            self.renderer.render(save=False, path=direc, manual_draw=True)

            # if self.path is not None:

            #     if not self.scatter:
            #         self.plot.update_path(full_path=self.path.T)
            #     else:
            #         self.plot.update_path_scatter(full_path=self.path.T)

            # self.plot.update_ship(self.robot.body, self.robot, move_yaxis_threshold=self.cfg.anim.move_yaxis_threshold)
            # self.plot.update_obstacles(obstacles=CostMap.get_obs_from_poly(self.boxes))
            # # get updated obstacles
            # self.plot.animate_sim(save_fig_dir=os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
            #                 if (self.cfg.anim.save and self.cfg.output_dir) else None, suffix=self.t)
        
        else:
            self.renderer.render(save=False, manual_draw=True)

            if self.cfg.render.show_obs and not self.low_dim_state and self.show_observation and self.observation is not None:# and self.t % self.cfg.render.frequency == 1:
                self.show_observation = False
                for ax, i in zip(self.state_ax, range(self.num_channels)):
                    ax.clear()
                    ax.set_title(f'Channel {i}')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    im = ax.imshow(self.observation[:,:,i], cmap='hot', interpolation='nearest')
                    # if self.colorbars[i] is not None:
                    #     self.colorbars[i].update_normal(im)
                    # else:
                    #     self.colorbars[i] = self.state_fig.colorbar(im, ax=ax)
                
                self.state_plot.draw()
                # self.state_plot.pause(0.001)
                self.state_plot.pause(0.1)

    # Helper functions
    def round_up_to_even(self, x):
        return int(np.ceil(x / 2) * 2)

    def distance(self, position1, position2):
        return np.linalg.norm(np.asarray(position1)[:2] - np.asarray(position2)[:2])

    def restrict_heading_range(self, heading):
        return np.mod(heading + np.pi, 2 * np.pi) - np.pi

    def heading_difference(self, heading1, heading2):
        return self.restrict_heading_range(heading1 - heading2)

    def position_to_pixel_indices(self, x, y, image_shape):
        pixel_i = np.floor(image_shape[0] / 2 - y * self.local_map_pixels_per_meter).astype(np.int32)
        pixel_j = np.floor(image_shape[1] / 2 + x * self.local_map_pixels_per_meter).astype(np.int32)
        pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
        pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
        return pixel_i, pixel_j

    def pixel_indices_to_position(self, pixel_i, pixel_j, image_shape):
        position_x = (pixel_j - image_shape[1] / 2) / self.local_map_pixels_per_meter
        position_y = (image_shape[0] / 2 - pixel_i) / self.local_map_pixels_per_meter
        return position_x, position_y
    
    def close(self):
        """Optional: close any resources or cleanup if necessary."""
        plt.close('all')
        if self.cfg.render.show:
            self.renderer.close()
