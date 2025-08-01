import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

import random
import pymunk
from pymunk import Vec2d
from matplotlib import pyplot as plt

# Bench_NPIN related imports
from benchnpin.common.controller.dp import DP
from benchnpin.common.controller.position_controller import PositionController
from benchnpin.common.cost_map import CostMap
from benchnpin.common.evaluation.metrics import total_work_done, obs_to_goal_difference
from benchnpin.common.geometry.polygon import poly_area
from benchnpin.common.utils.sim_utils import generate_sim_obs, generate_sim_agent, get_color
from benchnpin.common.geometry.polygon import poly_centroid, create_polygon_from_line
from benchnpin.common.utils.utils import DotDict
from benchnpin.common.types import ObstacleType
from benchnpin.common.utils.renderer import Renderer

from shapely.geometry import Polygon, LineString, Point

from benchnpin.environments.area_clearing.utils import round_up_to_even, position_to_pixel_indices, pixel_indices_to_position
from scipy.ndimage import distance_transform_edt, rotate as rotate_image
from skimage.morphology import disk, binary_dilation
import spfa

from cv2 import fillPoly

R = lambda theta: np.asarray([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

PAPER_RENDER_MODE = False

BOUNDARY_PENALTY = -0.25
BOX_PUTBACK_PENALTY = -10 # -1
TRUNCATION_PENALTY = 0
TERMINAL_REWARD = 50
BOX_CLEARED_REWARD = 10 # 1
BOX_PUSHING_REWARD_MULTIPLIER = 0.2
NONMOVEMENT_PENALTY = 0 #-0.25
# TIME_PENALTY = -0.01
TIME_PENALTY = 0


DISTANCE_SCALE_MAX = 0.5

OBSTACLE_SEG_INDEX = 0
FLOOR_SEG_INDEX = 1
GOAL_AREA_SEG_INDEX = 3
COMPLETED_CUBE_SEG_INDEX = 7
CUBE_SEG_INDEX = 4
ROBOT_SEG_INDEX = 5
MAX_SEG_INDEX = 8

FORWARD = 0
STOP_TURNING = 1
LEFT = 2
RIGHT = 3
STOP = 4
BACKWARD = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7

MOVE_STEP_SIZE = 0.05
TURN_STEP_SIZE = np.radians(15)
WAYPOINT_MOVING_THRESHOLD = 0.6
WAYPOINT_TURNING_THRESHOLD = np.radians(10)
NONMOVEMENT_DIST_THRESHOLD = 0.05
NONMOVEMENT_TURN_THRESHOLD = np.radians(0.05)
# STEP_LIMIT = 500
STEP_LIMIT = 5000
MAP_UPDATE_STEPS = 250

class AreaClearingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, cfg = None, **kwargs):
        super(AreaClearingEnv, self).__init__()

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

        env_cfg_file_path = os.path.join(self.current_dir, 'envs/' + self.cfg.env + '.yaml')

        if not os.path.exists(env_cfg_file_path):
            raise FileNotFoundError(f"Environment config file {env_cfg_file_path} not found")

        self.env_cfg = DotDict.load_from_file(env_cfg_file_path)

        self.env_max_trial = 4000
        self.beta = 500         # amount to scale the collision reward
        self.episode_idx = None
        self.path = None
        self.scatter = False

        # environment
        self.local_map_pixel_width = self.env_cfg.local_map_pixel_width
        self._configure_state_and_observation_channels()

        self.yaw_lim = (0, np.pi)       # lower and upper limit of ship yaw  

        # self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))

        self.demo_mode = False

        self.outer_boundary_vertices = self.env_cfg.outer_boundary
        self.boundary_vertices = self.env_cfg.boundary
        self.walls = self.env_cfg.walls if 'walls' in self.env_cfg else []
        self.static_obstacles = self.env_cfg.static_obstacles if 'static_obstacles' in self.env_cfg else []

        # move boundary to the center of the environment
        self.boundary_polygon = Polygon(self.boundary_vertices)
        self.outer_boundary_polygon = Polygon(self.outer_boundary_vertices)

        self.min_x_boundary = min([x for x, y in self.boundary_vertices])
        self.max_x_boundary = max([x for x, y in self.boundary_vertices])
        self.min_y_boundary = min([y for x, y in self.boundary_vertices])
        self.max_y_boundary = max([y for x, y in self.boundary_vertices])

        self.min_x_outer = min([x for x, y in self.outer_boundary_vertices])
        self.max_x_outer = max([x for x, y in self.outer_boundary_vertices])
        self.min_y_outer = min([y for x, y in self.outer_boundary_vertices])
        self.max_y_outer = max([y for x, y in self.outer_boundary_vertices])

        self.map_width = self.max_x_outer - self.min_x_outer
        self.map_height = self.max_y_outer - self.min_y_outer

        self.renderer = None

        self.cleared_box_count = 0

        self.state_fig, self.state_ax = plt.subplots(1, self.num_channels, figsize=(4 * self.num_channels, 6))

        ### DEBUG: Seperate figures for paper
        # self.state_figs = []
        # self.state_axes = []

        # for i in range(self.num_channels):
        #     fig, ax = plt.subplots(figsize=(6, 6))
        #     self.state_figs.append(fig)
        #     self.state_axes.append(ax)

        self.colorbars = [None] * self.num_channels

        self.boundary_goals, self.goal_points = self._compute_boundary_goals()

        self.position_controller = None

    def _configure_state_and_observation_channels(self):
        # observation
        self.num_channels = 4
        self.observation = None
        self.global_overhead_map = None
        self.small_obstacle_map = None
        self.configuration_space = None
        self.configuration_space_thin = None
        self.closest_cspace_indices = None
        self.goal_point_global_map = None

        self.local_map_width = self.env_cfg.local_map_width
        self.local_map_pixels_per_meter = self.local_map_pixel_width / self.local_map_width

        # robot state channel
        self.agent_info = self.cfg.agent
        self.robot_radius = ((self.agent_info.length**2 + self.agent_info.width**2)**0.5 / 2) * 1.2
        robot_pixel_width = int(2 * self.robot_radius * self.local_map_pixels_per_meter)
        self.robot_state_channel = np.zeros((self.local_map_pixel_width, self.local_map_pixel_width), dtype=np.float32)
        start = int(np.floor(self.local_map_pixel_width / 2 - robot_pixel_width / 2))
        for i in range(start, start + robot_pixel_width):
            for j in range(start, start + robot_pixel_width):
                # Circular robot mask
                if (((i + 0.5) - self.local_map_pixel_width / 2)**2 + ((j + 0.5) - self.local_map_pixel_width / 2)**2)**0.5 < robot_pixel_width / 2:
                    self.robot_state_channel[i, j] = 1

        self.target_speed = self.cfg.controller.target_speed

        # Define action space
        max_yaw_rate_step = (np.pi/2) / 15        # rad/sec
        print("max yaw rate per step: ", max_yaw_rate_step)
        
        if self.cfg.agent.action_type == 'velocity':
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'position':
            self.action_space = spaces.Box(low=0, high=self.local_map_pixel_width * self.local_map_pixel_width, shape=(1,), dtype=np.int32)
        elif self.cfg.agent.action_type == 'heading':
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.max_yaw_rate_step = max_yaw_rate_step

        # Define observation space
        self.low_dim_state = self.cfg.low_dim_state
        if self.low_dim_state:
            self.fixed_trial_idx = self.cfg.fixed_trial_idx
            self.observation_space = spaces.Box(low=-10, high=30, shape=(self.cfg.num_obstacles * 2,), dtype=np.float32)
        else:
            # self.observation_shape = (self.num_channels, self.local_map_pixel_width, self.local_map_pixel_width)
            self.observation_shape = (self.local_map_pixel_width, self.local_map_pixel_width, self.num_channels)
            self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

    def configure_env_for_SAM(self):
        self.local_map_pixel_width = self.env_cfg.sam_local_map_pixel_width
        self._configure_state_and_observation_channels()

    def _compute_boundary_goals(self, interpolated_points=10):
        if self.boundary_vertices is None:
            return None
        
        boundary_edges = []
        for i in range(len(self.boundary_vertices)):
            boundary_edges.append([self.boundary_vertices[i], self.boundary_vertices[(i + 1) % len(self.boundary_vertices)]])
        
        boundary_linestrings = [LineString(edge) for edge in boundary_edges]

        # remove walls from boundary
        for wall in self.walls:
            wall_polygon = LineString(wall)
            wall_polygon = wall_polygon.buffer(0.1)
            for i in range(len(boundary_linestrings)):
                boundary_linestrings[i] = boundary_linestrings[i].difference(wall_polygon)

        # convert multilinestrings to linestrings
        temp_boundary_linestrings = boundary_linestrings.copy()
        boundary_linestrings = []
        for line in temp_boundary_linestrings:
            if line.geom_type == 'MultiLineString':
                boundary_linestrings.extend([ls for ls in list(line.geoms) if ls.length > 0.1])
            elif line.geom_type == 'LineString':
                if line.length > 0.1:
                    boundary_linestrings.append(line)
            else:
                raise ValueError("Invalid geometry type to handle")

        boundary_goals = boundary_linestrings
        
        # get 5 evenly spaced points on each boundary goal line
        goal_points = []
        for line in boundary_goals:
            line_length = line.length
            for i in range(int(interpolated_points)):
                goal_points.append(line.interpolate(((i + 1/2) / interpolated_points) * line_length))

        return boundary_goals, goal_points

    
    def activate_demo_mode(self):
        self.demo_mode = True
        
        self.angular_speed = 0.0
        self.angular_speed_increment = 0.005
        self.linear_speed = 0.0
        self.linear_speed_increment = 0.02

    def init_area_clearing_sim(self):

        # initialize robot clearing environment
        self.steps = self.cfg.sim.steps
        self.t_max = self.cfg.sim.t_max if self.cfg.sim.t_max else np.inf
        self.dt = self.cfg.controller.dt

        # setup pymunk environment
        self.space = pymunk.Space()  # threaded=True causes some issues
        self.space.iterations = self.cfg.sim.iterations
        self.space.gravity = self.cfg.sim.gravity
        self.space.damping = self.cfg.sim.damping

        # keep track of running total of total kinetic energy / total impulse
        # computed using pymunk api call, source code here
        # https://github.com/slembcke/Chipmunk2D/blob/edf83e5603c5a0a104996bd816fca6d3facedd6a/src/cpArbiter.c#L158-L172
        self.system_ke_loss = []   # https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.total_ke
                                # source code in Chimpunk2D cpArbiterTotalKE
        self.total_ke = [0, []]  # keep track of both running total and ke at each collision
        self.total_impulse = [0, []]
        # keep track of running total of work
        self.total_work = [0, []]

        self.total_dis = 0 
        self.prev_state = None   

        # keep track of all the obstacles that collide with ship
        self.clln_obs = set()

        # keep track of contact points
        self.contact_pts = []
        self.robot_hit_obstacle = False

        # setup pymunk collision callbacks
        def pre_solve_handler(arbiter, space, data):
            obs_body = arbiter.shapes[1].body
            obs_body.pre_collision_KE = obs_body.kinetic_energy  # hacky, adding a field to pymunk body object
            return True

        def post_solve_handler(arbiter, space, data):
            self.system_ke_loss.append(arbiter.total_ke)

            self.total_ke[0] += arbiter.total_ke
            self.total_ke[1].append(arbiter.total_ke)

            self.total_impulse[0] += arbiter.total_impulse.length
            self.total_impulse[1].append(list(arbiter.total_impulse))

            if arbiter.is_first_contact:
                self.clln_obs.add(arbiter.shapes[1])

            # max of two sets of points, easy to see with a picture with two overlapping convex shapes
            # find the impact locations in the local coordinates of the ship
            for i in arbiter.contact_point_set.points:
                self.contact_pts.append(list(arbiter.shapes[0].body.world_to_local((i.point_b + i.point_a) / 2)))
        
        def robot_boundary_pre_solve(arbiter, space, data):
            self.robot_hit_obstacle = self.prevent_boundary_intersection(arbiter)
            return True
        
        def cube_boundary_pre_solve(arbiter, space, data):
            self.prevent_boundary_intersection(arbiter)
            return True

        # handler = space.add_default_collision_handler()
        self.handler = self.space.add_collision_handler(1, 2)
        # from pymunk docs
        # post_solve: two shapes are touching and collision response processed
        self.handler.pre_solve = pre_solve_handler
        self.handler.post_solve = post_solve_handler

        self.robot_boundary_handler = self.space.add_collision_handler(1, 3)
        self.robot_boundary_handler.pre_solve = robot_boundary_pre_solve
        
        self.cube_boundary_handler = self.space.add_collision_handler(2, 3)
        self.cube_boundary_handler.pre_solve = cube_boundary_pre_solve

        if self.cfg.render.show:
            if self.renderer is None:
                self.renderer = Renderer(self.space, env_width=self.map_width + 2, env_height=self.map_height + 2, render_scale=self.cfg.render_scale, 
                        background_color=(245, 245, 245), caption="Area Clearing", 
                        centered=True,
                        clearance_boundary=self.boundary_vertices
                        )
            else:
                self.renderer.reset(new_space=self.space)
        
    def init_area_clearing_env(self):

        # generate random start point, if specified
        if self.cfg.random_start:
            x_start = (self.min_x_boundary + 1) + random.random() * ((self.max_x_boundary - self.min_x_boundary) - 2)
            self.start = (x_start, self.min_y_boundary + 1.0, np.pi / 2)
        else:
            mid_x = (self.min_x_boundary + self.max_x_boundary) / 2
            self.start = (mid_x, self.min_y_boundary + 1.0, np.pi / 2)

        ### DEBUG: Used for paper image
        if PAPER_RENDER_MODE:
            self.start = (-0.07870898347806499, -4.0, 1.5707963267948966)

        self.agent_info['start_pos'] = self.start
        self.agent_info['color'] = (100, 100, 100, 255)

        self.obs_dicts = self.generate_obstacles()
        obs_dicts, self.static_obs_shapes = self.generate_static_obstacles()
        self.obs_dicts.extend(obs_dicts)
        obs_dicts, self.wall_shapes = self.generate_walls()
        self.obs_dicts.extend(obs_dicts)

        for b in self.static_obs_shapes + self.wall_shapes:
            b.collision_type = 3
        
        # filter out obstacles that have zero area
        self.obs_dicts[:] = [ob for ob in self.obs_dicts if (poly_area(ob['vertices']) != 0)]
        self.obstacles = [ob['vertices'] for ob in self.obs_dicts]

        # initialize ship sim objects
        self.dynamic_obs = [ob for ob in self.obs_dicts if ob['type'] == ObstacleType.DYNAMIC]
        self.box_shapes = generate_sim_obs(self.space, self.dynamic_obs, self.cfg.sim.obstacle_density, color=(204, 153, 102, 255))
        for p in self.box_shapes:
            p.collision_type = 2
        
        self.box_clearance_statuses = [False for i in range(len(self.box_shapes))]

        self.agent = generate_sim_agent(self.space, self.agent_info, body_type=pymunk.Body.KINEMATIC, 
                                        wheel_vertices_list=self.cfg.agent.wheel_vertices, front_bumper_vertices=self.cfg.agent.front_bumper_vertices)
        self.agent.collision_type = 1

        # Initialize configuration space (only need to compute once)
        if(self.configuration_space is None):
            self.update_configuration_space()

        for _ in range(1000):
            self.space.step(self.dt / self.steps)
        self.prev_obs = CostMap.get_obs_from_poly(self.box_shapes)

        self.position_controller = PositionController(self.cfg, self.robot_radius, self.map_width, self.map_height, 
                                                      self.configuration_space, self.configuration_space_thin, self.closest_cspace_indices,
                                                      self.local_map_pixel_width, self.local_map_width, self.local_map_pixels_per_meter, 
                                                      TURN_STEP_SIZE, MOVE_STEP_SIZE, WAYPOINT_MOVING_THRESHOLD, WAYPOINT_TURNING_THRESHOLD)
        
        self.dp = None

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

    def generate_static_obstacles(self):
        obs_dict = []
        static_obs_shapes = []
        for obstacle in self.static_obstacles:
            obs_info = {}
            obs_info['type'] = ObstacleType.STATIC
            obs_info['vertices'] = np.array(obstacle)

            shape = pymunk.Poly(self.space.static_body, obstacle, radius=0.1)
            shape.collision_type = 2
            shape.friction = 0.99

            self.space.add(shape)
            obs_dict.append(obs_info)
            static_obs_shapes.append(shape)

        return obs_dict, static_obs_shapes

    def generate_walls(self):
        obs_dict = []
        wall_shapes = []
        outer_boundary_walls = []
        for i in range(len(self.outer_boundary_vertices)):
            outer_boundary_walls.append([self.outer_boundary_vertices[i], self.outer_boundary_vertices[(i + 1) % len(self.outer_boundary_vertices)]])
        for wall_vertices in self.walls:# + outer_boundary_walls:
            # convert line to polygon
            wall_poly = create_polygon_from_line(wall_vertices)

            obs_info = {}
            obs_info['type'] = ObstacleType.BOUNDARY
            obs_info['vertices'] = wall_poly

            # convert np array to list
            wall_poly = [(x, y) for x, y in wall_poly]

            shape = pymunk.Poly(self.space.static_body, wall_poly, radius=0.1)
            shape.collision_type = 2
            shape.friction = 0.99

            self.space.add(shape)
            obs_dict.append(obs_info)
            wall_shapes.append(shape)
        
        # generate outer walls
        room_length = abs(self.outer_boundary_vertices[0][0])*2
        room_width = abs(self.outer_boundary_vertices[0][1])*2
        wall_thickness = 24
        
        for x, y, length, width in [
            (-room_length / 2 - wall_thickness / 2, 0, wall_thickness, room_width),
            (room_length / 2 + wall_thickness / 2, 0, wall_thickness, room_width),
            (0, -room_width / 2 - wall_thickness / 2, room_length + 2 * wall_thickness, wall_thickness),
            (0, room_width / 2 + wall_thickness / 2, room_length + 2 * wall_thickness, wall_thickness),
            ]:
            wall_poly = np.array([
                [x - length / 2, y - width / 2],  # bottom-left
                [x + length / 2, y - width / 2],  # bottom-right
                [x + length / 2, y + width / 2],  # top-right
                [x - length / 2, y + width / 2],  # top-left
            ])

            obs_info = {}
            obs_info['type'] = ObstacleType.BOUNDARY
            obs_info['vertices'] = wall_poly

            # convert np array to list
            wall_poly = [(x, y) for x, y in wall_poly]

            shape = pymunk.Poly(self.space.static_body, wall_poly, radius=0.1)
            shape.collision_type = 2
            shape.friction = 0.99

            self.space.add(shape)
            obs_dict.append(obs_info)
            wall_shapes.append(shape)

        return obs_dict, wall_shapes
    
    def generate_obstacles(self):
        obs_size = self.cfg.obstacle_size
        obstacles = []          # a list storing non-overlappin obstacle centers

        total_obs_required = self.cfg.num_obstacles
        self.num_box = self.cfg.num_obstacles
        obs_min_dist = self.cfg.min_obs_dist
        min_x = self.min_x_boundary + 1
        max_x = self.max_x_boundary - 1
        min_y = self.min_y_boundary + 1
        max_y = self.max_y_boundary - 1

        obs_count = 0
        while obs_count < total_obs_required:
            center_x = random.random() * (max_x - min_x) + min_x
            center_y = random.random() * (max_y - min_y) + min_y

            # loop through previous obstacles to check for overlap
            overlapped = False
            for prev_obs_x, pre_obs_y in obstacles:
                if ((center_x - prev_obs_x)**2 + (center_y - pre_obs_y)**2)**(0.5) <= obs_min_dist:
                    overlapped = True
                    break
            
            if not overlapped:
                obstacles.append([center_x, center_y])
                obs_count += 1

        ### DEBUG: Used for paper image
        if PAPER_RENDER_MODE:
            # obstacles = [[-3.5600568686136285, 2.882322061363647], [3.085977294836458, -3.901414500035961], [2.3334627567988138, 0.970017516489869], [2.6886525139225848, -0.8535647519508682], [-0.5116988994890237, -1.6805530789668408]]
            obstacles = [[3.868918186024633, 0.007132285722410536], [-3.7256488581466343, 3.422107836362504], [-0.8080131385400522, 0.18811131906768175]]

        
        # convert to obs dict
        obs_dict = []
        for obs_x, obs_y in obstacles:
            obs_info = {}
            obs_info['type'] = ObstacleType.DYNAMIC
            obs_info['centre'] = np.array([obs_x, obs_y])
            obs_info['vertices'] = np.array([[obs_x + obs_size, obs_y + obs_size], 
                                    [obs_x - obs_size, obs_y + obs_size], 
                                    [obs_x - obs_size, obs_y - obs_size], 
                                    [obs_x + obs_size, obs_y - obs_size]])
            obs_dict.append(obs_info)
        return obs_dict
        

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state and returns the initial observation."""

        if self.episode_idx is None:
            self.episode_idx = 0
        else:
            self.episode_idx += 1

        self.init_area_clearing_sim()
        self.init_area_clearing_env()

        # reset map
        # self.global_overhead_map = self.create_padded_room_ones()
        self.global_overhead_map = self.create_padded_room_zeros()
        self.update_global_overhead_map()

        if(self.goal_point_global_map is None):
            self.goal_point_global_map = self.create_global_shortest_path_to_goal_points()

        self.t = 0

        self.cleared_box_count = 0

        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.box_shapes)
        info = {'state': (round(self.agent.body.position.x, 2),
                                round(self.agent.body.position.y, 2),
                                round(self.agent.body.angle, 2)), 
                'total_work': self.total_work[0], 
                'obs': updated_obstacles, 
                'box_count': 0,
                'boundary': self.boundary_vertices,
                'walls': self.walls,
                'static_obstacles': self.static_obstacles,
                'goal_positions': self.goal_points}

        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)

        else:
            low_level_observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
            info['low_level_observation'] = low_level_observation
            
            observation = self.generate_observation()

        return observation, info
    

    def step(self, action):
        """Executes one time step in the environment and returns the result."""
        self.t += 1

        # initial pose
        robot_initial_position, robot_initial_heading = self.agent.body.position, restrict_heading_range(self.agent.body.angle)
        robot_initial_position = list(robot_initial_position)  

        robot_distance = 0
        robot_turn_angle = 0

        self.dp = None

        if self.demo_mode:

            if action == FORWARD:
                self.linear_speed = 0.3
            elif action == BACKWARD:
                self.linear_speed = -0.3
            elif action == STOP_TURNING:
                self.angular_speed = 0.0

            elif action == LEFT:
                self.angular_speed = 0.1
            elif action == RIGHT:
                self.angular_speed = -0.1

            elif action == SMALL_LEFT:
                self.angular_speed = 0.05
            elif action == SMALL_RIGHT:
                self.angular_speed = -0.05

            elif action == STOP:
                self.linear_speed = 0.0

            if abs(self.linear_speed) >= self.target_speed:
                self.linear_speed = self.target_speed*np.sign(self.linear_speed)

            # apply linear and angular speeds
            global_velocity = R(self.agent.body.angle) @ [self.linear_speed, 0]

            # apply velocity controller
            self.agent.body.angular_velocity = self.angular_speed
            self.agent.body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        elif self.cfg.agent.action_type == 'velocity':
            # apply velocity controller
            self.agent.body.angular_velocity = self.max_yaw_rate_step * action[1] / 2

            # apply linear and angular speeds
            scaled_vel = self.target_speed * action[0]
            global_velocity = R(self.agent.body.angle) @ [scaled_vel, 0]
            self.agent.body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        elif self.cfg.agent.action_type == 'position' or self.cfg.agent.action_type == 'heading':
            if self.cfg.agent.action_type == 'heading':
                ################################ Heading Control ################################
                # convert heading action to a pixel index in order to use the position control code

                # rescale heading action to be in range [0, 2*pi]
                angle = (action + 1) * np.pi + np.pi / 2
                step_size = self.cfg.agent.movement_step_size

                # calculate target position
                x_movement = step_size * np.cos(angle)
                y_movement = step_size * np.sin(angle)

                # convert target position to pixel coordinates
                x_pixel = int(self.local_map_pixel_width / 2 + x_movement * self.local_map_pixels_per_meter)
                y_pixel = int(self.local_map_pixel_width / 2 - y_movement * self.local_map_pixels_per_meter)

                # convert pixel coordinates to a single index
                action = y_pixel * self.local_map_pixel_width + x_pixel

            self.path, robot_move_sign = self.position_controller.get_waypoints_to_spatial_action(robot_initial_position, robot_initial_heading, action)
            # if self.cfg.render.show:
            #     self.renderer.update_path(self.path)

            robot_distance, robot_turn_angle = self.execute_robot_path(robot_initial_position, robot_initial_heading, robot_move_sign)

        # move simulation forward
        for _ in range(self.steps):
            self.space.step(self.dt / self.steps)
            
        collision_penalty = BOUNDARY_PENALTY if self.robot_hit_obstacle else 0
        
        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.box_shapes)
        num_completed, all_boxes_completed = self.boxes_completed(updated_obstacles, self.boundary_polygon, self.box_clearance_statuses)
        
        diff_reward = obs_to_goal_difference(self.prev_obs, updated_obstacles, self.goal_points, self.boundary_polygon)
        pushing_reward = diff_reward * BOX_PUSHING_REWARD_MULTIPLIER
        movement_reward = 0 if abs(diff_reward) > 0 else TIME_PENALTY

        box_completion_reward = 0
        if(num_completed > self.cleared_box_count):
            box_completion_reward = abs(num_completed - self.cleared_box_count) * BOX_CLEARED_REWARD
            self.t = 0 # reset time if box is cleared
        else:
            box_completion_reward = abs(num_completed - self.cleared_box_count) * BOX_PUTBACK_PENALTY
        if(self.cleared_box_count != num_completed):
            print("Boxes completed: ", num_completed)            
            self.cleared_box_count = num_completed
        
        # nonmovement penalty
        nonmovement_penalty = 0
        if not(self.cfg.agent.action_type == 'velocity') and (robot_distance < NONMOVEMENT_DIST_THRESHOLD and abs(robot_turn_angle) < NONMOVEMENT_TURN_THRESHOLD):
            nonmovement_penalty = NONMOVEMENT_PENALTY

        ### compute work done
        work = total_work_done(self.prev_obs, updated_obstacles)
        self.total_work[0] += work
        self.total_work[1].append(work)
        collision_reward = -work

        self.prev_obs = updated_obstacles
        self.obstacles = updated_obstacles

        # # check episode terminal condition
        if all_boxes_completed:
            terminated = True
        else:
            terminated = False

        reward = box_completion_reward + collision_penalty + pushing_reward + nonmovement_penalty
        truncated = self.t >= self.t_max

        # apply constraint penalty
        if truncated:
            reward += TRUNCATION_PENALTY
        # apply terminal reward
        elif terminated:
            reward += TERMINAL_REWARD
        
        done = terminated or truncated
        ministep_size = 2.5
        if(self.cfg.agent.action_type == 'position' or self.cfg.agent.action_type == 'heading'):
            ministeps = robot_distance / ministep_size
        else:
            ministeps = 1

        # Optionally, we can add additional info
        info = {'state': (round(self.agent.body.position.x, 2),
                                round(self.agent.body.position.y, 2),
                                round(self.agent.body.angle, 2)), 
                'total_work': self.total_work[0], 
                'collision reward': collision_reward, 
                'diff_reward': diff_reward,
                'box_completed_reward': box_completion_reward, 
                'obs': updated_obstacles,
                'box_completed_statuses': self.box_clearance_statuses,
                'box_count': num_completed,
                'ministeps': ministeps,}
        
        # generate observation
        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
        else:
            low_level_observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
            info['low_level_observation'] = low_level_observation
            
            observation = self.generate_observation()
            self.observation = observation

        self.update_global_overhead_map()
        self.robot_hit_obstacle = False
        
        return observation, reward, terminated, truncated, info
    
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
            heading_diff = heading_difference(robot_heading, robot_waypoint_heading)
            if np.abs(heading_diff) > TURN_STEP_SIZE and np.abs(heading_diff - prev_heading_diff) > 0.001:
                # turn towards next waypoint first
                robot_new_heading += np.sign(heading_diff) * TURN_STEP_SIZE
            else:
                done_turning = True
                dx = robot_waypoint_position[0] - robot_position[0]
                dy = robot_waypoint_position[1] - robot_position[1]
                if distance(robot_position, robot_waypoint_position) < MOVE_STEP_SIZE:
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
            robot_position, robot_heading = self.agent.body.position, restrict_heading_range(self.agent.body.angle)
            robot_position = list(robot_position)
            prev_heading_diff = heading_diff

            # stop moving if robot collided with obstacle
            if distance(robot_prev_waypoint_position, robot_position) > MOVE_STEP_SIZE:
                if self.robot_hit_obstacle:
                    # self.robot_hit_obstacle = False
                    robot_is_moving = False
                    break  # Note: self.robot_distance does not get not updated
            
            # stop if robot reached waypoint
            if (distance(robot_position, robot_waypoint_positions[robot_waypoint_index]) < WAYPOINT_MOVING_THRESHOLD
                and np.abs(robot_heading - robot_waypoint_headings[robot_waypoint_index]) < WAYPOINT_TURNING_THRESHOLD):
                
                # update distance moved
                robot_distance += distance(robot_prev_waypoint_position, robot_position)

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
                # self.observation = self.generate_observation()
                self.render()

            # break if robot is stuck
            if sim_steps > STEP_LIMIT:
                break

            if sim_steps % MAP_UPDATE_STEPS == 0:
                self.update_global_overhead_map()
        
        robot_heading = restrict_heading_range(self.agent.body.angle)
        robot_turn_angle = heading_difference(robot_initial_heading, robot_heading)
        return robot_distance, robot_turn_angle

    def apply_controller(self, omega, v):
        self.agent.body.angular_velocity = omega / 2
        # self.agent.body.velocity = (v).tolist()
        self.agent.body.velocity = (v*5).tolist()

    def generate_observation_low_dim(self, updated_obstacles):
        """
        The observation is a vector of shape (num_obstacles * 2) specifying the 2d position of the obstacles
        <obs1_x, obs1_y, obs2_x, obs2_y, ..., obsn_x, obsn_y>
        """
        observation = np.zeros((len(updated_obstacles) * 2))
        for i in range(len(updated_obstacles)):
            obs = updated_obstacles[i]
            center = np.abs(poly_centroid(obs))
            observation[i * 2] = center[0]
            observation[i * 2 + 1] = center[1]
        return observation


    def update_path(self, new_path):
        self.path = new_path
        # if(self.renderer):
        #     self.renderer.update_path(path=self.path)

    def generate_observation(self):
        self.update_global_overhead_map()
        
        # Overhead map
        channels = []
        obs_array_1 = self.get_local_map(self.global_overhead_map, self.agent.body.position, self.agent.body.angle)
        channels.append(self.scale_obs_to_image_space(obs_array_1))
        
        obs_array_2 = self.robot_state_channel.copy()
        channels.append(self.scale_obs_to_image_space(obs_array_2))

        obs_array_3 = self.get_local_distance_map(self.create_global_shortest_path_map(self.agent.body.position), self.agent.body.position, self.agent.body.angle)
        channels.append(self.scale_obs_to_image_space(obs_array_3))

        obs_array_4 = self.get_local_distance_map(self.goal_point_global_map, self.agent.body.position, self.agent.body.angle)
        channels.append(self.scale_obs_to_image_space(obs_array_4))

        try:
            # observation = np.stack(channels).astype(np.uint8)
            observation = np.stack(channels, axis=2).astype(np.uint8)
        except Exception as e:
            print(channels[0].shape, channels[1].shape)
            raise e
        return observation
    
    def scale_obs_to_image_space(self, obs_array):
        obs_array = (obs_array * 255).astype(np.uint8)
        return obs_array
    
    def create_padded_room_zeros(self):
        return np.zeros((
            int(2 * np.ceil((self.map_width * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2)),
            int(2 * np.ceil((self.map_height * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2))
        ), dtype=np.float32)
    
    def create_padded_room_ones(self):
        return np.ones((
            int(2 * np.ceil((self.map_width * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2)),
            int(2 * np.ceil((self.map_height * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2))
        ), dtype=np.float32)
    
    def update_global_overhead_map(self):
        small_overhead_map = self.small_obstacle_map.copy()
        small_overhead_map[small_overhead_map == 1] = FLOOR_SEG_INDEX/MAX_SEG_INDEX
        # self.global_overhead_map[self.global_overhead_map == 1] = FLOOR_SEG_INDEX/MAX_SEG_INDEX


        # generate receptacle boundaries
        inner_area_length = abs(self.boundary_vertices[0][0])*2
        inner_area_width = abs(self.boundary_vertices[0][1])*2
        receptacle_thickness = abs(self.outer_boundary_vertices[0][0]) - abs(self.boundary_vertices[0][0])
        
        for x, y, length, width in [
            (-inner_area_length / 2 - receptacle_thickness / 2, 0, receptacle_thickness, inner_area_width),
            (inner_area_length / 2 + receptacle_thickness / 2, 0, receptacle_thickness, inner_area_width),
            (0, -inner_area_width / 2 - receptacle_thickness / 2, inner_area_length + 2 * receptacle_thickness, receptacle_thickness),
            (0, inner_area_width / 2 + receptacle_thickness / 2, inner_area_length + 2 * receptacle_thickness, receptacle_thickness),
            ]:
            receptacle_poly = np.array([
                [x - length / 2, y - width / 2],  # bottom-left
                [x + length / 2, y - width / 2],  # bottom-right
                [x + length / 2, y + width / 2],  # top-right
                [x - length / 2, y + width / 2],  # top-left
            ])
            # convert world coordinates to pixel coordinates
            receptacle_poly_px = (receptacle_poly * self.local_map_pixels_per_meter).astype(np.int32)
            receptacle_poly_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            receptacle_poly_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            receptacle_poly_px[:, 1] = small_overhead_map.shape[0] - receptacle_poly_px[:, 1]
            # draw the boundary on the small_overhead_map
            fillPoly(small_overhead_map, [receptacle_poly_px], color=GOAL_AREA_SEG_INDEX/MAX_SEG_INDEX)

        for i in range(len(self.box_shapes)):
            poly = self.box_shapes[i]
            vertices = [poly.body.local_to_world(v) for v in poly.get_vertices()]
            vertices_np = np.array([[v.x, v.y] for v in vertices])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * self.local_map_pixels_per_meter).astype(np.int32)
            vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] = small_overhead_map.shape[0] - vertices_px[:, 1]

            # draw the boundary on the small_overhead_map
            if self.box_clearance_statuses[i]:
                fillPoly(small_overhead_map, [vertices_px], color=COMPLETED_CUBE_SEG_INDEX/MAX_SEG_INDEX)
            else:
                fillPoly(small_overhead_map, [vertices_px], color=CUBE_SEG_INDEX/MAX_SEG_INDEX)

        vertices = [self.agent.body.local_to_world(v) for v in self.agent_info.footprint_vertices]
        robot_vertices = np.array([[v.x, v.y] for v in vertices])
        robot_vertices_px = (robot_vertices * self.local_map_pixels_per_meter).astype(np.int32)
        robot_vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
        robot_vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
        robot_vertices_px[:, 1] = small_overhead_map.shape[0] - robot_vertices_px[:, 1]
        
        fillPoly(small_overhead_map, [robot_vertices_px], color=ROBOT_SEG_INDEX/MAX_SEG_INDEX)

        start_i, start_j = int(self.global_overhead_map.shape[0] / 2 - small_overhead_map.shape[0] / 2), int(self.global_overhead_map.shape[1] / 2 - small_overhead_map.shape[1] / 2)
        self.global_overhead_map[start_i:start_i + small_overhead_map.shape[0], start_j:start_j + small_overhead_map.shape[1]] = small_overhead_map

    def get_local_distance_map(self, global_map, robot_position, robot_heading):
        local_map = self.get_local_map(global_map, robot_position, robot_heading)
        local_map -= local_map.min() # move the min to 0 to make invariant to size of environment
        return local_map
    
    def get_local_map(self, global_map, robot_position, robot_heading):
        crop_width = round_up_to_even(self.local_map_pixel_width * np.sqrt(2))
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
    
    def create_global_shortest_path_map(self, robot_position):
        pixel_i, pixel_j = position_to_pixel_indices(robot_position[0], robot_position[1], self.configuration_space.shape, self.local_map_pixels_per_meter)
        pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
        global_map, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
        global_map /= self.local_map_pixels_per_meter
        global_map /= (np.sqrt(2) * self.local_map_pixel_width) / self.local_map_pixels_per_meter
        
        return global_map
    
    def create_global_shortest_path_to_goal_points(self):
        global_map = self.create_padded_room_zeros() + np.inf
        for point in self.goal_points:
            rx, ry = point.x, point.y
            pixel_i, pixel_j = position_to_pixel_indices(rx, ry, self.configuration_space.shape, self.local_map_pixels_per_meter)
            pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
            shortest_path_image, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
            shortest_path_image /= self.local_map_pixels_per_meter
            global_map = np.minimum(global_map, shortest_path_image)
        global_map /= (np.sqrt(2) * self.local_map_pixel_width) / self.local_map_pixels_per_meter

        max_value = np.max(global_map)
        min_value = np.min(global_map)
        global_map = (global_map - min_value) / (max_value - min_value) * DISTANCE_SCALE_MAX

        # fill points outside boundary polygon with 0
        for i in range(global_map.shape[0]):
            for j in range(global_map.shape[1]):
                x, y = pixel_indices_to_position(i, j, self.configuration_space.shape, self.local_map_pixels_per_meter)
                if not self.boundary_polygon.contains(Point(x, y)):
                    global_map[i, j] = 0
                if not self.outer_boundary_polygon.contains(Point(x, y)):
                    global_map[i, j] = 1
        

        global_map += 1 - self.configuration_space

        return global_map
    
    def closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]
    
    def update_configuration_space(self):
        """
        Obstacles are dilated based on the robot's radius to define a collision-free space
        """

        obstacle_map = self.create_padded_room_zeros()
        small_obstacle_map = np.zeros((self.local_map_pixel_width+20, self.local_map_pixel_width+20), dtype=np.float32)

        for poly in self.wall_shapes + self.static_obs_shapes:
            # get world coordinates of vertices
            vertices = [poly.body.local_to_world(v) for v in poly.get_vertices()]
            vertices_np = np.array([[v.x, v.y] for v in vertices])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * self.local_map_pixels_per_meter).astype(np.int32)
            vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] = small_obstacle_map.shape[0] - vertices_px[:, 1]

            fillPoly(small_obstacle_map, [vertices_px], color=1)
        
        start_i, start_j = int(obstacle_map.shape[0] / 2 - small_obstacle_map.shape[0] / 2), int(obstacle_map.shape[1] / 2 - small_obstacle_map.shape[1] / 2)
        obstacle_map[start_i:start_i + small_obstacle_map.shape[0], start_j:start_j + small_obstacle_map.shape[1]] = small_obstacle_map

        # Dilate obstacles and walls based on robot size
        robot_pixel_width = int(2 * self.robot_radius * self.local_map_pixels_per_meter)
        selem = disk(np.floor(robot_pixel_width / 4))
        self.configuration_space = 1 - binary_dilation(obstacle_map, selem).astype(np.float32)
        
        selem_thin = disk(np.floor(robot_pixel_width / 4))
        self.configuration_space_thin = 1 - binary_dilation(obstacle_map, selem_thin).astype(np.float32)

        self.closest_cspace_indices = distance_transform_edt(1 - self.configuration_space, return_distances=False, return_indices=True)
        self.small_obstacle_map = 1 - small_obstacle_map
    
    def boxes_completed(self, updated_obstacles, boundary_polygon, box_clearance_statuses):
        """
        Returns a tuple: (int: number of boxes completed, bool: whether pushing task is complete)
        """
        completed_count = 0
        completed = False

        for i in range(len(updated_obstacles)):
            obs = updated_obstacles[i]

            # if center[1] - self.cfg.obstacle_size >= self.cfg.goal_y:
            if not(boundary_polygon.intersects(Polygon(obs))):
                completed_count += 1
            box_clearance_statuses[i] = not(boundary_polygon.intersects(Polygon(obs)))
        
        if completed_count == self.num_box:
            completed = True
        
        return completed_count, completed

    def render(self, mode='human', close=False):
        """Renders the environment."""

        if self.t % self.cfg.anim.plot_steps == 0:

            path = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + '.png')
            if(self.renderer):
                self.renderer.render(save=True, path=path)

            if self.cfg.render.log_obs and not self.low_dim_state:

                for ax, i in zip(self.state_ax, range(self.num_channels)):
                    ax.clear()
                    ax.set_title(f'Channel {i}')
                    # im = ax.imshow(self.observation[i,:,:], cmap='hot', interpolation='nearest')
                    im = ax.imshow(self.observation[:,:,i], cmap='hot', interpolation='nearest')
                    if self.colorbars[i] is not None:
                        self.colorbars[i].update_normal(im)
                    else:
                        self.colorbars[i] = self.state_fig.colorbar(im, ax=ax)
                
                self.state_fig.savefig(os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + '_obs.png'))


                ### DEBUG: Seperate figures for paper
                
                # for i in range(self.num_channels):
                #     self.state_axes[i].clear()
                #     im = self.state_axes[i].imshow(self.observation[:,:,i], cmap='hot', interpolation='nearest')
                #     if self.colorbars[i] is not None:
                #         self.colorbars[i].update_normal(im)
                #     else:
                #         self.colorbars[i] = self.state_figs[i].colorbar(im, ax=self.state_axes[i])

                #     self.state_axes[i].axis('off')
                #     self.state_figs[i].savefig(os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + f'_obs_{i}.png'), bbox_inches='tight', pad_inches=0)

        else:
            if(self.renderer):
                self.renderer.render(save=False)


    def close(self):
        """Optional: close any resources or cleanup if necessary."""
        pass

# Helper functions

def restrict_heading_range(heading):
    return np.mod(heading + np.pi, 2 * np.pi) - np.pi

def distance(position1, position2):
    return np.linalg.norm(np.asarray(position1)[:2] - np.asarray(position2)[:2])

def heading_difference(heading1, heading2):
    return restrict_heading_range(heading1 - heading2)