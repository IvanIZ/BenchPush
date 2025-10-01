import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

import random
import pymunk
from pymunk import Vec2d
from matplotlib import pyplot as plt

# maze NAMO specific imports
from benchpush.common.cost_map import CostMap
from benchpush.common.evaluation.metrics import total_work_done
from benchpush.common.geometry.polygon import poly_area
from benchpush.common.robot import Robot
from benchpush.common.utils.renderer import Renderer
from benchpush.common.utils.sim_utils import generate_sim_obs, generate_sim_maze
from benchpush.common.geometry.polygon import poly_centroid
from benchpush.common.utils.utils import DotDict
from benchpush.common.occupancy_grid.occupancy_map import OccupancyGrid


R = lambda theta: np.asarray([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

YAW_CONSTRAINT_PENALTY = 0
BOUNDARY_PENALTY = -50
TERMINAL_REWARD = 200

FORWARD = 0
STOP_TURNING = 1
LEFT = 2
RIGHT = 3
STOP = 4
OTHER = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7
BACKWARD = 8

class MazeNAMO(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, cfg=None):
        super(MazeNAMO, self).__init__()

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

        # get correct maze version
        if self.cfg.maze_version == 1:
            self.cfg.env = self.cfg.env1
        elif self.cfg.maze_version == 2:
            self.cfg.env = self.cfg.env2
        else:
            raise Exception("Invalid Maze Version!")

        grid_size = 1 / self.cfg.occ.m_to_pix_scale
        self.occupancy = OccupancyGrid(grid_width=grid_size, grid_height= grid_size, map_width=self.cfg.env.width, map_height=self.cfg.env.length, 
                                       local_width=self.cfg.occ.local_width, local_height=self.cfg.occ.local_height,
                                       ship_body=None, meter_to_pixel_scale=self.cfg.occ.m_to_pix_scale)

        self.beta = 1.5         # amount to scale the collision reward
        self.k = 2        # amount to scale the distance reward
        self.k_increment = 150
        self.episode_idx = None     # the increment of this index is handled in reset()

        self.path = None
        self.scatter = False

        self.low_dim_state = self.cfg.low_dim_state

        
        self.env_max_trial = 4000
        
        self.prev_dist_value = None

        #robot head and tail for orientation map
        self.robot_head = (self.cfg.robot.vertices[0][0]+self.cfg.robot.vertices[3][0])/2, (self.cfg.robot.vertices[0][1]+self.cfg.robot.vertices[3][1])/2
        self.robot_tail = (self.cfg.robot.vertices[1][0]+self.cfg.robot.vertices[2][0])/2, (self.cfg.robot.vertices[1][1]+self.cfg.robot.vertices[2][1])/2
        # Define action space
        self.max_linear_speed = 1.0
        self.min_linear_speed = 0.0
        self.max_yaw_rate_step = (np.pi/2) / 15        # rad/sec
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float64)
        
        # Define observation space
        self.low_dim_state = self.cfg.low_dim_state
        if self.low_dim_state:
            #low dimensional observation space comprises of the 2D positions of each obstacle in addition to the robot
            self.fixed_trial_idx = self.cfg.fixed_trial_idx
            if self.cfg.randomize_obstacles:
                self.observation_space = spaces.Box(low=-10, high=30, shape=((self.cfg.num_obstacles+1) * 2,), dtype=np.float64) 
            else:
                self.observation_space = spaces.Box(low=-10, high=30, shape=(8,), dtype=np.float64) # 8 for 3 obstacles and the robot
        
        else:
            #high dimensional observation space comprises of the occupancy grid map with 4 channels
            #each channel represnets a local moving window where the agent is at the center
            #channel dimensions are (local_window_height, local_window_width) 
            #example if the local window is 10 meters by 10 meters, and the grid size is 0.1 meters, then the channel dimensions are (100, 100)
            #channel 1 - occupancy grid map with fixed obstacles
            #channel 2 - occupancy grid map with movable obstacles
            #channel 3 - occupancy grid map with robot footprint
            #channel 4 - distance map to the goal point
            self.observation_shape = (4, self.occupancy.local_window_height, self.occupancy.local_window_width)
            self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        self.yaw_lim = (0, np.pi)       # lower and upper limit of ship yaw  
        self.boundary_violation_limit = 0.0       # if the ship is out of boundary more than this limit, terminate and truncate the episode 

        self.renderer = None
        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))

        #robot and obstacles occupancy grid
        self.occupancy_plot = plt
        self.occupancy_plot.ion()
        
   
    def init_maze_NAMO_sim(self):

        # initialize maze environment
        self.steps = self.cfg.sim.steps
        self.dt = self.cfg.dt
        self.target_speed = self.cfg.target_speed

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

        # setup a collision callback to keep track of total ke
        # def pre_solve_handler(arbiter, space, data):
        #     nonlocal ship_ke
        #     ship_ke = arbiter.shapes[0].body.kinetic_energy
        #     print('ship_ke', ship_ke, 'mass', arbiter.shapes[0].body.mass, 'velocity', arbiter.shapes[0].body.velocity)
        #     return True
        # # http://www.pymunk.org/en/latest/pymunk.html#pymunk.Body.each_arbiter

        # setup pymunk collision callbacks
        def pre_solve_handler(arbiter, space, data):
            obstacle_body = arbiter.shapes[1].body
            obstacle_body.pre_collision_KE = obstacle_body.kinetic_energy  # hacky, adding a field to pymunk body object
            return True

        def post_solve_handler(arbiter, space, data):
            # nonlocal self.total_ke, self.system_ke_loss, self.total_impulse, self.clln_obs
            robot_shape, obstacle_shape = arbiter.shapes

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

        def pre_solve_handler_walls(arbiter, space, data):
            self.wall_collision = True
            return True

        # handler = space.add_default_collision_handler()
        self.handler = self.space.add_collision_handler(1, 2)
        #handle collision between robot and walls (ternimate the episode)
        self.handler_robot_wall = self.space.add_collision_handler(1, 3)
        # from pymunk docs
        # post_solve: two shapes are touching and collision response processed
        self.handler.pre_solve = pre_solve_handler
        self.handler.post_solve = post_solve_handler
        # pre handler for robot-wall collision
        self.handler_robot_wall.pre_solve = pre_solve_handler_walls

        if self.renderer is not None:
            self.renderer.reset(new_space=self.space)
        
    def init_maze_NAMO_env(self):

        #initialize the maze walls in a list (temporary)
        self.construct_maze_walls()
        #add maze walls to pymunk
        generate_sim_maze(self.space, self.maze_walls)
        
        # generate random start point, if specified and avoid maze walls
        if self.cfg.random_start:
            while True:
                x_start = 1 + random.random() * (self.cfg.start_x_range - 1)
                y_start = 1 + random.random() * (self.cfg.start_y_range - 1)
                #check if the start and goal points are not in the maze walls
                min_dist = self.cfg.robot.min_obstacle_dist
                if not self.space.point_query((x_start, y_start), min_dist, pymunk.ShapeFilter()): 
                    break

            self.start = (x_start, y_start,np.pi*3/2)
        else:
            
            if self.cfg.maze_version == 1:
                self.start = (11.25, 3.75, np.pi / 2)         # for 15x15, v1

            elif self.cfg.maze_version == 2:
                self.start = (16.66, 16.66, 3 * np.pi / 2)         # for 15x15, v2

        self.obs_dicts = self.generate_obstacles()
        
        # filter out obstacles that have zero area
        self.obs_dicts[:] = [ob for ob in self.obs_dicts if poly_area(ob['vertices']) != 0]
        self.obstacles = [ob['vertices'] for ob in self.obs_dicts]

        self.goal = (self.cfg.env.goal_x, self.cfg.env.goal_y)
        
        # initialize ship sim objects
        self.polygons = generate_sim_obs(self.space, self.obs_dicts, self.cfg.sim.obstacle_density, color=(204, 153, 102, 255))
        for p in self.polygons:
            p.collision_type = 2

        self.robot_body, self.robot_shape, self.wheels = Robot.sim(self.cfg.robot.vertices, self.start, body_type=pymunk.Body.KINEMATIC, color=(100, 100, 100, 255), wheel_vertices_list=self.cfg.robot.wheel_vertices)
        self.robot_shape.collision_type = 1
        self.space.add(self.robot_body, self.robot_shape, self.wheels[0], self.wheels[1], self.wheels[2], self.wheels[3])

        #compute the global distance map
        self.global_distance_map, self.unnormalized_dist_map = self.occupancy.global_goal_point_dist_transform(self.goal, self.maze_walls)
        # run initial simulation steps to let environment settle
        for _ in range(1000):
            self.space.step(self.dt / self.steps)
        self.prev_obs = CostMap.get_obs_from_poly(self.polygons)
     
    def generate_obstacles(self):
        obs_size = self.cfg.obstacle_size
        obstacles = []          # a list storing non-overlappin obstacle centers

        if self.cfg.randomize_obstacles:
            total_obs_required = self.cfg.num_obstacles
            self.num_box = self.cfg.num_obstacles
            obs_min_dist = self.cfg.min_obs_dist
            min_x = 0
            max_x = self.cfg.env.length
            min_y = 0
            max_y = self.cfg.env.length

            obs_count = 0
            while obs_count < total_obs_required:
                center_x = random.random() * (max_x - min_x) + min_x
                center_y = random.random() * (max_y - min_y) + min_y

                # loop through previous obstacles to check for overlap with other obstacles or maze walls
                overlapped = False
                for prev_obs_x, pre_obs_y in obstacles:
                    if ((center_x - prev_obs_x)**2 + (center_y - pre_obs_y)**2)**(0.5) <= obs_min_dist:
                        overlapped = True
                        break
                    if self.space.point_query((center_x, center_y), obs_min_dist, pymunk.ShapeFilter()):
                        overlapped = True
                        break
                
                if not overlapped:
                    obstacles.append([center_x, center_y])
                    obs_count += 1
        
        else:
            obstacles.append([8.5, 11])
            obstacles.append([10, 9])
            obstacles.append([11.25, 11.5])
            
            obstacles.append([6, 10])
            obstacles.append([3.5, 8.5])
            self.num_box = 3
        
        # convert to obs dict
        obs_dict = []
        for obs_x, obs_y in obstacles:
            obs_info = {}
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

        self.init_maze_NAMO_sim()
        self.init_maze_NAMO_env()

        self.t = 0
        self.prev_dist_value = None

        #reset termination flags
        self.wall_collision = False

        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        info = {'state': (round(self.robot_body.position.x, 2),
                                round(self.robot_body.position.y, 2),
                                round(self.robot_body.angle, 2)), 
                'total_work': self.total_work[0], 
                'obs': updated_obstacles, 
                'box_count': 0, 
                'goal_dt': self.unnormalized_dist_map, 
                'm_to_pix_scale': self.cfg.occ.m_to_pix_scale}

        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)

        else:
            observation = self.generate_observation()
        return observation, info

          
    def construct_maze_walls(self):
        self.length = self.cfg.env.length
        self.width = self.cfg.env.width
        self.maze_version = self.cfg.maze_version
        if self.maze_version == 1:
            self.maze_walls = [[(0,0),(self.width,0)] , [(0,0),(0,self.length)],
                    [(self.width,0),(self.width,self.length)], 
                    [(0,self.length),(self.width,self.length)],
                     [(2*self.width/2,self.length),(2*self.width/2,5)],
                    [(self.width/2,0),(self.width/2,self.length - self.length/3)]]
        elif self.maze_version == 2:
            self.maze_walls = [[(0,0),(self.width,0)] , [(0,0),(0,self.length)],
                    [(self.width,0),(self.width,self.length)], 
                    [(0,self.length),(self.width,self.length)],
                    [(self.width/3,0),(self.width/3,2*self.length/3)], [(2*self.width/3,self.length),(2*self.width/3, self.length/3)]]
        else:
            #abort the program
            print("Invalid maze version")
            exit(1)

    def randomize_obstacles(self):
        """
        NOTE this function is called only when using low-dimensional observation
        """
        for obs in self.obs_dicts:
            prev_centre = np.array(obs['centre'])

            rand_x = 0.5 + random.random() * (self.cfg.env.length - 0.5)
            rand_y = 1 + random.random() * (self.cfg.env.length - 1)
            new_centre = np.array([rand_x, rand_y])

            # translate vertices and reset center
            obs['vertices'] = obs['vertices'] - prev_centre + new_centre
            obs['centre'] = new_centre
    
    def get_distance_value(self, x, y):
        robot_pose = (self.robot_body.position.x, self.robot_body.position.y, self.robot_body.angle)
        robot_pixel_x = int(robot_pose[0] * self.cfg.occ.m_to_pix_scale) 
        robot_pixel_y = int(robot_pose[1] * self.cfg.occ.m_to_pix_scale)
        return self.global_distance_map[robot_pixel_y, robot_pixel_x]

    def step(self, action):
        """Executes one time step in the environment and returns the result."""
        self.t += 1

        global_velocity = R(self.robot_body.angle) @ [self.target_speed, 0]           # fixed linear forward velocity
        action = action * self.max_yaw_rate_step                        # scaling from [-1, 1] to [-max_yaw_rate_step, max_yaw_rate_step]
        self.robot_body.angular_velocity = action

        self.robot_body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        # move simulation forward
        boundary_constraint_violated = False
        for _ in range(self.steps):
            self.space.step(self.dt / self.steps)

            # apply boundary constraints
            if self.robot_body.position.x < 0 or self.robot_body.position.x > self.cfg.env.width:
                boundary_constraint_violated = True
        
        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        

        # compute work done
        work = total_work_done(self.prev_obs, updated_obstacles)
        self.total_work[0] += work
        self.total_work[1].append(work)
        self.prev_obs = updated_obstacles
        self.obstacles = updated_obstacles

        # check episode terminal condition
        if self.goal_is_reached() or self.wall_collision:
            terminated = True
        else:
            terminated = False

        # compute reward
        if self.robot_body.position.x != self.goal[0] or self.robot_body.position.y != self.goal[1]:
            dist_value = self.get_distance_value(self.robot_body.position.x, self.robot_body.position.y)

            if self.prev_dist_value is None:
                dist_increment_reward = 0
            else:
                dist_increment_reward = (self.prev_dist_value - dist_value)*self.k_increment
            self.prev_dist_value = dist_value
        else:
            dist_value = self.get_distance_value(self.robot_body.position.x, self.robot_body.position.y)
            dist_increment_reward = 0
        collision_reward = -work

        reward = self.beta * collision_reward + dist_increment_reward

        # apply constraint penalty
        if boundary_constraint_violated or self.wall_collision:
            reward += BOUNDARY_PENALTY

        # apply terminal reward
        trial_success = False
        if terminated and not self.wall_collision:
            reward += TERMINAL_REWARD
            trial_success = True
        
        # Optionally, we can add additional info
        info = {'state': (round(self.robot_body.position.x, 2),
                                round(self.robot_body.position.y, 2),
                                round(self.robot_body.angle, 2)), 
                'total_work': self.total_work[0], 
                'collision reward': collision_reward, 
                'scaled collision reward': collision_reward * self.beta, 
                'dist increment reward': dist_increment_reward, 
                'trial_success': trial_success,
                'obs': updated_obstacles, 
               }    
        
        # generate observation
        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
        else:
            observation = self.generate_observation()

        if self.cfg.log_obs:
            self.log_observation()
        
        return observation, reward, terminated, False, info


    def generate_observation_low_dim(self, updated_obstacles):
        """
        The observation is a vector of shape (num_obstacles * 2)+ 2 specifying the 2d position of the obstacles and the robot
        <robot_x, robot_y, obs1_x, obs1_y, obs2_x, obs2_y, ..., obsn_x, obsn_y>
        """
        # print("num obs: ", len(updated_obstacles))
        observation = np.zeros(((len(updated_obstacles)+1) * 2))
        #robot position
        observation[0] = self.robot_body.position.x
        observation[1] = self.robot_body.position.y
        #obstacle positions
        for i in range(1,len(updated_obstacles)):
            obs = updated_obstacles[i]
            center = np.abs(poly_centroid(obs))
            observation[i * 2] = center[0]
            observation[i * 2 + 1] = center[1]
        return observation


    def update_path(self, new_path, scatter=False):
        if scatter:
            self.scatter = True
        self.path = new_path
        self.renderer.update_path(path=self.path)
    

    def generate_observation(self):
        #Compute Binary Occupancy Grids
        #get robot state
        robot_pose = (self.robot_body.position.x, self.robot_body.position.y, self.robot_body.angle)
        
        robot_footprint_local , movable_obstacles_local, wall_local, distance_map_local = self.occupancy.ego_view_map_maze(robot_pose, 
                                                                self.cfg.robot.vertices,self.obstacles, self.maze_walls, self.global_distance_map)
        observation = np.concatenate((np.array([robot_footprint_local]), 
                                      np.array([movable_obstacles_local]), np.array([wall_local]), np.array([distance_map_local])))  # (5, local H, local W)
        #for resnet input
        observation = (observation*255).astype(np.uint8)
        return observation


    def goal_is_reached(self):
        #check if the goal is within the robot's dimensions
        robot_x = self.robot_body.position.x
        robot_y = self.robot_body.position.y
        goal_dist = ((robot_x - self.goal[0])**2 + (robot_y - self.goal[1])**2)**(0.5)
        if goal_dist <= self.cfg.goal_radius + self.cfg.robot.min_r:
            return True
        else:
            return False

    
    def log_observation(self):
        robot_footprint, movable_obs, fixed_obs, distance_map = self.generate_observation()

        directory = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
        if directory:
            os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist

        # visualize footprint
        self.con_ax.clear()
        occ_map_render = np.copy(robot_footprint)
        occ_map_render = np.flip(occ_map_render, axis=0)
        self.con_ax.imshow(occ_map_render, cmap='gray')
        self.con_ax.axis('off')
        save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
        fp = os.path.join(save_fig_dir, str(self.t) + '_footprint.png')
        self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

        #visualize movable obstacles
        self.con_ax.clear()
        occ_map_render = np.copy(movable_obs)
        occ_map_render = np.flip(occ_map_render, axis=0)
        self.con_ax.imshow(occ_map_render, cmap='gray')
        self.con_ax.axis('off')
        save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
        fp = os.path.join(save_fig_dir, str(self.t) + '_movable_obs.png')
        self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)
        
        #visualize fixed obstacles
        self.con_ax.clear()
        occ_map_render = np.copy(fixed_obs)
        occ_map_render = np.flip(occ_map_render, axis=0)
        self.con_ax.imshow(occ_map_render, cmap='gray')
        self.con_ax.axis('off')
        save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
        fp = os.path.join(save_fig_dir, str(self.t) + '_fixed_obs.png')
        self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

        #visualize distance map
        self.con_ax.clear()
        occ_map_render = np.copy(self.global_distance_map)
        occ_map_render = np.flip(occ_map_render, axis=0)
        self.con_ax.imshow(occ_map_render, cmap='gray')
        self.con_ax.axis('off')
        save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
        fp = os.path.join(save_fig_dir, str(self.t) + '_distance_map.png')
        self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

        #local distance map
        self.con_ax.clear()
        occ_map_render = np.copy(distance_map)
        occ_map_render = np.flip(occ_map_render, axis=0)
        self.con_ax.imshow(occ_map_render, cmap='gray')
        self.con_ax.axis('off')
        save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
        fp = os.path.join(save_fig_dir, str(self.t) + '_local_distance_map.png')
        self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

    
    def render(self, mode='human', close=False):
        """Renders the environment."""

        if self.renderer is None:
            self.renderer = Renderer(self.space, env_width=self.cfg.env.width, env_height=self.cfg.env.length, render_scale=self.cfg.render_scale, 
                    background_color=(200, 200, 200), caption="Maze", goal_region=((self.cfg.env.goal_x, self.cfg.env.goal_y), self.cfg.goal_radius))

        if self.cfg.render_snapshot:
            path = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + '.png')
            self.renderer.render(save=True, path=path)
        else:
            self.renderer.render(save=False)


    def close(self):
        """Optional: close any resources or cleanup if necessary."""
        pass
