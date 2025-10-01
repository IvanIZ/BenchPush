import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import time

import pickle
import random
import pymunk
from pymunk import Vec2d
from matplotlib import pyplot as plt

# ship-ice related imports
from benchpush.common.cost_map import CostMap
from benchpush.common.evaluation.metrics import total_work_done
from benchpush.common.geometry.polygon import poly_area
from benchpush.common.ship import Ship
from benchpush.common.utils.renderer import Renderer
from benchpush.common.utils.sim_utils import generate_sim_obs
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

class ShipIceEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, cfg=None):
        super(ShipIceEnv, self).__init__()

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
        
        self.local_window_v_shift = 2

        self.beta = 30         # amount to scale the collision reward
        self.directional_reward_scale = 1.0

        self.episode_idx = None     # the increment of this index is handled in reset()

        self.goal = (0, self.cfg.goal_y)
        self.path = None

        self.low_dim_state = self.cfg.low_dim_state

        # Define action space
        self.max_yaw_rate_step = (np.pi/2) / 7        # rad/sec
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32)
        
        # load ice field environment
        assert self.cfg.concentration in [0.1, 0.2, 0.3, 0.4, 0.5], print("PLease check environment config. Concentration value should be set to one of the followings: 0.1, 0.2, 0.3, 0.4, 0.5")
        ice_file = os.path.join(self.current_dir, 'ice_environments', 'experiments_' + str(int(self.cfg.concentration * 100)) + '_100_r06_d40x12.pk')
        ddict = pickle.load(open(ice_file, 'rb'))

        self.experiment = ddict['exp'][self.cfg.concentration]
        self.env_max_trial = len(self.experiment)
        
        # Define observation space
        if self.low_dim_state:
            self.fixed_trial_idx = self.cfg.fixed_trial_idx
            init_queue={**self.experiment[self.fixed_trial_idx]}
            _, _, self.obs_dicts = init_queue.values()
            self.observation_space = spaces.Box(low=-10, high=30, shape=(len(self.obs_dicts) * 2,), dtype=np.float64)

        else:

            if self.cfg.egocentric_obs:
                grid_size = 1 / self.cfg.occ.m_to_pix_scale
                self.occupancy = OccupancyGrid(grid_width=grid_size, grid_height=grid_size, map_width=self.cfg.occ.map_width, map_height=self.cfg.occ.map_height, local_width=6, local_height=6, ship_body=None, meter_to_pixel_scale=self.cfg.occ.m_to_pix_scale)
                obs_shape = (4, self.occupancy.local_window_height, self.occupancy.local_window_width)
            
            else:
                self.occupancy = OccupancyGrid(grid_width=0.2, grid_height=0.2, map_width=self.cfg.occ.map_width, map_height=self.cfg.occ.map_height, local_width=6, local_height=6, ship_body=None, meter_to_pixel_scale=self.cfg.occ.m_to_pix_scale)
                obs_shape = (2, self.occupancy.occ_map_height, self.occupancy.occ_map_width)
            self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

        self.yaw_lim = (0, np.pi)       # lower and upper limit of ship yaw  
        self.boundary_violation_limit = 0.0       # if the ship is out of boundary more than this limit, terminate and truncate the episode 

        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))

        self.renderer = None
        

    def init_ship_ice_sim(self):

        # initialize ship-ice environment
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
            ice_body = arbiter.shapes[1].body
            ice_body.pre_collision_KE = ice_body.kinetic_energy  # hacky, adding a field to pymunk body object
            return True

        def post_solve_handler(arbiter, space, data):
            # nonlocal self.total_ke, self.system_ke_loss, self.total_impulse, self.clln_obs
            ship_shape, ice_shape = arbiter.shapes

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

        # handler = space.add_default_collision_handler()
        self.handler = self.space.add_collision_handler(1, 2)
        # from pymunk docs
        # post_solve: two shapes are touching and collision response processed
        self.handler.pre_solve = pre_solve_handler
        self.handler.post_solve = post_solve_handler

        if self.renderer is not None:
            self.renderer.reset(new_space=self.space)
        
        
    def init_ship_ice_env(self):

        trial_idx = self.episode_idx % self.env_max_trial       # this will warp around the environment trial after reaching the last one
        if self.low_dim_state:
            trial = self.experiment[self.fixed_trial_idx]  # 0.5 is the concentration, 0 is the trial number
        else:
            trial = self.experiment[trial_idx]  # 0.5 is the concentration, 0 is the trial number

        init_queue={
                    **trial
                }
        
        _, self.start, self.obs_dicts = init_queue.values()

        # generate random start point, if specified
        if self.cfg.random_start:
            x_start = 1 + random.random() * (self.cfg.start_x_range - 1)    # [1, start_x_range]
            self.start = (x_start, 1.0, np.pi / 2)
        
        # filter out obstacles that have zero area
        self.obs_dicts[:] = [ob for ob in self.obs_dicts if poly_area(ob['vertices']) != 0]
        self.obstacles = [ob['vertices'] for ob in self.obs_dicts]

        # initialize ship sim objects
        self.polygons = generate_sim_obs(self.space, self.obs_dicts, self.cfg.sim.obstacle_density, color=(173, 216, 230, 255))
        for p in self.polygons:
            p.collision_type = 2

        self.ship_body, self.ship_shape = Ship.sim(self.cfg.ship.vertices, self.start, color=(64, 64, 64, 255))
        self.ship_shape.collision_type = 1
        self.space.add(self.ship_body, self.ship_shape)
        # run initial simulation steps to let environment settle
        for _ in range(1000):
            self.space.step(self.dt / self.steps)
        self.prev_obs = CostMap.get_obs_from_poly(self.polygons)
        

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state and returns the initial observation."""

        if self.episode_idx is None:
            self.episode_idx = 0
        else:
            self.episode_idx += 1

        self.init_ship_ice_sim()
        self.init_ship_ice_env()

        self.t = 0

        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        info = {'state': (round(self.ship_body.position.x, 2),
                                round(self.ship_body.position.y, 2),
                                round(self.ship_body.angle, 2)), 
                'total_work': self.total_work[0], 
                'obs': updated_obstacles}

        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)

        else:
            observation = self.generate_observation()
        return observation, info


    def _directional_reward(self):
        goal_vector = np.array([0, 1])
        robot_heading = np.array([np.cos(self.ship_body.angle), np.sin(self.ship_body.angle)])

        cos_phi = np.dot(robot_heading, goal_vector)

        return self.directional_reward_scale * cos_phi
    

    def step(self, action):
        """Executes one time step in the environment and returns the result."""
        self.t += 1

        action = action * self.max_yaw_rate_step

        self.target_speed = self.cfg.target_speed

        # constant forward speed in global frame
        global_velocity = R(self.ship_body.angle) @ [self.target_speed, 0]

        # apply velocity controller
        self.ship_body.angular_velocity = action
        self.ship_body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        # move simulation forward
        yaw_constraint_violated = False
        boundary_constraint_violated = False
        boundary_violation_terminal = False      # if out of boundary for too much, terminate and truncate the episode
        for _ in range(self.steps):
            self.space.step(self.dt / self.steps)

            # apply yaw constraints
            if self.ship_body.angle <= self.yaw_lim[0] or self.ship_body.angle >= self.yaw_lim[1]:
                self.ship_body.angular_velocity = 0.0
                yaw_constraint_violated = True

            # apply boundary constraints
            if self.ship_body.position.x < 0 or self.ship_body.position.x > self.cfg.occ.map_width:
                boundary_constraint_violated = True
        if self.ship_body.position.x < 0 and abs(self.ship_body.position.x - 0) >= self.boundary_violation_limit:
            boundary_violation_terminal = True
        if self.ship_body.position.x > self.cfg.occ.map_width and abs(self.ship_body.position.x - self.cfg.occ.map_width) >= self.boundary_violation_limit:
            boundary_violation_terminal = True
            

        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)

        # compute work done
        work = total_work_done(self.prev_obs, updated_obstacles)
        self.total_work[0] += work
        self.total_work[1].append(work)
        self.prev_obs = updated_obstacles
        self.obstacles = updated_obstacles

        # check episode terminal condition
        if self.ship_body.position.y >= self.goal[1]:
            terminated = True
        elif boundary_violation_terminal:
            terminated = True
        else:
            terminated = False

        # compute reward
        if self.ship_body.position.y < self.goal[1]:
            dist_reward = self._directional_reward()
        else:
            dist_reward = 0
        collision_reward = -work

        reward = self.beta * collision_reward + dist_reward

        # apply constraint penalty
        if yaw_constraint_violated:
            reward += YAW_CONSTRAINT_PENALTY
        if boundary_constraint_violated:
            reward += BOUNDARY_PENALTY

        # apply terminal reward
        trial_success = False
        if terminated and not boundary_violation_terminal:
            reward += TERMINAL_REWARD
            trial_success = True

        # Optionally, we can add additional info
        info = {'state': (round(self.ship_body.position.x, 2),
                                round(self.ship_body.position.y, 2),
                                round(self.ship_body.angle, 2)), 
                'total_work': self.total_work[0], 
                'collision reward': collision_reward, 
                'scaled collision reward': collision_reward * self.beta, 
                'dist reward': dist_reward, 
                'trial_success': trial_success,
                'obs': updated_obstacles}
        
        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
        else:
            observation = self.generate_observation()

        if self.cfg.log_obs:
            self.log_observation()

        return observation, reward, terminated, False, info


    def generate_observation_low_dim(self, updated_obstacles):
        """
        The observation is a vector of shape (num_obstacles * 2) specifying the 2d position of the obstacles
        <obs1_x, obs1_y, obs2_x, obs2_y, ..., obsn_x, obsn_y>
        """
        # print("num obs: ", len(updated_obstacles))
        observation = np.zeros((len(updated_obstacles) * 2))
        for i in range(len(updated_obstacles)):
            obs = updated_obstacles[i]
            center = np.abs(poly_centroid(obs))
            observation[i * 2] = center[0]
            observation[i * 2 + 1] = center[1]
        return observation


    def update_path(self, new_path):
        self.path = new_path
        self.renderer.update_path(path=self.path)
    

    def generate_observation(self):

        if self.cfg.egocentric_obs:
            ship_pose = (self.ship_body.position.x, self.ship_body.position.y, self.ship_body.angle)

            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.obstacles, 
                            ice_binary_w=int(self.occupancy.map_width * self.cfg.occ.m_to_pix_scale), 
                            ice_binary_h=int(self.occupancy.map_height * self.cfg.occ.m_to_pix_scale), 
                            local_range=[12, 12], ship_state=ship_pose)
            occupancy = self.occupancy.ego_view_obstacle_map(raw_ice_binary=raw_ice_binary, ship_state=ship_pose, vertical_shift=self.local_window_v_shift)
            local_footprint = self.occupancy.ego_view_footprint(ship_state=ship_pose, ship_vertices=self.cfg.ship.vertices, vertical_shift=self.local_window_v_shift)
            local_edt = self.occupancy.ego_view_goal_dist_transform(goal_y=self.goal[1], ship_state=ship_pose, vertical_shift=self.local_window_v_shift)
            local_orientation_map = self.occupancy.ego_view_orientation_map(ship_state=ship_pose, head=self.cfg.ship.head, tail=self.cfg.ship.tail, vertical_shift=self.local_window_v_shift)
            
            observation = np.concatenate((np.array([local_footprint]), np.array([local_edt]), np.array([local_orientation_map]), np.array([occupancy])))          # (2, H, W)
        
        else:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.obstacles, 
                        ice_binary_w=int(self.occupancy.map_width * self.cfg.occ.m_to_pix_scale), 
                        ice_binary_h=int(self.occupancy.map_height * self.cfg.occ.m_to_pix_scale))
            self.occupancy.compute_con_gridmap(raw_ice_binary=raw_ice_binary, save_fig_dir=None)
            occupancy = np.copy(self.occupancy.occ_map)         # (H, W)

            # compute footprint observation  NOTE: here we want unscaled, unpadded vertices
            ship_pose = (self.ship_body.position.x, self.ship_body.position.y, self.ship_body.angle)
            self.occupancy.compute_ship_footprint_planner(ship_state=ship_pose, ship_vertices=self.cfg.ship.vertices)
            footprint = np.copy(self.occupancy.footprint)       # (H, W)

            observation = np.concatenate((np.array([occupancy]), np.array([footprint])))          # (2, H, W)
        
        observation = (observation*255).astype(np.uint8)                                      # NOTE SB3 image format
        return observation


    def log_observation(self):
        
        directory = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
        if directory:
            os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist

        if self.cfg.egocentric_obs:

            # visualize local occupancy map
            self.con_ax.clear()
            occ_map_render = np.copy(self.occupancy.local_obstacle_map)
            occ_map_render = np.flip(occ_map_render, axis=0)
            self.con_ax.imshow(occ_map_render, cmap='gray')
            self.con_ax.axis('off')
            save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
            fp = os.path.join(save_fig_dir, str(self.t) + '_con.png')
            self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

            # visualize local orientation map
            self.con_ax.clear()
            occ_map_render = np.copy(self.occupancy.local_orientation)
            occ_map_render = np.flip(occ_map_render, axis=0)
            self.con_ax.imshow(occ_map_render, cmap='gray')
            self.con_ax.axis('off')
            save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
            fp = os.path.join(save_fig_dir, str(self.t) + '_orientation.png')
            self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

            # visualize local dist transform map
            self.con_ax.clear()
            occ_map_render = np.copy(self.occupancy.local_edt)
            occ_map_render = np.flip(occ_map_render, axis=0)
            self.con_ax.imshow(occ_map_render, cmap='gray')
            self.con_ax.axis('off')
            save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
            fp = os.path.join(save_fig_dir, str(self.t) + '_edt.png')
            self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

            # visualize local footprint
            self.con_ax.clear()
            occ_map_render = np.copy(self.occupancy.local_footprint)
            occ_map_render = np.flip(occ_map_render, axis=0)
            self.con_ax.imshow(occ_map_render, cmap='gray')
            self.con_ax.axis('off')
            save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
            fp = os.path.join(save_fig_dir, str(self.t) + '_footprint.png')
            self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

        else:
            # visualize global occupancy map
            self.con_ax.clear()
            occ_map_render = np.copy(self.occupancy.occ_map)
            occ_map_render = np.flip(occ_map_render, axis=0)
            self.con_ax.imshow(occ_map_render, cmap='gray')
            self.con_ax.axis('off')
            save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
            fp = os.path.join(save_fig_dir, str(self.t) + '_con.png')
            self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

            # visualize global footprint
            self.con_ax.clear()
            occ_map_render = np.copy(self.occupancy.footprint)
            occ_map_render = np.flip(occ_map_render, axis=0)
            self.con_ax.imshow(occ_map_render, cmap='gray')
            self.con_ax.axis('off')
            save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
            fp = os.path.join(save_fig_dir, str(self.t) + '_footprint.png')
            self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)


    def render(self, mode='human', close=False):
        """Renders the environment."""

        if self.renderer is None:
            self.renderer = Renderer(self.space, env_width=self.cfg.occ.map_width, env_height=self.cfg.occ.map_height, render_scale=self.cfg.render_scale, 
                    background_color=(28, 107, 160), caption="ASV Navigation", goal_line=self.cfg.goal_y)

        if self.cfg.render_snapshot:
            path = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + '.png')
            self.renderer.render(save=True, path=path)

        else:
            self.renderer.render(save=False)


    def close(self):
        plt.close('all')

        if self.renderer is not None:
            self.renderer.close()
