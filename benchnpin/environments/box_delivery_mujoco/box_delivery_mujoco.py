from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium import error, spaces
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union
import numpy as np
import os

# Bench-NPIN imports
from benchnpin.common.controller.position_controller import PositionController
from benchnpin.common.utils.utils import DotDict
# from benchnpin.environments.area_clearing.area_clearing import MOVE_STEP_SIZE, STEP_LIMIT, TURN_STEP_SIZE, WAYPOINT_MOVING_THRESHOLD, WAYPOINT_TURNING_THRESHOLD
from benchnpin.environments.box_delivery_mujoco.box_delivery_utils import generate_boxDelivery_xml, transport_box_from_recept, precompute_static_vertices, dynamic_vertices, receptacle_vertices, intersects_keepout
from benchnpin.common.utils.mujoco_utils import vw_to_wheels, make_controller, quat_z, inside_poly, quat_z_yaw, get_body_pose_2d, large_divider_corner_vertices
from benchnpin.common.utils.sim_utils import get_color

# SAM imports
from scipy.ndimage import distance_transform_edt, rotate as rotate_image
from cv2 import fillPoly
from skimage.draw import line
from skimage.measure import approximate_polygon
from skimage.morphology import disk, binary_dilation
import spfa

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

# Image segmentation indices
OBSTACLE_SEG_INDEX = 0
FLOOR_SEG_INDEX = 1
RECEPTACLE_SEG_INDEX = 3
NON_WHEELED_BOX_SEG_INDEX = 4
WHEELED_BOX_SEG_INDEX = 5
ROBOT_SEG_INDEX = 6
MAX_SEG_INDEX = 8

scale_factor = (2.845/10) # NOTE: this scales thresholds to be proportionately the same as in the 2d environment

MOVE_STEP_SIZE = 0.05 * scale_factor
TURN_STEP_SIZE = np.radians(15)

WAYPOINT_MOVING_THRESHOLD = 0.6 * scale_factor
WAYPOINT_TURNING_THRESHOLD = np.radians(10)
NOT_MOVING_THRESHOLD = 0.005 * scale_factor
NOT_TURNING_THRESHOLD = np.radians(0.05)
NONMOVEMENT_DIST_THRESHOLD = 0.05 * scale_factor
NONMOVEMENT_TURN_THRESHOLD = np.radians(0.05)
STEP_LIMIT = 1000


class BoxDeliveryMujoco(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        cfg=None,
        **kwargs,
    ):

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

        # Checking if the configuration is valid
        if self.cfg.wheels_on_boxes.wheels_on_boxes and self.cfg.wheels_on_boxes.num_boxes_with_wheels > self.cfg.boxes.num_boxes:
            raise ValueError("Number of boxes with wheels cannot be more than the total number of boxes.")
        

        # Setting up the environment parameters
        self.room_length = self.cfg.env.room_length
        env_size = self.cfg.env.obstacle_config.strip()
        if 'small' in env_size:
            self.room_width = self.cfg.env.room_width_small
        else:
            self.room_width = self.cfg.env.room_width_large

        # Receptacle position and size
        self.receptacle_position= self.cfg.env.receptacle_position
        self.receptacle_half= self.cfg.env.receptacle_half

        # Pillar position and size
        self.num_pillars = None
        self.pillar_half = None
        self.pillar_type = None
        self.adjust_num_pillars = False
        self.divider_thickness = self.cfg.large_divider.divider_thickness if env_size.strip() == "large_divider" else 0.0 
        if env_size.strip()=="small_columns":
            self.num_pillars = self.cfg.small_pillars.num_pillars
            self.pillar_half = self.cfg.small_pillars.pillar_half
            self.adjust_num_pillars= self.cfg.small_pillars.adjust_num_pillars
        elif env_size.strip()=="large_columns":
            self.num_pillars = self.cfg.large_pillars.num_pillars
            self.pillar_half = self.cfg.large_pillars.pillar_half
            self.adjust_num_pillars= self.cfg.large_pillars.adjust_num_pillars
        
        
        # environment
        self.local_map_pixel_width = self.cfg.env.local_map_pixel_width if self.cfg.train.job_type != 'sam' else self.cfg.env.local_map_pixel_width_sam
        self.local_map_width = max(self.room_length, self.room_width)
        self.local_map_pixels_per_meter = self.local_map_pixel_width / self.local_map_width
        self.wall_thickness = self.cfg.env.wall_thickness
        self.num_boxes = self.cfg.boxes.num_boxes
        self.num_completed_boxes_new = 0

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

        # Box with wheels
        self.wheels_on_boxes = self.cfg.wheels_on_boxes.wheels_on_boxes
        self.wheels_mass = self.cfg.wheels_on_boxes.wheels_mass
        self.wheels_support_mass = self.cfg.wheels_on_boxes.wheels_support_mass
        self.wheels_sliding_friction = self.cfg.wheels_on_boxes.wheels_sliding_friction
        self.wheels_torsional_friction = self.cfg.wheels_on_boxes.wheels_torsional_friction
        self.wheels_rolling_friction = self.cfg.wheels_on_boxes.wheels_rolling_friction
        self.wheels_support_damping_ratio = self.cfg.wheels_on_boxes.wheels_support_damping_ratio
        self.num_boxes_with_wheels = self.cfg.wheels_on_boxes.num_boxes_with_wheels if self.wheels_on_boxes else 0
        self.num_boxes_without_wheels = self.num_boxes - self.num_boxes_with_wheels
        self.names_boxes_without_wheels = [f"box{i}" for i in range(self.num_boxes_without_wheels)]

        # robot
        self.robot_hit_obstacle = False
        self.robot_info = self.cfg.agent
        self.robot_info['color'] = get_color('agent')
        self.robot_radius = ((self.robot_info.length**2 + self.robot_info.width**2)**0.5 / 2) * 1.2
        self.robot_dimen= (self.robot_info.length, self.robot_info.width)
        self.robot_half_width = max(self.robot_info.length, self.robot_info.width) / 2
        robot_pixel_width = int(2 * self.robot_radius * self.local_map_pixels_per_meter)
        self.robot_state_channel = np.zeros((self.local_map_pixel_width, self.local_map_pixel_width), dtype=np.float32)
        start = int(np.floor(self.local_map_pixel_width / 2 - robot_pixel_width / 2))
        for i in range(start, start + robot_pixel_width):
            for j in range(start, start + robot_pixel_width):
                # Circular robot mask
                if (((i + 0.5) - self.local_map_pixel_width / 2)**2 + ((j + 0.5) - self.local_map_pixel_width / 2)**2)**0.5 < robot_pixel_width / 2:
                    self.robot_state_channel[i, j] = 1
        self.agent_type = self.cfg.agent.agent_type

        if self.agent_type == "turtlebot_3":
            self.robot_name_in_xml = "base"
            self.joint_name_in_xml = "base_joint"
            self.placement_height = 0.01
            self.edges_space_left_for_robot = 0.1
            xml_file = os.path.join(self.current_dir, 'turtlebot3_burger_xml_file.xml')
        else:
            self.robot_name_in_xml = "jackal_base"
            self.joint_name_in_xml = "base_joint_jackal"
            self.placement_height = 0.02
            self.edges_space_left_for_robot = 0.5
            xml_file = os.path.join(self.current_dir, 'jackal_xml_file.xml')

        # generate random environmnt
        _, self.initialization_keepouts, self.clearance_poly = generate_boxDelivery_xml(N=self.cfg.boxes.num_boxes, env_type=self.cfg.env.obstacle_config, file_name=xml_file,
                        ROBOT_clear=self.cfg.agent.robot_clear, CLEAR=self.cfg.boxes.clearance, goal_half=self.receptacle_half, goal_center=self.receptacle_position, Z_BOX=self.cfg.boxes.box_half_size, ARENA_X=(0.0, self.room_length), 
                        ARENA_Y=(0.0, self.room_width), box_half_size=self.cfg.boxes.box_half_size, num_pillars=self.num_pillars, pillar_half=self.pillar_half, adjust_num_pillars=self.adjust_num_pillars, sim_timestep=self.cfg.env.sim_timestep,
                        divider_thickness=self.divider_thickness, bumper_type=self.cfg.agent.type_of_bumper, bumper_mass= self.cfg.agent.bumper_mass,
                        wheels_on_boxes=self.wheels_on_boxes, wheels_mass=self.wheels_mass, wheels_support_mass=self.wheels_support_mass, wheels_sliding_friction=self.wheels_sliding_friction,
                        wheels_torsional_friction=self.wheels_torsional_friction, wheels_rolling_friction=self.wheels_rolling_friction, wheels_support_damping_ratio=self.wheels_support_damping_ratio, box_mass=self.cfg.boxes.box_mass,
                        box_sliding_friction= self.cfg.boxes.box_sliding_friction, box_torsional_friction= self.cfg.boxes.box_torsional_friction, box_rolling_friction= self.cfg.boxes.box_rolling_friction, 
                        num_boxes_with_wheels=self.cfg.wheels_on_boxes.num_boxes_with_wheels, wheels_axle_damping_ratio=self.cfg.wheels_on_boxes.wheels_axle_damping_ratio, agent_type=self.cfg.agent.agent_type)

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            **kwargs,
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # OBSERVATION SPACES
        self.show_observation = False
        self.observation_shape = (self.local_map_pixel_width, self.local_map_pixel_width, self.num_channels)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        # Define action space
        if self.cfg.agent.action_type == 'velocity':
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'heading':
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'position':
            self.action_space = spaces.Box(low=0, high=self.local_map_pixel_width * self.local_map_pixel_width, dtype=np.float32)

        # get robot body & joint addresses
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot_name_in_xml)
        joint_adr = self.model.body_jntadr[self.base_body_id]
        self.qpos_index_base = self.model.jnt_qposadr[joint_adr]
        self.base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name_in_xml)

        # Box joint addresses
        joint_id_boxes=[]
        for i in range (self.num_boxes):
            joint_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"box{i}_joint")
            joint_id_boxes.append(joint_id)
        self.joint_id_boxes = joint_id_boxes

        # rewards
        if self.cfg.train.job_type == 'sam':
            rewards = self.cfg.rewards_sam
        else:
            rewards = self.cfg.rewards
        self.partial_rewards_scale = rewards.partial_rewards_scale
        self.goal_reward = rewards.goal_reward
        self.collision_penalty = rewards.collision_penalty
        self.non_movement_penalty = rewards.non_movement_penalty

        # misc
        self.ministep_size = self.cfg.misc.ministep_size * scale_factor
        self.inactivity_cutoff = self.cfg.misc.inactivity_cutoff if self.cfg.train.job_type != 'sam' else self.cfg.misc.inactivity_cutoff_sam
        self.random_seed = self.cfg.misc.random_seed
        # TODO implement random_state
        self.random_state = np.random.RandomState(self.random_seed)

        self.episode_idx = None
        self.path = None
        self.tries_before_inactive = self.cfg.train.tries_before_inactive

        # pillar clearance
        self.pillar_clearance=0.3

        # placing inactive pillars
        # height of the plane to be placed at
        self.STANDBY_Z   = 9.7
        # extra clearance between parked pillars
        self.STANDBY_GAP = 0.20
        # extra clearance between parked pillars and the wall
        self.X_clearance = 0.3
        
        if self.cfg.render.show_obs or self.cfg.render.show:
            # show state
            num_plots = self.num_channels
            self.state_plot = plt
            self.state_fig, self.state_ax = self.state_plot.subplots(1, num_plots, figsize=(4 * num_plots, 6))
            self.colorbars = [None] * num_plots
            if self.cfg.render.show_obs:
                self.state_plot.ion()  # Interactive mode on      


    def save_local_map(self, obs_uint8, out_path) -> None:
        """
        Saves the local map observation as a 2x2 figure with each channel in a separate subplot.
        LATER REMOVING THIS FUCNTION AND USING RENDERING INSTEAD"""
    
        if obs_uint8.ndim != 3 or obs_uint8.shape[2] != 4:
            raise ValueError("Expected observation of shape (W, W, 4)")

        # 2×2 figure
        fig, ax = plt.subplots(2, 2, figsize=(4, 4))
        titles = ["Static", "Movable", "Goal DT", " Ergocenttric DT"]

        for k in range(4):
            i, j   = divmod(k, 2)
            ax_ij  = ax[i, j]
            ax_ij.imshow(obs_uint8[:, :, k], cmap="hot", vmin=0, vmax=255)
            ax_ij.set_xticks([]); ax_ij.set_yticks([])
            ax_ij.set_title(titles[k], fontsize=6, pad=2)

        plt.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def render_env(self, mode='human', close=False):
        """Renders the environment."""

        if self.cfg.render.show and self.render_mode == "human":
            self.render()

        if self.cfg.render.show_obs and self.show_observation and self.observation is not None:# and self.t % self.cfg.render.frequency == 1:
            self.show_observation = False
            channel_name = ["Occupancy", "Footprint", "Goal DT", "Egocentric DT"]
            for ax, i in zip(self.state_ax, range(self.num_channels)):
                ax.clear()
                ax.set_title(channel_name[i])
                ax.set_xticks([])
                ax.set_yticks([])
                im = ax.imshow(self.observation[:,:,i], cmap='hot', interpolation='nearest')
            
            self.state_plot.draw()
            self.state_plot.pause(0.1)


    def update_path(self, waypoints):
        for i, point in enumerate(waypoints):
            self.model.site(f"wp{i}").pos[:2] = point[:2]           # set visualization point (x, y)
            self.model.site(f"wp{i}").pos[2] = 0.3                  # set visualization point z-axis

        # Move the rest out of view
        for i in range(len(waypoints), 50):
            site_id = self.model.site(f"wp{i}").id
            self.data.site_xpos[site_id] = np.array([1000, 1000, 1000])

    def step(self, action):

        self.robot_hit_obstacle = False
        robot_boxes = 0
        robot_reward = 0

        robot_initial_position = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        robot_initial_heading = quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7])

        # TODO check if move_sign is necessary
        self.path, robot_move_sign = self.position_controller.get_waypoints_to_spatial_action(robot_initial_position, robot_initial_heading, action)

        # if self.cfg.render.show:
        #     self.renderer.update_path(self.path)
        self.update_path(self.path)

        robot_distance, robot_turn_angle = self.execute_robot_path(robot_initial_position, robot_initial_heading, robot_move_sign)
        
        self._step_dx = robot_distance
        self._step_dyaw = robot_turn_angle

        # check if episode is done
        terminated = False
        if len(self.joint_id_boxes) == 0:
            terminated = True
        
        self.inactivity_counter += 1
        truncated = False
        if self.inactivity_counter >= self.inactivity_cutoff:
            # print('inactive')
            terminated = True
            truncated = True

        # items to return
        self.observation = self.generate_observation(done=terminated)
        reward = self._get_rew()
        self.robot_cumulative_distance += robot_distance
        self.robot_cumulative_boxes += self.num_completed_boxes_new
        self.robot_cumulative_reward += reward
        ministeps = robot_distance / self.ministep_size
        info = {
            'cumulative_distance': self.robot_cumulative_distance,
            'cumulative_boxes': self.robot_cumulative_boxes,
            'cumulative_reward': self.robot_cumulative_reward,
            'ministeps': ministeps,
        }

        # render environment
        if self.cfg.render.show:
            self.show_observation = True
            self.render_env()

        return self.observation, reward, terminated, truncated, info
    
    def execute_robot_path(self, robot_initial_position, robot_initial_heading, robot_move_sign):
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
            if np.abs(heading_diff) > TURN_STEP_SIZE / 2 and np.abs(heading_diff - prev_heading_diff) > 0.001:
                pass
            else:
                done_turning = True
                if self.distance(robot_position, robot_waypoint_position) < MOVE_STEP_SIZE:
                    robot_new_position = robot_waypoint_position

            # change robot pose (use controller)
            v, w, _ = make_controller(robot_prev_position, robot_prev_heading, robot_waypoint_position)
            if not done_turning:
                v = 0
            v_l, v_r = vw_to_wheels(v, w)

            # apply the control 'frame_skip' steps
            self.do_simulation([v_l, v_r], self.frame_skip)

            # get new robot pose
            robot_position = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
            robot_heading = quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7])
            prev_heading_diff = heading_diff
            
            # stop moving if robot collided with obstacle
            self.robot_hit_obstacle = self.robot_hits_static()
            if self.distance(robot_prev_waypoint_position, robot_position) > MOVE_STEP_SIZE:
            # if self.distance(robot_prev_position, robot_position) < MOVE_STEP_SIZE / 100:
                # if self.robot_hit_obstacle or done_turning:
                if self.robot_hit_obstacle:
                    robot_is_moving = False
                    break   # Note: self.robot_distance does not get updated

            # stop if robot reached waypoint
            if (self.distance(robot_position, robot_waypoint_positions[robot_waypoint_index]) < WAYPOINT_MOVING_THRESHOLD/2
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
                    self.path = self.path[1:]

            sim_steps += 1
            if sim_steps % 10 == 0 and self.cfg.render.show:
                self.render_env()

            # break if robot is stuck
            # if done_turning:
            #     print(f'distance: {self.distance(robot_prev_position, robot_position)}, notmoving threshold: {NOT_MOVING_THRESHOLD*0.25}')
            if sim_steps > STEP_LIMIT:# or (self.distance(robot_prev_position, robot_position) < NOT_MOVING_THRESHOLD * 0.25 and done_turning):
                print(f"Robot is stuck after {sim_steps} steps.")
                break

        print(f"Simulated {sim_steps} steps to execute action.")
        robot_angle = quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7])
        robot_heading = self.restrict_heading_range(robot_angle)
        robot_turn_angle = self.heading_difference(robot_initial_heading, robot_heading)
        return robot_distance, robot_turn_angle

    # Observation generation functions

    def generate_observation(self, done=False):
        """ Generates the observation for the environment."""

        if done:
            return None
        
        # Getting the robot and boxes vertices
        robot_properties, self.wheeled_boxes_vertices, self.non_wheeled_boxes_vertices =dynamic_vertices(self.model,self.data, self.qpos_index_base,self.joint_id_boxes, self.robot_dimen, self.cfg.boxes.box_half_size, self.names_boxes_without_wheels)

        # Update the global overhead map with the current robot and boundaries
        self.update_global_overhead_map(robot_properties[1])

        robot_postition = robot_properties[3]
        robot_angle = robot_properties[2]

        # Create the robot state channel
        channels = []
        
        channels.append(self.get_local_map(self.global_overhead_map, robot_postition, robot_angle))
        channels.append(self.robot_state_channel)
        channels.append(self.get_local_distance_map(self.create_global_shortest_path_to_receptacle_map(self.receptacle_position), robot_postition, robot_angle))
        channels.append(self.get_local_distance_map(self.create_global_shortest_path_map(robot_postition), robot_postition, robot_angle))
        observation = np.stack(channels, axis=2)
        observation = (observation * 255).astype(np.uint8)
        return observation
    
    def get_local_map(self, global_map, robot_position, robot_heading):
        """Takes a global map and returns a local map centered around the robot's position."""
        
        # sqrt 2 to provide some clearance around the image when the frame is rotated
        W        = self.local_map_pixel_width
        crop_W   = self.round_up_to_even(W * np.sqrt(2))
        rot_deg  = 90 - np.degrees(robot_heading)

        # robot (x,y) to pixel indices in the global map
        ci = int(np.floor(-robot_position[1] * self.local_map_pixels_per_meter + global_map.shape[0] / 2))
        cj = int(np.floor( robot_position[0] * self.local_map_pixels_per_meter + global_map.shape[1] / 2))

        # clip to valid pixel indices
        half = crop_W // 2
        i0, i1 = ci - half, ci + half
        j0, j1 = cj - half, cj + half
        
        # how many pixels to pad
        pad_top    = max(0, -i0)
        pad_left   = max(0, -j0)
        pad_bottom = max(0,  i1 - global_map.shape[0])
        pad_right  = max(0,  j1 - global_map.shape[1])

        # pad the global map if needed
        if pad_top or pad_left or pad_bottom or pad_right:
            pad_val = FLOOR_SEG_INDEX / MAX_SEG_INDEX
            global_map = np.pad(global_map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=pad_val)
            # shift window after padding
            i0 += pad_top;  i1 += pad_top
            j0 += pad_left; j1 += pad_left

        # crop the global map and rotate it
        crop = global_map[i0:i1, j0:j1]
        rot  = rotate_image(crop, rot_deg, order=0)
        mid  = rot.shape[0] // 2
        local_map = rot[mid - W // 2 : mid + W // 2,
                        mid - W // 2 : mid + W // 2]
        return local_map
    
    def get_local_distance_map(self, global_map, robot_position, robot_heading):
        """Takes a global distance map and returns a local distance map centered around the robot's position."""

        local_map = self.get_local_map(global_map, robot_position, robot_heading)
        local_map -= local_map.min() 
        return local_map

    def create_padded_room_zeros(self):
        """ Creates a padded room of zeros with the size of the local map."""

        return np.zeros((
            int(2 * np.ceil((self.room_width * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2)),
            int(2 * np.ceil((self.room_length * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2))
        ), dtype=np.float32)
    
    def create_global_shortest_path_to_receptacle_map(self,receptacle_position):
        """ Creates a global shortest path map to the receptacle."""

        # Create a padded room of zeros and compute the shortest path to the receptacle
        global_map = self.create_padded_room_zeros() + np.inf
        rx, ry = receptacle_position[:2]
        pixel_i, pixel_j = self.position_to_pixel_indices(rx, ry, self.configuration_space.shape)
        pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
        shortest_path_image, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))

        # Scale the shortest path image to the local map pixel width
        shortest_path_image /= self.local_map_pixels_per_meter
        global_map = np.minimum(global_map, shortest_path_image)

        # Scale the global map to the local map pixel width and invert if needed
        global_map /= (np.sqrt(2) * self.local_map_pixel_width) / self.local_map_pixels_per_meter
        global_map *= self.cfg.env.shortest_path_channel_scale

        # Invert the global map if needed
        if self.cfg.env.invert_receptacle_map:
            global_map += 1-self.configuration_space
            global_map[global_map==(1-self.configuration_space)] = 1

        return global_map
    
    def create_global_shortest_path_map(self, robot_position):
        """ Creates a global shortest path map from the robot's position."""

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

        # Create a padded room of zeros to draw the walls and columns
        obstacle_map = self.create_padded_room_zeros()
        small_obstacle_map = np.zeros((self.local_map_pixel_width+20, self.local_map_pixel_width+20), dtype=np.float32)
        
        # Precompute static vertices for walls and columns
        wall_vertices, columns_from_keepout, corners = precompute_static_vertices(self.initialization_keepouts, self.wall_thickness, self.room_length, self.room_width)

        # Iterating through each wall vertice and keepout columns
        for wall_vertices_each_wall in wall_vertices+columns_from_keepout+corners:

            # get world coordinates of vertices
            vertices_np = np.array([[v[0], v[1]] for v in wall_vertices_each_wall[1]])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * self.local_map_pixels_per_meter).astype(np.int32)
            vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] = small_obstacle_map.shape[0] - vertices_px[:, 1]

            # draw the boundary on the small_obstacle_map
            fillPoly(small_obstacle_map, [vertices_px], color=1)
        
        start_i, start_j = int(obstacle_map.shape[0] / 2 - small_obstacle_map.shape[0] / 2), int(obstacle_map.shape[1] / 2 - small_obstacle_map.shape[1] / 2)
        obstacle_map[start_i:start_i + small_obstacle_map.shape[0], start_j:start_j + small_obstacle_map.shape[1]] = small_obstacle_map

        # Dilate obstacles and walls based on robot size
        selem = disk(np.floor(self.robot_radius * self.local_map_pixels_per_meter))
        self.configuration_space = 1 - binary_dilation(obstacle_map, selem).astype(np.float32)
        
        # Dilate obstacles and walls based on robot size just for the thin configuration space
        selem_thin = disk(np.floor(self.robot_half_width * self.local_map_pixels_per_meter))
        self.configuration_space_thin = 1 - binary_dilation(obstacle_map, selem_thin).astype(np.float32)

        # closest_cspace_indices is used to find the closest valid configuration space indices for a given position
        self.closest_cspace_indices = distance_transform_edt(1 - self.configuration_space, return_distances=False, return_indices=True)
        # Converting the value as in 1 for obstacle and 0 for free space
        self.small_obstacle_map = 1 - small_obstacle_map

    def update_global_overhead_map(self, robot_vertices):
        """Updates the global overhead map with the current robot and boundaries."""

        # Makes a copy of immovable obstacles
        small_overhead_map = self.small_obstacle_map.copy()

        # draw the boundary on the small_overhead_map
        small_overhead_map[small_overhead_map == 1] = FLOOR_SEG_INDEX / MAX_SEG_INDEX

        def draw_object(vertices, seg_index):

            # get world coordinates of vertices
            vertices_np = np.array([[v[0], v[1]] for v in vertices])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * self.local_map_pixels_per_meter).astype(np.int32)
            vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] = small_overhead_map.shape[0] - vertices_px[:, 1]

            fillPoly(small_overhead_map, [vertices_px], color=seg_index/MAX_SEG_INDEX)
        
        # Draw the receptacle
        receptacle_vertice=receptacle_vertices(self.receptacle_position, self.receptacle_half)
        draw_object(receptacle_vertice, RECEPTACLE_SEG_INDEX)

        # Draw the robot
        draw_object(robot_vertices, ROBOT_SEG_INDEX)
        
        # Draw the wheeled and non-wheeled boxes
        for box_vertices in self.wheeled_boxes_vertices:
            draw_object(box_vertices[0], WHEELED_BOX_SEG_INDEX)

        for box_vertices in self.non_wheeled_boxes_vertices:
            draw_object(box_vertices[0], NON_WHEELED_BOX_SEG_INDEX)

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
        """ Calculates the distance of the shortest path between two positions in the configuration space."""

        path = self.shortest_path(source_position, target_position, configuration_space=configuration_space)

        return sum(self.distance(path[i - 1], path[i]) for i in range(1, len(path)))
    
    def closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]
    
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
        """ Converts a position (x, y) in meters to pixel indices in the local map."""

        pixel_i = np.floor(image_shape[0] / 2 - y * self.local_map_pixels_per_meter).astype(np.int32)
        pixel_j = np.floor(image_shape[1] / 2 + x * self.local_map_pixels_per_meter).astype(np.int32)
        pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
        pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
        return pixel_i, pixel_j

    def pixel_indices_to_position(self, pixel_i, pixel_j, image_shape):
        """ Converts pixel indices (i, j) in the local map to a position (x, y) in meters."""

        position_x = (pixel_j - image_shape[1] / 2) / self.local_map_pixels_per_meter
        position_y = (image_shape[0] / 2 - pixel_i) / self.local_map_pixels_per_meter

        return position_x, position_y

    # Reward functions

    def _get_rew(self):

        robot_reward = 0

        # partial reward for moving boxes towards receptacle
        self.motion_dict, boxes_total_distance= self.update_motion_dict(self.motion_dict)

        if boxes_total_distance<-0.01 or 0.01<boxes_total_distance:
            robot_reward += self.partial_rewards_scale * boxes_total_distance / scale_factor

        # reward for boxes in receptacle
        self.joint_id_boxes, self.num_completed_boxes_new = transport_box_from_recept(self.model, self.data, self.joint_id_boxes, self.room_length,
                                                                         self.room_width, goal_half=self.receptacle_half, goal_center=self.receptacle_position, box_half_size=self.cfg.boxes.box_half_size)
        if self.num_completed_boxes_new > 0:
            print(f'Delivered {self.num_completed_boxes_new} boxes!')
            print(f'reward: {self.goal_reward * self.num_completed_boxes_new}')
            self.inactivity_counter = 0
        robot_reward += self.goal_reward * self.num_completed_boxes_new
        
        # penalty for hitting obstacles
        if self.robot_hit_obstacle:
            print('Robot hit an obstacle!')
            robot_reward -= self.collision_penalty
        
        # penalty for small movements
        if self._step_dx < NONMOVEMENT_DIST_THRESHOLD and self._step_dyaw < NONMOVEMENT_TURN_THRESHOLD:
            print('Robot not moving enough!')
            robot_reward -= self.non_movement_penalty
        
        # self.robot_cumulative_reward += robot_reward
        return robot_reward

    def _sample_pillar_centres(self):
        """
        Sample non-overlapping (cx, cy) pairs for every pillar and the matching keep-outs.
        """

        def boxes_overlap(cx, cy, px, py):
            """Axis-aligned overlap test with clearance."""
            return (
                abs(cx - px) < (2 * hx + self.pillar_clearance) and
                abs(cy - py) < (2 * hy + self.pillar_clearance)
            )

        # “small_empty” scene
        if self.num_pillars is None or self.pillar_half is None:
            return [], [], []

        rng = np.random.default_rng()

        if self.adjust_num_pillars: 

            # decide how many pillars are in play this episode
            n_active = rng.integers(1, self.num_pillars + 1)
        else:
            n_active = self.num_pillars

        active_centres = []
        corners_active = []
        tries = 0
        while len(active_centres) < n_active and tries < 500000:
            tries += 1
            cx = rng.uniform(-self.room_length/2+0.3, self.room_length/2 - 0.30)
            cy = rng.uniform(-self.room_width/2+0.3, self.room_width/2-0.3)

            # pillar corners for clearance checks
            hx, hy, _ = self.pillar_half
            corners = [(cx-hx, cy-hy), (cx+hx, cy-hy),
                    (cx+hx, cy+hy), (cx-hx, cy+hy)]

            if not all(inside_poly(x, y, self.clearance_poly) for x, y in corners):
                continue
            
            if any(np.hypot(cx-px, cy-py) < 2*hx + 0.30 for px, py in active_centres):
                continue

            active_centres.append((cx, cy))
            corners_active.append(corners)

        HX, HY, _      = self.pillar_half
        STANDBY_Y      = self.room_width / 2 + HY + 0.1
        STANDBY_STEP   = 2*self.pillar_half[2] + self.STANDBY_GAP

        # parked (inactive) pillars – laid out in a neat row on the stand-by plane
        parked_centres = [
            (-self.room_length/2+self.X_clearance+ k * STANDBY_STEP, STANDBY_Y)
            for k in range(self.num_pillars - n_active)
        ]

        return active_centres, parked_centres, corners_active

    def init_motion_dict(self):
        """
        Returns {body_id: [length_travelled, last_position(x,y), mass]}
        """

        track = {}

        # robot base
        m_robot = self.model.body_mass[self.base_body_id]
        cx, cy = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        track[self.base_body_id] = [0.0, np.array([cx, cy], dtype=float), float(m_robot)]

        # boxes
        for jid in self.joint_id_boxes:
            bid = self.model.jnt_bodyid[jid]
            adr = self.model.jnt_qposadr[jid]
            bx, by = self.data.qpos[adr:adr+2]
            track[bid] = [0.0, np.array([bx, by], dtype=float),
                        float(self.model.body_mass[bid])]

        return track

    def update_motion_dict(self, motion_dict):
        """
        In-place update of motion_dict.
        Skips boxes that have already been teleported away (their body_id is gone).
        """
        goal_xy = np.array([
        self.receptacle_position[0] + 0.5 * self.room_length,
        self.receptacle_position[1] + 0.5 * self.room_width])

        # robot
        length, last_xy, mass = motion_dict[self.base_body_id]
        cx, cy = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        dist = np.linalg.norm(np.array([cx, cy]) - last_xy)
        motion_dict[self.base_body_id][0] += dist
        motion_dict[self.base_body_id][1][:] = (cx, cy)

        # boxes still present
        live_bodies = {self.model.jnt_bodyid[jid] for jid in self.joint_id_boxes}
        boxes_total_distance=0

        for body_id in live_bodies:

            length, last_xy, mass = motion_dict[body_id]

            adr  = self.model.body_jntadr[body_id]
            jpos = self.model.jnt_qposadr[adr]
            curr_xy = self.data.qpos[jpos : jpos+2]

            # progress *before* we mutate last_xy
            # prev_d = np.linalg.norm(last_xy - goal_xy)
            # curr_d = np.linalg.norm(curr_xy  - goal_xy)
            prev_d = self.shortest_path_distance(last_xy, goal_xy)
            curr_d = self.shortest_path_distance(curr_xy, goal_xy)
            boxes_total_distance += prev_d - curr_d

            # book-keeping for next step
            # motion_dict[body_id][0] += np.linalg.norm(curr_xy - last_xy)
            motion_dict[body_id][0] += self.shortest_path_distance(curr_xy, last_xy)
            motion_dict[body_id][1][:] = curr_xy

        return motion_dict, boxes_total_distance

    # contacts
    def robot_hits_static(self) -> bool:

        ROBOT_PREFIX    = ("base","jackal")
        STATIC_PREFIXES = ("wall", "small_col", "large_col", "large_divider", "corner")
        SKIP_GEOMS = {"floor", "pillars_kept"}

        for k in range(self.data.ncon):
            c   = self.data.contact[k]
            g1, g2 = c.geom1, c.geom2

            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)

            # skip any unnamed geoms or if no names
            if not n1 or not n2 or n1 in SKIP_GEOMS or n2 in SKIP_GEOMS:
                continue

            # check robot vs any static prefix
            hit1 = any(pref in n1.lower() for pref in ROBOT_PREFIX) and any(pref in n2.lower() for pref in STATIC_PREFIXES)
            hit2 = any(pref in n2.lower() for pref in ROBOT_PREFIX) and any(pref in n1.lower() for pref in STATIC_PREFIXES)

            if hit1 or hit2:
                return True
        return False

    def reset_model(self):
        """
        Randomly sample non-overlapping (x, y, theta) for robot and boxes.
        Teleport them in simulation using sim.data.qpos.
        """
        positions = []
        cfg_obs = self.cfg.env.obstacle_config.strip()

        if cfg_obs in ("small_columns", "large_columns"):

            # pillars
            active, parked, self.initialization_keepouts = self._sample_pillar_centres()
            
            # float-safe membership test
            def close(pt1, pt2, tol=1e-6):
                return abs(pt1[0] - pt2[0]) <= tol and abs(pt1[1] - pt2[1]) <= tol

            def in_list(pt, lst):
                return any(close(pt, other) for other in lst)

            # place every pillar (active + parked)
            all_centres = active + parked

            for k, (cx, cy) in enumerate(all_centres):
                prefix = "small_col" if "small" in self.cfg.env.obstacle_config else "large_col"
                name   = f"{prefix}{k}"

                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
                if j_id == -1:
                    print(f"Joint {name}_joint not found in the model.")

                adr = self.model.jnt_qposadr[j_id]
                if in_list((cx, cy), active):
                    #active pillar
                    self.data.qpos[adr:adr+2] = [cx, cy]

                else:
                    # parked pillar
                    self.data.qpos[adr:adr+3] = [cx, cy, -self.STANDBY_Z]
        
                self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
                self.data.qvel[adr:adr+6]   = 0

        elif cfg_obs == "large_divider":
            
            hx = 0.8 * self.room_length / 2
            hy = self.divider_thickness / 2
            hz = 0.1

            # Divider X-centre and Y-centre
            cx = 0.2 * self.room_length / 2
            # Left some clearance from corners
            cy = 0.0

            # Teleporting the divider body (joint) to the new pose

            # FIXME: What happened to the joint_name variable?
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,"large_divider_joint")

            if j_id < 0:
                raise RuntimeError(f"Joint {joint_name} not found in model")

            adr = self.model.jnt_qposadr[j_id]
            self.data.qpos[adr     : adr+3] = [cx, cy, hz]
            self.data.qpos[adr+3   : adr+7] = [1, 0, 0, 0]
            self.data.qvel[adr     : adr+6] = 0

            corner_4_coordinates, corner_5_coordinates = large_divider_corner_vertices(cx, cy, hy, self.room_length/2)

            divider_poly = [
                (cx - hx, cy - hy), (cx + hx, cy - hy),
                (cx + hx, cy + hy), (cx - hx, cy + hy)
            ]

            # This is to ensure that that the robot/boxes do not come too close to the divider during initialization
            self.initialization_keepouts = [divider_poly, corner_4_coordinates, corner_5_coordinates]

        def is_valid(pos, radius, positions):

            # unpack immediately
            x, y, _ = pos

            # check against previously placed robot/boxes
            for (px, py, _), pr in positions:
                dist = np.hypot(x - px, y - py)
                if dist < (pr + radius + self.cfg.boxes.clearance):
                    return False

            # check against pillar keep-outs
            if intersects_keepout(x, y, self.initialization_keepouts):
                return False

            # finally check your overall clearance polygon
            if not inside_poly(x,y , self.clearance_poly):
                return False

            return True

        def resolve_robot_box_overlaps(proposed_x_y, shift=0.6):

            # geometry / margins
            box_half = float(self.cfg.boxes.box_half_size)
            r_box = float(np.hypot(box_half, box_half))

            # extra margin to keep from robot
            need_robot_box = self.cfg.agent.robot_clear

            # box–box spacing needed
            box_margin = 0.0
            need_box_box = 2 * box_half

            # bounds
            x_min = -self.room_length / 2 + 0.1
            x_max =  self.room_length / 2 - 0.1
            y_min = -self.room_width  / 2 + 0.1
            y_max =  self.room_width  / 2 - 0.1

            def in_bounds(x, y):
                return (x_min <= x <= x_max) and (y_min <= y <= y_max)

            # current robot center (x,y)
            rx, ry = proposed_x_y

            # collecting active box centers
            active_boxes = []  # list of (idx, jid, x, y)
            for i, jid in enumerate(getattr(self, "joint_id_boxes", [])):
        
                adr = int(self.model.jnt_qposadr[jid])
                bx, by, bz = self.data.qpos[adr:adr+3]
                active_boxes.append([i, jid, float(bx), float(by)])

            # helper: validity of a proposed (x,y) for box i
            def box_pose_ok(i, x, y, centers_xy):
                
                # keepouts / polygons
                if intersects_keepout(x, y, self.initialization_keepouts):
                    return False
                if not inside_poly(x, y, self.clearance_poly):
                    return False
                # robot clearance
                if np.hypot(x - rx, y - ry) < need_robot_box:
                    return False
                # other boxes
                for (j_idx, _jid, ox, oy) in centers_xy:
                    if j_idx == i:
                        continue
                    if np.hypot(x - ox, y - oy) < need_box_box:
                        return False
                return True

            centers = [(i, jid, bx, by) for (i, jid, bx, by) in active_boxes]

            # passing over boxes; if overlapping, trying to ±shift in X
            for k, (i, jid, bx, by) in enumerate(list(centers)):

                if np.hypot(bx - rx, by - ry) >= need_robot_box:
                    continue

                # candidate moves: +shift (right), -shift (left)
                candidates = [(bx + shift, by), (bx - shift, by)]

                # keeping those inside plane and valid wrt robot/boxes/keepouts
                good = []
                for (cx, cy) in candidates:
                    if in_bounds(cx, cy) and box_pose_ok(i, cx, cy, centers):

                        # score by distance to robot; prefer the farther one
                        score = np.hypot(cx - rx, cy - ry)
                        good.append((score, cx, cy))

                if good:
    
                    _, nx, ny = max(good, key=lambda t: t[0])

                    adr = int(self.model.jnt_qposadr[jid])
                    self.data.qpos[adr:adr+2] = [nx, ny]           # keep z, quat as-is

                    # updating local centers list so subsequent checks see the move
                    centers[k] = (i, jid, nx, ny)
                    return True
                else:
                    return False

        
        # Define bounds of the placement area (slightly inside the walls)
        y_min_robot, y_max_robot = -self.room_width / 2 + self.edges_space_left_for_robot, self.room_width / 2 - self.edges_space_left_for_robot
        x_min_robot, x_max_robot = -self.room_length / 2 + self.edges_space_left_for_robot, self.room_length / 2 - self.edges_space_left_for_robot

        # Sample robot pose
        while True:
            x = np.random.uniform(x_min_robot, x_max_robot)
            y = np.random.uniform(y_min_robot, y_max_robot)
            theta = np.random.uniform(-np.pi, np.pi)
            if is_valid((x, y, theta), self.cfg.agent.robot_clear, positions):
                proposed_x_y = (x, y)
                if resolve_robot_box_overlaps(proposed_x_y, 1.0):
                    positions.append(((x, y, theta), self.cfg.agent.robot_clear))
                    break
                else:
                    continue


        # Set robot pose
        base_qpos_addr = self.model.jnt_qposadr[self.base_joint_id]
        self.data.qpos[base_qpos_addr:base_qpos_addr+3] = [x, y, self.placement_height]  # x, y, z
        if self.agent_type == "turtlebot_3":
            self.data.qpos[base_qpos_addr+3:base_qpos_addr+7] = quat_z(theta)

        self.data.qvel[base_qpos_addr:base_qpos_addr+6] = 0

        # Box joint addresses
        joint_id_boxes=[]
        for i in range (self.num_boxes):
            joint_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"box{i}_joint")
            joint_id_boxes.append(joint_id)
        self.joint_id_boxes = joint_id_boxes

        # Assume box is square with radius from center to corner (diagonal/2)
        box_half_size = self.cfg.boxes.box_half_size
        box_r = np.sqrt(box_half_size ** 2 + box_half_size ** 2)
        
        if self.wheels_on_boxes:
            z = box_half_size+0.03
        else:
            z = box_half_size+0.005
        
        # Define bounds of the placement area (slightly inside the walls)
        y_min, y_max = -self.room_width / 2 + 0.1, self.room_width / 2 - 0.1
        x_min, x_max = -self.room_length / 2 + 0.1, self.room_length / 2 - 0.1

        for i in range(self.num_boxes):
            while True:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                theta = np.random.uniform(-np.pi, np.pi)
                if is_valid((x, y, theta), box_r, positions):
                    positions.append(((x, y, theta), box_r))
                    break
            
            qadr = self.model.jnt_qposadr[self.joint_id_boxes[i]]
            self.data.qpos[qadr:qadr+3] = np.array([x, y, z])
            self.data.qpos[qadr+3:qadr+7] = quat_z(theta)
            self.data.qvel[qadr:qadr+6] = 0

        self._prev_robot_xy = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        self._prev_robot_heading = quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7])

        # get the robot and boxes vertices
        robot_properties, self.wheeled_boxes_vertices, self.non_wheeled_boxes_vertices =dynamic_vertices(self.model,self.data, self.qpos_index_base,self.joint_id_boxes, self.robot_dimen, self.cfg.boxes.box_half_size, self.names_boxes_without_wheels)

        self.motion_dict = self.init_motion_dict()

        # reset stats
        self.inactivity_counter = 0
        self.robot_cumulative_distance = 0
        self.robot_cumulative_boxes = 0
        self.robot_cumulative_reward = 0

        self.update_configuration_space()

        # reset map
        self.global_overhead_map = self.create_padded_room_zeros()
        self.update_global_overhead_map(robot_properties[1])

        observation = self.generate_observation()

        self.position_controller = PositionController(self.cfg, self.robot_radius, self.room_width, self.room_length, 
                                                      self.configuration_space, self.configuration_space_thin, self.closest_cspace_indices, 
                                                      self.local_map_pixel_width, self.local_map_width, self.local_map_pixels_per_meter,
                                                      TURN_STEP_SIZE, MOVE_STEP_SIZE, WAYPOINT_MOVING_THRESHOLD, WAYPOINT_TURNING_THRESHOLD)

        return observation

    def _get_reset_info(self):
        info = {
            'cumulative_distance': self.robot_cumulative_distance,
            'cumulative_boxes': self.robot_cumulative_boxes,
            'cumulative_reward': self.robot_cumulative_reward,
            'ministeps': 0,
        }
        return info
