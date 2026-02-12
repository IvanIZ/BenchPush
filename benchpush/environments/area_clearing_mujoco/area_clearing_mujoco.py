import time
from PIL import Image
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium import error, spaces
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union
import numpy as np
from numpy.typing import NDArray
import os
from shapely.geometry import Polygon, LineString, Point

# Bench-NPIN imports
from benchpush.common.controller.position_controller import PositionController
from benchpush.common.evaluation.metrics import total_work_done, obs_to_goal_difference
from benchpush.common.utils.utils import DotDict
from benchpush.environments.area_clearing_mujoco.area_clearing_utils import generate_area_clearing_xml, precompute_static_vertices, dynamic_vertices, intersects_keepout, receptacle_vertices, transport_box_from_recept
from benchpush.common.utils.mujoco_utils import vw_to_wheels, make_controller, quat_z, inside_poly, quat_z_yaw, get_body_pose_2d
from benchpush.common.utils.sim_utils import get_color


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
    "distance": 5.0,
}

DEFAULT_SIZE = 480

DISTANCE_SCALE_MAX = 0.5

#Image segmentation indices
OBSTACLE_SEG_INDEX = 0
FLOOR_SEG_INDEX = 1
RECEPTACLE_SEG_INDEX = 2
WHEELED_BOX_SEG_INDEX = 3
NON_WHEELED_BOX_SEG_INDEX = 4
ROBOT_SEG_INDEX = 5
COMPLETED_NON_WHEELED_BOX_SEG_INDEX = 7
COMPLETED_WHEELED_BOX_SEG_INDEX = 8
MAX_SEG_INDEX = 8

scale_factor = (2.845/10) # scales thresholds to be proportionately the same as in the 2d environment

MOVE_STEP_SIZE = 0.05 * scale_factor
TURN_STEP_SIZE = np.radians(15)

WAYPOINT_MOVING_THRESHOLD = 0.2 * scale_factor
WAYPOINT_TURNING_THRESHOLD = np.radians(10)
NOT_MOVING_THRESHOLD = 0.005 * scale_factor
NOT_TURNING_THRESHOLD = np.radians(0.05)
NONMOVEMENT_DIST_THRESHOLD = 0.05 * scale_factor
NONMOVEMENT_TURN_THRESHOLD = np.radians(0.05)
STEP_LIMIT = 400


class AreaClearingMujoco(MujocoEnv, utils.EzPickle):

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
        render_width=DEFAULT_SIZE,
        render_height=DEFAULT_SIZE,        
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

        self.room_width = self.cfg.env.room_width
        self.room_length = self.cfg.env.room_length

        # Setting up the environment parameters
        self.room_length_inner = self.room_length - 2 * self.cfg.env.distance_between_inner_goal_and_outer_wall_length
        self.room_width_inner = self.room_width - 2 * self.cfg.env.distance_between_inner_goal_and_outer_wall_width

        # Pillar position and size
        self.num_pillars = self.cfg.small_pillars.num_pillars
        self.pillar_half = self.cfg.small_pillars.pillar_half

        # environment
        self.local_map_pixel_width = self.cfg.env.local_map_pixel_width if self.cfg.train.job_type != 'sam' else self.cfg.env.local_map_pixel_width_sam
        self.local_map_width = max(self.room_length, self.room_width)
        # self.local_map_width = max(self.room_length_inner + 2 * self.cfg.env.distance_between_inner_goal_and_outer_wall_length, self.room_width_inner + 2 * self.cfg.env.distance_between_inner_goal_and_outer_wall_width)
        self.local_map_pixels_per_meter = self.local_map_pixel_width / self.local_map_width
        self.wall_thickness = self.cfg.env.wall_thickness
        self.num_boxes = self.cfg.boxes.num_boxes
        self.internal_clearance_length = self.cfg.env.internal_clearance_length
        self.receptacle_position = [0, 0]  # The center of the plane is assumed to be 0,0
        self.receptacle_half = [self.room_length_inner / 2, self.room_width_inner / 2]
        self.num_completed_boxes_new = 0
        self.num_completed_boxes = 0

        self.boundary_vertices = receptacle_vertices(self.receptacle_position, self.receptacle_half)
        self.outer_boundary_vertices = receptacle_vertices(
            self.receptacle_position, 
            [self.receptacle_half[0] + self.cfg.env.distance_between_inner_goal_and_outer_wall_width, 
             self.receptacle_half[1] + self.cfg.env.distance_between_inner_goal_and_outer_wall_length]
        )
        self.walls = []
        self.boundary_polygon = Polygon(self.boundary_vertices)
        self.outer_boundary_polygon = Polygon(self.outer_boundary_vertices)

        # state
        self.num_channels = 4
        self.observation = None
        self.global_overhead_map = None
        self.small_obstacle_map = None
        self.configuration_space = None
        self.configuration_space_thin = None
        self.closest_cspace_indices = None
        self.observation_init = False
        self.goal_point_global_map = None
        
        # stats
        self.inactivity_counter = None
        self.robot_cumulative_distance = None
        self.robot_cumulative_boxes = None
        self.robot_cumulative_reward = None
        self.total_work = [0, []]

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
        self.robot_distance = 0
        self.robot_turn_angle = 0

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

        # Box with wheels
        self.wheels_on_boxes = self.cfg.wheels_on_boxes.wheels_on_boxes
        self.wheels_mass = self.cfg.wheels_on_boxes.wheels_mass
        self.wheels_support_mass = self.cfg.wheels_on_boxes.wheels_support_mass
        self.wheels_sliding_friction = self.cfg.wheels_on_boxes.wheels_sliding_friction
        self.wheels_torsional_friction = self.cfg.wheels_on_boxes.wheels_torsional_friction
        self.wheels_rolling_friction = self.cfg.wheels_on_boxes.wheels_rolling_friction
        self.wheels_axle_damping_ratio = self.cfg.wheels_on_boxes.wheels_axle_damping_ratio
        self.wheels_support_damping_ratio = self.cfg.wheels_on_boxes.wheels_support_damping_ratio
        self.num_boxes_with_wheels = self.cfg.wheels_on_boxes.num_boxes_with_wheels
        self.num_boxes_without_wheels = self.num_boxes - self.num_boxes_with_wheels

        # generate random environment
        _, self.initialization_keepouts = generate_area_clearing_xml(N=self.cfg.boxes.num_boxes, env_type=self.cfg.env.area_clearing_version, file_name=xml_file,
                        ROBOT_clear=self.robot_radius, BOXES_clear=self.cfg.boxes.clearance, Z_BOX=self.cfg.boxes.box_half_size, ARENA_X=(0.0, self.room_length_inner), 
                        ARENA_Y=(0.0, self.room_width_inner), box_half_size=self.cfg.boxes.box_half_size, num_pillars=self.cfg.small_pillars.num_pillars, pillar_half=self.cfg.small_pillars.pillar_half,
                        wall_clearence_outer=[self.cfg.env.distance_between_inner_goal_and_outer_wall_length, self.cfg.env.distance_between_inner_goal_and_outer_wall_width], 
                        wall_clearence_inner=self.cfg.env.wall_clearence_inner, internal_clearance_length=self.cfg.env.internal_clearance_length, robot_radius=self.robot_radius, 
                        bumper_type=self.cfg.agent.type_of_bumper, bumper_mass= self.cfg.agent.bumper_mass, sim_timestep=self.cfg.env.sim_timestep,
                        wheels_on_boxes=self.wheels_on_boxes, wheels_mass=self.wheels_mass, wheels_support_mass=self.wheels_support_mass, wheels_sliding_friction=self.wheels_sliding_friction,
                        wheels_torsional_friction=self.wheels_torsional_friction, wheels_rolling_friction=self.wheels_rolling_friction, wheels_support_damping_ratio=self.wheels_support_damping_ratio, box_mass=self.cfg.boxes.box_mass,
                        box_sliding_friction= self.cfg.boxes.box_sliding_friction, box_torsional_friction= self.cfg.boxes.box_torsional_friction, box_rolling_friction= self.cfg.boxes.box_rolling_friction, 
                        num_boxes_with_wheels=self.cfg.wheels_on_boxes.num_boxes_with_wheels, wheels_axle_damping_ratio=self.wheels_axle_damping_ratio, agent_type=self.cfg.agent.agent_type)

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
            width=render_width,
            height=render_height,
            camera_id=0,
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

        self.robot_mass = self.model.body_mass[self.base_body_id]

        # Box joint addresses
        joint_id_boxes=[]
        for i in range (self.cfg.boxes.num_boxes):
            joint_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"box{i}_joint")
            joint_id_boxes.append(joint_id)
        self.joint_id_boxes = joint_id_boxes
        self.initial_box_joint_id = joint_id_boxes[0]
        self.completed_box_ids=[]

        # rewards
        if self.cfg.train.job_type == 'sam':
            rewards = self.cfg.rewards_sam
        else:
            rewards = self.cfg.rewards
        self.partial_rewards_scale = rewards.partial_rewards_scale
        self.goal_reward = rewards.goal_reward
        self.collision_penalty = rewards.collision_penalty
        self.non_movement_penalty = rewards.non_movement_penalty
        self.terminal_reward = rewards.terminal_reward

        # misc
        self.ministep_size = self.cfg.misc.ministep_size * scale_factor
        self.inactivity_cutoff = self.cfg.misc.inactivity_cutoff if self.cfg.train.job_type != 'sam' else self.cfg.misc.inactivity_cutoff_sam
        self.random_seed = self.cfg.misc.random_seed
        self.random_state = np.random.RandomState(self.random_seed)

        self.episode_idx = None

        self.path = None

        self.tries_before_inactive= self.cfg.train.tries_before_inactive
        
        if self.cfg.render.show_obs or self.cfg.render.show:
            # show state
            num_plots = self.num_channels
            self.state_plot = plt
            self.state_fig, self.state_ax = self.state_plot.subplots(1, num_plots, figsize=(4 * num_plots, 6))
            self.colorbars = [None] * num_plots
            if self.cfg.render.show_obs:
                self.state_plot.ion()  # Interactive mode on      
        
        self.boundary_goals, self.goal_points = self._compute_boundary_goals()

        self.position_controller = None

        ### DEBUG: Seperate figures for paper
        self.state_figs = []
        self.state_axes = []

        for i in range(self.num_channels):
            fig, ax = plt.subplots(figsize=(6, 6))
            self.state_figs.append(fig)
            self.state_axes.append(ax)

        self.colorbars = [None] * self.num_channels

    def render_env(self, mode='human', close=False):
        """Renders the environment."""

        if self.cfg.render.show:
            self.render()    
        
        if self.cfg.render.log_obs:
            directory = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
            if not os.path.exists(directory):
                os.makedirs(directory)

        if self.show_observation and self.observation is not None:# and self.t % self.cfg.render.frequency == 1:
            self.show_observation = False
            channel_name = ["Occupancy", "Footprint", "Egocentric DT", "Goal DT"]
            for ax, i in zip(self.state_ax, range(self.num_channels)):
                ax.clear()
                ax.set_title(channel_name[i])
                ax.set_xticks([])
                ax.set_yticks([])
                im = ax.imshow(self.observation[:,:,i], cmap='hot', interpolation='nearest')
                # if self.colorbars[i] is not None:
                #     self.colorbars[i].update_normal(im)
                # else:
                #     self.colorbars[i] = self.state_fig.colorbar(im, ax=ax)
            
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

            if self.cfg.render.show_obs:
                self.state_plot.draw()
                # self.state_plot.pause(0.001)
                self.state_plot.pause(0.1)
            
            if self.cfg.render.log_obs:
                self.state_fig.savefig(os.path.join(directory, str(self.t) + '_obs.png'))

                frame = self.mujoco_renderer.render(render_mode='rgb_array')
                Image.fromarray(frame).save(os.path.join(directory, str(self.t) + '_mujoco.png'))

        self.t += 1


    def render(self):
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode
        """
        if self.render_mode is not None:
            return self.mujoco_renderer.render(self.render_mode)


    def update_path(self, waypoints):
        # for i, point in enumerate(waypoints):
        #     self.model.site(f"wp{i}").pos[:2] = point[:2]           # set visualization point (x, y)
        #     self.model.site(f"wp{i}").pos[2] = 0.3                  # set visualization point z-axis

        # Move the rest out of view
        for i in range(len(waypoints), 100):
            site_id = self.model.site(f"wp{i}").id
            self.data.site_xpos[site_id] = np.array([1000, 1000, 1000])

    
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

        
    def step(self, action):

        self.robot_hit_obstacle = False

        robot_initial_position = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        robot_initial_heading = self.restrict_heading_range(quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7]))
        
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

        # TODO check if move_sign is necessary
        self.path, robot_move_sign = self.position_controller.get_waypoints_to_spatial_action(robot_initial_position, robot_initial_heading, action)

        # if self.cfg.render.show:
        #     self.renderer.update_path(self.path)
        self.update_path(self.path)

        self.execute_robot_path(robot_initial_position, robot_initial_heading)

        # check if episode is done
        terminated = False
        if len(self.joint_id_boxes) == len(self.completed_box_ids):
            terminated = True
        
        self.inactivity_counter += 1
        truncated = False
        if self.inactivity_counter >= self.inactivity_cutoff:
            terminated = True
            truncated = True

        # work
        _, boxes_vertices = dynamic_vertices(self.model, self.data, self.qpos_index_base, self.joint_id_boxes, self.robot_dimen, self.cfg.boxes.box_half_size)
        updated_boxes = [np.array(poly[0]) for poly in boxes_vertices]
        work = total_work_done(self.prev_boxes, updated_boxes, mass=self.cfg.boxes.box_mass)
        self.total_work[0] += work
        self.total_work[1].append(work)
        self.prev_boxes = updated_boxes

        # update position
        robot_position = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        robot_heading = self.restrict_heading_range(quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7]))

        # items to return
        reward = self._get_rew()
        self.robot_cumulative_distance += self.robot_distance
        self.robot_cumulative_boxes += self.num_completed_boxes_new
        self.robot_cumulative_reward += reward
        ministeps = self.robot_distance / self.ministep_size
        info = {
            'state': (round(robot_position[0], 2),
                      round(robot_position[1], 2),
                      round(robot_heading, 2)),
            'cumulative_distance': self.robot_cumulative_distance,
            'cumulative_boxes': self.robot_cumulative_boxes,
            'cumulative_reward': self.robot_cumulative_reward,
            'ministeps': ministeps,
            'total_work': self.total_work[0],
            'obs': updated_boxes,
            'box_completed_statuses': self.box_clearance_statuses,
            'goal_positions': self.goal_points,
        }
        
        self.observation=self.generate_observation(done=terminated)

        # render environment
        if self.cfg.render.show:
            self.show_observation = True
            self.render()

        return self.observation, reward, terminated, truncated, info
    
    def execute_robot_path(self, robot_initial_position, robot_initial_heading):
        robot_position = robot_initial_position.copy()
        robot_heading = robot_initial_heading
        robot_is_moving = True
        self.robot_distance = 0

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
            heading_diff = self.heading_difference(robot_heading, robot_waypoint_heading)
            if np.abs(heading_diff) > TURN_STEP_SIZE and np.abs(heading_diff - prev_heading_diff) > 0.001:
                pass
            else:
                done_turning = True
                if self.distance(robot_position, robot_waypoint_position) < MOVE_STEP_SIZE:
                    robot_new_position = robot_waypoint_position

            # change robot pose (use controller)
            v, w, dist = make_controller(robot_prev_position, robot_prev_heading, robot_waypoint_position)

            # otherwise drive as normal
            v_l, v_r = vw_to_wheels(v, w)

            # apply the control 'frame_skip' steps
            self.do_simulation([v_l, v_r], self.frame_skip)

            # get new robot pose
            robot_position = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
            robot_heading = self.restrict_heading_range(quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7]))
            prev_heading_diff = heading_diff

            # stop moving if robot collided with obstacle
            self.robot_hit_obstacle = self.robot_hits_static()
            if self.distance(robot_prev_waypoint_position, robot_position) > MOVE_STEP_SIZE:
                if self.robot_hit_obstacle:
                    robot_is_moving = False
                    break   # Note: self.robot_distance does not get updated

            # stop if robot reached waypoint
            if (self.distance(robot_position, robot_waypoint_positions[robot_waypoint_index]) < WAYPOINT_MOVING_THRESHOLD
                    and np.abs(robot_heading - robot_waypoint_headings[robot_waypoint_index]) < WAYPOINT_TURNING_THRESHOLD):

                # update distance moved
                self.robot_distance += self.distance(robot_prev_waypoint_position, robot_position)

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
            if sim_steps % 20 == 0 and self.cfg.render.show:
                self.render_env()

            # break if robot is stuck
            if sim_steps > STEP_LIMIT:
                break

        robot_angle = quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7])
        robot_heading = self.restrict_heading_range(robot_angle)
        self.robot_turn_angle = self.heading_difference(robot_initial_heading, robot_heading)

    # Observation generation functions

    def generate_observation(self, done=False):
        """ Generates the observation for the environment."""

        if self.cfg.train.job_type == 'sam' and done:
            return None

        # Getting the robot and boxes vertices
        robot_properties, boxes_vertices = dynamic_vertices(self.model, self.data, self.qpos_index_base, 
                                                            self.joint_id_boxes, self.robot_dimen, 
                                                            self.cfg.boxes.box_half_size)

        # Update the global overhead map with the current robot and boundaries
        self.update_global_overhead_map(robot_properties[1], boxes_vertices)

        robot_position = robot_properties[3]
        robot_angle = robot_properties[2]

        # Create the robot state channel
        channels = []

        channels.append(self.get_local_map(self.global_overhead_map, robot_position, robot_angle))
        channels.append(self.robot_state_channel)
        channels.append(self.get_local_distance_map(self.create_global_shortest_path_map(robot_position), robot_position, robot_angle))
        channels.append(self.get_local_distance_map(self.goal_point_global_map, robot_position, robot_angle))
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
    
    def create_global_shortest_path_to_goal_points(self):
        """ Creates a global shortest path map to the goal area."""
        # Create a padded room of zeros and compute the shortest path to each goal point
        global_map = self.create_padded_room_zeros() + np.inf
        for point in self.goal_points:
            rx, ry = point.x, point.y
            pixel_i, pixel_j = self.position_to_pixel_indices(rx, ry, self.configuration_space.shape)
            pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
            shortest_path_image, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
            # Scale the shortest path image to the local map pixel width
            shortest_path_image /= self.local_map_pixels_per_meter
            global_map = np.minimum(global_map, shortest_path_image)

        # Scale the global map to the local map pixel width and normalize it
        global_map /= (np.sqrt(2) * self.local_map_pixel_width) / self.local_map_pixels_per_meter

        max_value = np.max(global_map)
        min_value = np.min(global_map)
        global_map = (global_map - min_value) / (max_value - min_value) * DISTANCE_SCALE_MAX

        # fill points outside boundary polygon with 0
        for i in range(global_map.shape[0]):
            for j in range(global_map.shape[1]):
                x, y = self.pixel_indices_to_position(i, j, self.configuration_space.shape)
                if not self.boundary_polygon.contains(Point(x, y)):
                    global_map[i, j] = 0
                if not self.outer_boundary_polygon.contains(Point(x, y)):
                    global_map[i, j] = 1

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
        wall_vertices, self.columns_from_keepout, side_vertices, self.walls = precompute_static_vertices(self.initialization_keepouts, self.wall_thickness, self.room_width_inner, self.room_length_inner, [self.cfg.env.distance_between_inner_goal_and_outer_wall_length, self.cfg.env.distance_between_inner_goal_and_outer_wall_width], self.cfg.env.wall_clearence_inner, self.cfg.env.area_clearing_version )

        # Iterating through each wall vertex and keepout columns
        for wall_vertices_each_wall in wall_vertices + self.columns_from_keepout + side_vertices:

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

    def update_global_overhead_map(self, robot_vertices, boxes_vertices):
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

        # Draw the pillars
        for column in self.columns_from_keepout:
            draw_object(column[1], 0)

        # Draw the robot
        draw_object(robot_vertices, ROBOT_SEG_INDEX)

        # Draw the boxes
        first_box_id = self.joint_id_boxes[0]
        for i in range(len(boxes_vertices)):
            if (i + first_box_id) in self.completed_box_ids and not self.wheels_on_boxes:
                draw_object(boxes_vertices[i][0], COMPLETED_NON_WHEELED_BOX_SEG_INDEX)
            elif (i + first_box_id) in self.completed_box_ids and self.wheels_on_boxes and i>=(self.num_boxes_without_wheels):
                draw_object(boxes_vertices[i][0], COMPLETED_WHEELED_BOX_SEG_INDEX)
            elif not self.wheels_on_boxes or i<(self.num_boxes_without_wheels):
                draw_object(boxes_vertices[i][0], NON_WHEELED_BOX_SEG_INDEX)
            else:
                draw_object(boxes_vertices[i][0], WHEELED_BOX_SEG_INDEX)

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
        self.motion_dict, boxes_total_distance = self.update_motion_dict(self.motion_dict)

        if abs(boxes_total_distance) > 0.01:
            robot_reward += self.partial_rewards_scale * boxes_total_distance

        # reward for boxes in receptacle
        self.completed_box_ids, self.num_completed_boxes_new = transport_box_from_recept(self.model, self.data, self.joint_id_boxes, self.room_width_inner,
                                                                                          self.room_length_inner, self.completed_box_ids, goal_half=self.receptacle_half, goal_center=self.receptacle_position, box_half_size=self.cfg.boxes.box_half_size)
        if self.num_completed_boxes_new > 0:
            self.inactivity_counter = 0
        robot_reward += self.goal_reward * self.num_completed_boxes_new

        for id in self.completed_box_ids:
            self.box_clearance_statuses[id - self.joint_id_boxes[0]] = True

        self.num_completed_boxes = len(self.completed_box_ids)
        if(self.num_completed_boxes_new != 0):
            print('Num of completed boxes:', self.num_completed_boxes)

        if not(self.cfg.train.job_type == 'sam') and self.num_completed_boxes == self.num_boxes:
            robot_reward += self.terminal_reward

        # penalty for hitting obstacles
        if self.robot_hit_obstacle:
            robot_reward -= self.collision_penalty
    
        # penalty for small movements
        if self.robot_distance < NONMOVEMENT_DIST_THRESHOLD and abs(self.robot_turn_angle) < NONMOVEMENT_TURN_THRESHOLD:
            robot_reward -= self.non_movement_penalty
        
        # Compute stats
        self.robot_cumulative_reward += robot_reward

        return robot_reward

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

    def _distance_to_nearest_edge(self, xy):
        """
        Shortest-path distance from a point (x,y) to the closest of the
        four room edges, measured in the current configuration space.
        """
        x, y = xy

        # four “targets” lying on each wall at the same y or x
        targets = [
            (0.0,          y),               # left   edge   (x = 0)
            (self.room_width_inner,  y),           # right  edge   (x = W)
            (x,           0.0),              # bottom edge   (y = 0)
            (x,  self.room_length_inner),          # top    edge   (y = L)
        ]
        
        # # # compute straight-line distance to each target
        straight_dists = [self.distance(xy, t) for t in targets]
        dist = min(straight_dists)  # quick exit if possible

        # # return the minimum SPFA distance
        # # TODO: Enable for environments with columns
        # dist = min(self.shortest_path_distance(xy, t) for t in targets)

        return dist

    def update_motion_dict(self, motion_dict):
        """
        In-place update of motion_dict.
        Skips boxes that have already been teleported away (their body_id is gone).
        """
        goal_xy = np.array([
        self.receptacle_position[0] + 0.5 * self.room_width_inner,
        self.receptacle_position[1] + 0.5 * self.room_length_inner])

        # robot
        length, last_xy, mass = motion_dict[self.base_body_id]
        cx, cy = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        dist = np.linalg.norm(np.array([cx, cy]) - last_xy)
        motion_dict[self.base_body_id][0] += dist
        motion_dict[self.base_body_id][1][:] = (cx, cy)

        # boxes still present
        live_bodies_joint_id = list(set(self.joint_id_boxes) - set(self.completed_box_ids))
        live_bodies = {self.model.jnt_bodyid[jid] for jid in live_bodies_joint_id}
        boxes_total_distance=0

        # obs_to_goal_difference(self.prev_obs, updated_obstacles, self.goal_points, self.boundary_polygon)
        for body_id in live_bodies:

            length, last_xy, mass = motion_dict[body_id]

            adr  = self.model.body_jntadr[body_id]
            jpos = self.model.jnt_qposadr[adr]
            curr_xy = self.data.qpos[jpos : jpos+2]

            # progress *before* we mutate last_xy
            prev_d = self._distance_to_nearest_edge(last_xy)
            curr_d = self._distance_to_nearest_edge(curr_xy)
            boxes_total_distance += prev_d - curr_d

            # book-keeping for next step
            motion_dict[body_id][0] += np.linalg.norm(curr_xy - last_xy)
            motion_dict[body_id][1][:] = curr_xy

        return motion_dict, boxes_total_distance

    # contacts
    def robot_hits_static(self) -> bool:

        ROBOT_PREFIX    = "base"
        STATIC_PREFIXES = ("wall", "small_col")

        for k in range(self.data.ncon):
            c   = self.data.contact[k]
            g1, g2 = c.geom1, c.geom2

            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)

            # skip any unnamed geoms
            if not n1 or not n2:
                continue

            # check robot vs any static prefix
            hit1 = ROBOT_PREFIX in n1.lower() and any(pref in n2.lower() for pref in STATIC_PREFIXES)
            hit2 = ROBOT_PREFIX in n2.lower() and any(pref in n1.lower() for pref in STATIC_PREFIXES)

            if hit1 or hit2:
                return True
        return False

    def reset_model(self):
        """
        Randomly sample non-overlapping (x, y, theta) for robot and boxes.
        Teleport them in simulation using sim.data.qpos.
        """
        positions=[]
        
        if self.episode_idx is None:
            self.episode_idx = 0
        else:
            self.episode_idx += 1

        self.t = 0

        def is_valid(pos, radius):
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

            return True

        def resolve_robot_box_overlaps(proposed_x_y, shift=0.6):

            # geometry / margins
            box_half = float(self.cfg.boxes.box_half_size)
            r_box = float(np.hypot(box_half, box_half))

            # extra margin to keep from robot (use your robot_clear as requested)
            need_robot_box = self.cfg.agent.robot_clear

            # box–box spacing needed (keep using your boxes.clearance here)
            box_margin = 0.0
            need_box_box = 2 * box_half

            # bounds (same idea as your sampler uses for boxes)
            x_min = -self.room_length_inner / 2 + 0.1
            x_max =  self.room_length_inner / 2 - 0.1
            y_min = -self.room_width_inner  / 2 + 0.1
            y_max =  self.room_width_inner  / 2 - 0.1

            def in_bounds(x, y):
                return (x_min <= x <= x_max) and (y_min <= y <= y_max)

            # current robot center (x,y)
            rx, ry = proposed_x_y

            # collect active box centers (ignore "parked" ones far below)
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

            # make an editable list of current box centers for pairwise checks
            centers = [(i, jid, bx, by) for (i, jid, bx, by) in active_boxes]

            # pass over boxes; if overlapping, try ±shift in X
            for k, (i, jid, bx, by) in enumerate(list(centers)):
                # overlap with robot?
                if np.hypot(bx - rx, by - ry) >= need_robot_box:
                    continue  # already fine

                # candidate moves: +shift (right), -shift (left)
                candidates = [(bx + shift, by), (bx - shift, by)]

                # keep those inside plane and valid wrt robot/boxes/keepouts
                good = []
                for (cx, cy) in candidates:
                    if in_bounds(cx, cy) and box_pose_ok(i, cx, cy, centers):
                        # score by distance to robot; prefer the farther one
                        score = np.hypot(cx - rx, cy - ry)
                        good.append((score, cx, cy))

                if good:
                    # choose the candidate giving max distance to robot
                    _, nx, ny = max(good, key=lambda t: t[0])

                    # apply to MuJoCo state
                    adr = int(self.model.jnt_qposadr[jid])
                    self.data.qpos[adr:adr+2] = [nx, ny]           # keep z, quat as-is

                    # update local centers list so subsequent checks see the move
                    centers[k] = (i, jid, nx, ny)
                    return True  # moved box i
                else:
                    return False

        # Define bounds of the placement area (slightly inside the walls)
        y_min_robot, y_max_robot = -self.room_width_inner / 2 + self.edges_space_left_for_robot, self.room_width_inner / 2 - self.edges_space_left_for_robot
        x_min_robot, x_max_robot = -self.room_length_inner / 2 + self.edges_space_left_for_robot, self.room_length_inner / 2 - self.edges_space_left_for_robot

        # Sample robot pose
        while True:
            x = np.random.uniform(x_min_robot, x_max_robot)
            y = np.random.uniform(y_min_robot, y_max_robot)
            theta = np.random.uniform(-np.pi, np.pi)
            if is_valid((x, y, theta), self.cfg.agent.robot_clear):
                positions.append(((x, y, theta), self.cfg.agent.robot_clear))
                break

        # Set robot pose
        base_qpos_addr = self.model.jnt_qposadr[self.base_joint_id]
        self.data.qpos[base_qpos_addr:base_qpos_addr+3] = [x, y, self.placement_height]  # x, y, z
        if self.agent_type == "turtlebot_3":
            self.data.qpos[base_qpos_addr+3:base_qpos_addr+7] = quat_z(theta)

        self.data.qvel[base_qpos_addr:base_qpos_addr+6] = 0

        # Define bounds of the placement area (slightly inside the walls)
        y_min = -self.room_width_inner / 2 + self.internal_clearance_length
        y_max = self.room_width_inner / 2 - self.internal_clearance_length
        x_min = -self.room_length_inner / 2 + self.internal_clearance_length
        x_max = self.room_length_inner / 2 - self.internal_clearance_length

        # Assume box is square with radius from center to corner (diagonal/2)
        box_half_size = self.cfg.boxes.box_half_size
        box_r = np.sqrt(box_half_size ** 2 + box_half_size ** 2)

        if self.wheels_on_boxes:
            z = box_half_size+0.03
        else:
            z = box_half_size+0.005

        for i in range(self.num_boxes):
            while True:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                theta = np.random.uniform(-np.pi, np.pi)
                if is_valid((x, y, theta), box_r):
                    positions.append(((x, y, theta), box_r))
                    break
            
            qadr = self.model.jnt_qposadr[self.joint_id_boxes[i]]
            self.data.qpos[qadr:qadr+3] = np.array([x, y, z])
            self.data.qpos[qadr+3:qadr+7] = quat_z(theta)
            self.data.qvel[qadr:qadr+6] = 0
        
        # Update the simulation state
        mujoco.mj_forward(self.model, self.data)

        # self._prev_robot_xy = get_body_pose_2d(self.model, self.data, 'base')[:2]
        self.do_simulation([0, 0], 1)
        self._prev_robot_heading = self.restrict_heading_range(quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7]))

        # get the robot and boxes vertices
        robot_properties, boxes_vertices = dynamic_vertices(self.model, self.data, self.qpos_index_base, self.joint_id_boxes, self.robot_dimen, self.cfg.boxes.box_half_size)

        self.motion_dict = self.init_motion_dict()

        # reset stats
        self.inactivity_counter = 0
        self.robot_cumulative_distance = 0
        self.robot_cumulative_boxes = 0
        self.robot_cumulative_reward = 0
        self.total_work = [0, []]
        self.completed_box_ids = []

        self.update_configuration_space()

        # reset map
        self.global_overhead_map = self.create_padded_room_zeros()
        self.update_global_overhead_map(robot_properties[1], boxes_vertices)

        if self.goal_point_global_map is None:
            self.goal_point_global_map = self.create_global_shortest_path_to_goal_points()

        observation = self.generate_observation()

        self.box_clearance_statuses = [False for _ in range(self.num_boxes)]

        self.prev_boxes = [np.array(poly[0]) for poly in boxes_vertices]

        self.position_controller = PositionController(self.cfg, self.robot_radius, self.room_width, self.room_length, 
                                                      self.configuration_space, self.configuration_space_thin, self.closest_cspace_indices, 
                                                      self.local_map_pixel_width, self.local_map_width, self.local_map_pixels_per_meter,
                                                      TURN_STEP_SIZE, MOVE_STEP_SIZE, WAYPOINT_MOVING_THRESHOLD, WAYPOINT_TURNING_THRESHOLD)

        self.num_completed_boxes = 0
        self.num_completed_boxes_new = 0
        self.completed_box_ids=[]

        self.start_time = time.time()

        return observation

    def _get_reset_info(self):
        robot_position = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        robot_heading = self.restrict_heading_range(quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7]))
        info = {
            'state': (round(robot_position[0], 2),
                      round(robot_position[1], 2),
                      round(robot_heading, 2)),
            'cumulative_distance': self.robot_cumulative_distance,
            'cumulative_boxes': self.robot_cumulative_boxes,
            'cumulative_reward': self.robot_cumulative_reward,
            'ministeps': 0,
            'total_work': self.total_work[0],
            'obs': self.prev_boxes, # updated from reset
            'box_completed_statuses': self.box_clearance_statuses,
            'goal_positions': self.goal_points,
        }
        return info