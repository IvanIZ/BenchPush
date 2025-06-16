from typing import Dict, Union
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt

from benchnpin.common.utils.utils import DotDict
from benchnpin.common.occupancy_grid.occupancy_map_mujoco import OccupancyGrid
from benchnpin.environments.maze_NAMO_mujoco.maze_utils import generate_maze_xml, intersects_keepout
from benchnpin.common.utils.mujoco_utils import vw_to_wheels, quat_z, inside_poly, get_body_pose_2d, get_box_2d_vertices, quat_z_yaw, corners_xy

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
from gymnasium import error, spaces

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class MazeNAMOMujoco(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        frame_skip: int = 10,
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
        
        # get correct maze version
        if self.cfg.maze_version == 1:
            self.cfg.env = self.cfg.env1
        elif self.cfg.maze_version == 2:
            self.cfg.env = self.cfg.env2
        else:
            raise Exception("Invalid Maze Version!")

        # initialize occupancy helper class
        grid_size = 1 / self.cfg.occ.m_to_pix_scale
        self.occupancy = OccupancyGrid(grid_width=grid_size, grid_height=grid_size, map_width=self.cfg.env.width, map_height=self.cfg.env.length, 
                                       local_width=self.cfg.occ.local_width, local_height=self.cfg.occ.local_height,
                                       ship_body=None, meter_to_pixel_scale=self.cfg.occ.m_to_pix_scale)

        self.beta = 1.5             # amount to scale the collision reward
        self.k = 2                  # amount to scale the distance reward
        self.k_increment = 150
        self.episode_idx = None     # the increment of this index is handled in reset()

        # Define observation space
        self.low_dim_state = self.cfg.low_dim_state
        if self.low_dim_state:
            #low dimensional observation space comprises of the 2D positions of each obstacle in addition to the robot
            self.observation_space = spaces.Box(low=-10, high=30, shape=((self.cfg.boxes.num_boxes+1) * 2,), dtype=np.float64)
        
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


        # generate random environmnt
        xml_file = os.path.join(self.current_dir, 'turtlebot3_burger_updated.xml')
        _, self.maze_walls = generate_maze_xml(N=self.cfg.boxes.num_boxes, maze_version=self.cfg.maze_version, file_name=xml_file,
                        ROBOT_R=self.cfg.agent.robot_r, CLEAR=self.cfg.boxes.clearance, Z_CUBE=0.02, ARENA_X=(0.0, self.cfg.env.width), 
                        ARENA_Y=(0.0, self.cfg.env.length), cube_half_size=0.04, clearance_poly=self.cfg.env.clearance_poly, goal_center=self.cfg.env.goal_position, 
                        goal_half=self.cfg.env.goal_size)


        # compute the global distance map
        self.global_distance_map, self.unnormalized_dist_map = self.occupancy.global_goal_point_dist_transform(self.cfg.env.goal_position, self.maze_walls)

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
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # Define action space
        self.max_linear_speed = 1.0
        self.min_linear_speed = 0.0
        self.max_yaw_rate_step = (np.pi/2) / 15        # rad/sec
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float64)

        # get robot body & joint addresses
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        joint_adr = self.model.body_jntadr[self.base_body_id]
        self.qpos_index_base = self.model.jnt_qposadr[joint_adr]
        self.base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint")

        # Cube joint addresses
        joint_id_boxes=[]
        for i in range (self.cfg.boxes.num_boxes):
            joint_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube{i}_joint")
            joint_id_boxes.append(joint_id)
        self.joint_id_boxes = joint_id_boxes

        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))


    def step(self, action):
        self.t += 1
        
        v, w = action

        # otherwise drive as normal
        v_l, v_r = vw_to_wheels(v, w)

        # apply the control 'frame_skip' steps, by default 5
        self.do_simulation([v_l, v_r], self.frame_skip)

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        # get observation
        observation = self.generate_observation()

        if self.cfg.log_obs:
            self.log_observation(observation)

        reward = 0
        info = {}
        return observation, reward, False, False, info


    def generate_observation(self):

        # get robot state
        robot_pose = get_body_pose_2d(self.model, self.data, body_name='base')

        # get local robot vertives
        robot_half = (0.07, 0.09)
        local_robot_vertices = np.array([[-robot_half[0], -robot_half[1]],
                                [ robot_half[0], -robot_half[1]],
                                [ robot_half[0],  robot_half[1]],
                                [-robot_half[0],  robot_half[1]]])

        obstacles = self.get_box_vertices()
        
        robot_footprint_local , movable_obstacles_local, wall_local, distance_map_local = self.occupancy.ego_view_map_maze(robot_pose, 
                                                                local_robot_vertices, obstacles, self.maze_walls, self.global_distance_map)
        observation = np.concatenate((np.array([robot_footprint_local]), 
                                      np.array([movable_obstacles_local]), np.array([wall_local]), np.array([distance_map_local])))  # (5, local H, local W)
        #for resnet input
        observation = (observation*255).astype(np.uint8)
        return observation

    
    def log_observation(self, observation):
        robot_footprint, movable_obs, fixed_obs, distance_map = observation

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

    
    def get_box_vertices(self):
        obstacles = []
        for i in range(self.cfg.boxes.num_boxes):
            body_name = "cube" + str(i)
            vertices = get_box_2d_vertices(self.model, self.data, body_name)
            obstacles.append(vertices)
        return np.array(obstacles)


    def get_robot_vertices(self):
        # robot vertices
        cx, cy = self.data.qpos[self.qpos_index_base:self.qpos_index_base+2]
        qw, qx, qy, qz = self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7]
        yaw  = quat_z_yaw(qw, qx, qy, qz)
        robot_half = (0.07, 0.09)
        local_robot = np.array([[-robot_half[0], -robot_half[1]],
                                [ robot_half[0], -robot_half[1]],
                                [ robot_half[0],  robot_half[1]],
                                [-robot_half[0],  robot_half[1]]])

        return corners_xy(np.array([cx, cy]), yaw, local_robot)


    def _get_rew(self):
        reward = 0

        # decomposition of reward
        reward_info = {
            
        }
        return reward, reward_info


    def _get_obs(self):

        observation = self.generate_observation()
        return observation


    def reset_model(self):
        """
        Randomly sample non-overlapping (x, y, theta) for robot and boxes.
        Teleport them in simulation using sim.data.qpos.

        Args:
            robot_r (float): Robot radius, used for clearance check.
            clearance (float): Minimum distance between any two boxes.
        """

        # update counters
        if self.episode_idx is None:
            self.episode_idx = 0
        else:
            self.episode_idx += 1
        self.t = 0

        positions = []

        def is_valid(pos, radius):

            # 1) check against previously placed robot/boxes
            for p, r in positions:
                if np.linalg.norm(np.array(pos[:2]) - np.array(p[:2])) < (r + radius + self.cfg.boxes.clearance):
                    return False
            x, y, _ = pos

            # 2) check overall clearance polygon
            if not inside_poly(x, y, self.cfg.env.clearance_poly):
                return False

            # 2) check against pillar keep-outs
            if intersects_keepout(x, y, self.maze_walls):
                return False

            return True

        # Define bounds of the placement area (slightly inside the walls)
        x_min, x_max = 0.1, 1.5
        y_min, y_max = 0.1, 2.7

        # set robot pose
        if self.cfg.maze_version == 1:
            x = self.cfg.env.width * (3 / 4)
            y = self.cfg.env.width * (1 / 4)
            theta = np.pi / 2
            positions.append(((x, y, theta), self.cfg.agent.robot_r))

        elif self.cfg.maze_version == 2:
            x = self.cfg.env.width * (5 / 6)
            y = self.cfg.env.length - self.cfg.env.width * (1 / 6)
            theta = -np.pi / 2

        # Set robot pose
        base_qpos_addr = self.model.jnt_qposadr[self.base_joint_id]
        self.data.qpos[base_qpos_addr:base_qpos_addr+3] = [x, y, 0.01]  # x, y, z
        self.data.qpos[base_qpos_addr+3:base_qpos_addr+7] = quat_z(theta)

        self.data.qvel[base_qpos_addr:base_qpos_addr+6] = 0

        # Assume box is square with radius from center to corner (diagonal/2)
        box_r = np.sqrt(0.04 ** 2 + 0.04 ** 2)

        for i in range(self.cfg.boxes.num_boxes):
            while True:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                theta = np.random.uniform(-np.pi, np.pi)
                if is_valid((x, y, theta), box_r):
                    positions.append(((x, y, theta), box_r))
                    break
            
            qadr = self.model.jnt_qposadr[self.joint_id_boxes[i]]
            self.data.qpos[qadr:qadr+2] = np.array([x, y])
            self.data.qpos[qadr+3:qadr+7] = quat_z(theta)
            self.data.qvel[qadr:qadr+6] = 0

        observation = self._get_obs()
        return observation


    def _get_reset_info(self):
        return {
        }
