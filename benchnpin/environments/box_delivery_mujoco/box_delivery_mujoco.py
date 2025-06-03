from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray
from benchnpin.common.utils.utils import DotDict
from benchnpin.environments.box_delivery_mujoco.box_delivery_utils import generate_boxDelivery_xml, transporting
from benchnpin.common.utils.mujoco_utils import vw_to_wheels, make_controller, quat_z, inside_poly

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

class BoxDeliveryMujoco(MujocoEnv, utils.EzPickle):

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
        frame_skip: int = 1,
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


        # generate random environmnt
        xml_file = os.path.join(self.current_dir, 'turtlebot3_burger_updated.xml')
        _, self.initialization_keepouts = generate_boxDelivery_xml(N=self.cfg.boxes.num_boxes, env_type=self.cfg.env.obstacle_config, file_name=xml_file,
                        ROBOT_R=self.cfg.agent.robot_r, CLEAR=self.cfg.boxes.clearance, Z_CUBE=0.02, ARENA_X=(0.0, 1.575), 
                        ARENA_Y=(0.0, 2.845), cube_half_size=0.04, clearance_poly=self.cfg.env.clearance_poly)

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

        # environment
        self.local_map_pixel_width = self.cfg.env.local_map_pixel_width if self.cfg.train.job_type != 'sam' else self.cfg.env.local_map_pixel_width_sam
        self.local_map_width = self.cfg.env.local_map_width
        self.local_map_pixels_per_meter = self.local_map_pixel_width / self.local_map_width
        self.room_length = self.cfg.env.room_length
        self.wall_thickness = self.cfg.env.wall_thickness
        env_size = self.cfg.env.obstacle_config.split('_')[0]
        self.num_boxes = self.cfg.boxes.num_boxes
        if env_size == 'small':
            self.room_width = self.cfg.env.room_width_small
        else:
            self.room_width = self.cfg.env.room_width_large

        # observation space
        self.observation_space = Box(low=0, high=255, shape=(100, 100), dtype=np.uint8)

        # Define action space
        if self.cfg.agent.action_type == 'velocity':
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'heading':
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'position':
            self.action_space = spaces.Box(low=0, high=self.local_map_pixel_width * self.local_map_pixel_width, dtype=np.float32)

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


    def step(self, action):
        
        goal = action

        # navigate to the desired goal
        step_count = 0
        while True:
            step_count += 1

            v, w, dist = make_controller(self.model, self.data, self.qpos_index_base, goal)

            if dist < 0.02:
                    
                # Arrived at location
                self.data.ctrl[:] = 0.0
                print(f"Reached goal {goal}")
                break

            # otherwise drive as normal
            v_l, v_r = vw_to_wheels(v, w)

            # apply the control 'frame_skip' steps
            self.do_simulation([v_l, v_r], self.frame_skip)

            # teleporting boxes inside the receptacle
            transporting(self.model, self.data, self.joint_id_boxes)

            if self.render_mode == "human" and step_count % 10 == 0:
                self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        # get observation
        # obs = self.generate_observation()

        observation = np.zeros((100, 100)).astype(np.uint8)
        reward = 0
        info = {}
        return observation, reward, False, False, info


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


    def _get_rew(self):
        reward = 0

        # decomposition of reward
        reward_info = {
            
        }
        return reward, reward_info


    def _get_obs(self):

        observation = np.zeros((100, 100)).astype(np.uint8)
        return observation


    def reset_model(self):
        """
        Randomly sample non-overlapping (x, y, theta) for robot and boxes.
        Teleport them in simulation using sim.data.qpos.

        Args:
            robot_r (float): Robot radius, used for clearance check.
            clearance (float): Minimum distance between any two boxes.
        """
        positions = []

        def is_valid(pos, radius):
            for p, r in positions:
                if np.linalg.norm(np.array(pos[:2]) - np.array(p[:2])) < (r + radius + self.cfg.boxes.clearance):
                    return False
            
            x, y, _ = pos
            if not inside_poly(x, y, self.cfg.env.clearance_poly):
                return False
            return True

        # Define bounds of the placement area (slightly inside the walls)
        x_min, x_max = 0.1, 1.5
        y_min, y_max = 0.1, 2.7

        # Sample robot pose
        while True:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            theta = np.random.uniform(-np.pi, np.pi)
            if is_valid((x, y, theta), self.cfg.agent.robot_r):
                positions.append(((x, y, theta), self.cfg.agent.robot_r))
                break

        # Set robot pose
        base_qpos_addr = self.model.jnt_qposadr[self.base_joint_id]
        self.data.qpos[base_qpos_addr:base_qpos_addr+3] = [x, y, 0.01]  # x, y, z
        self.data.qpos[base_qpos_addr+3:base_qpos_addr+7] = quat_z(theta)

        self.data.qvel[base_qpos_addr:base_qpos_addr+6] = 0

        # Assume box is square with radius from center to corner (diagonal/2)
        box_r = np.sqrt(0.04 ** 2 + 0.04 ** 2)

        for i in range(self.num_boxes):
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
