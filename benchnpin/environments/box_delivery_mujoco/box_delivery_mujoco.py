from typing import Dict, Union

import numpy as np
from benchnpin.common.utils.utils import DotDict
from benchnpin.environments.box_delivery_mujoco.box_delivery_utils import generate_boxDelivery_xml, transporting
# from benchnpin.environments.box_delivery_mujoco.Structured_env import transporting
from benchnpin.common.utils.mujoco_utils import vw_to_wheels, make_controller

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
        generate_boxDelivery_xml(N=self.cfg.boxes.num_boxes, env_type=self.cfg.env.obstacle_config, file_name=xml_file, stl_model_path_entered=None,
                        ROBOT_R=0.15, CLEAR=0.20, Z_CUBE=0.02, ARENA_X=(0.0, 1.575), ARENA_Y=(0.0, 2.845), cube_half_size=0.04)

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

        self.observation_space = Box(low=0, high=255, shape=(100, 100), dtype=np.uint8)

        # get robot body & joint addresses
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        joint_adr = self.model.body_jntadr[self.base_body_id]
        self.qpos_index_base = self.model.jnt_qposadr[joint_adr]

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

        observation = np.zeros((100, 100)).astype(np.uint8)
        reward = 0
        info = {}
        return observation, reward, False, False, info


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
        self.set_state(self.init_qpos, self.init_qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
        }
