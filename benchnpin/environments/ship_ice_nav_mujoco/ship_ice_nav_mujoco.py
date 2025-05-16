from typing import Dict, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class ShipIceMujoco(MujocoEnv, utils.EzPickle):

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
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        cfg=None,
        **kwargs,
    ):

        # get current directory of this script
        self.current_dir = os.path.dirname(__file__)

        # construct absolute path to the env_config folder
        xml_file = os.path.join(self.current_dir, 'asv_ice_planar_random.xml')

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


    def step(self, action):
        
        # apply the control 'frame_skip' steps
        self.do_simulation(action, self.frame_skip)

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