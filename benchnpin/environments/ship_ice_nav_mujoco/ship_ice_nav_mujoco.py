from typing import Dict, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
from gymnasium import error, spaces
import random

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e

from benchnpin.common.utils.mujoco_utils import get_body_pose_2d, get_box_2d_vertices
from benchnpin.common.utils.utils import DotDict
from benchnpin.environments.ship_ice_nav_mujoco.ship_ice_utils import generate_shipice_xml, apply_fluid_forces_to_body, load_ice_field


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

        # get configurations
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

        # construct absolute path to the env_config folder
        xml_file = os.path.join(self.current_dir, 'asv_ice_planar_random.xml')

        # build xml file
        self.num_floes, self.ice_area_dict = generate_shipice_xml(self.cfg.concentration, xml_file, self.cfg.sim.timestep_sim, 
            self.cfg.environment.channel_len, self.cfg.environment.channel_wid, 
            self.cfg.environment.icefield_len, self.cfg.environment.icefield_wid, 
            load_cached=True, trial_idx=0)
        self.num_floes, self.ice_area_dict = load_ice_field(self.cfg.concentration, xml_file, self.cfg.sim.timestep_sim, 
            self.cfg.environment.channel_len, self.cfg.environment.channel_wid, 
            self.cfg.environment.icefield_len, self.cfg.environment.icefield_wid, 
            load_cached=True, trial_idx=0)

        self.phase = 0.0

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

        self.goal = (0, self.cfg.goal_y)
        self.max_yaw_rate_step = (np.pi/2) / 7        # rad/sec

        self.observation_space = Box(low=0, high=255, shape=(100, 100), dtype=np.uint8)


    def step(self, action):
        
        # apply the control 'frame_skip' steps
        self.do_simulation(action, self.frame_skip)

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        observation = np.zeros((100, 100)).astype(np.uint8)
        reward = 0

        # get ship state
        state = get_body_pose_2d(self.model, self.data, body_name='asv')

        # get vertices of all floes
        obs = self.get_floe_vertices()

        info = {
            'obs': obs, 
            'state': state
        }

        return observation, reward, False, False, info


    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulation.
        """
        self.phase += 0.2 * self.cfg.sim.timestep_sim
        self.data.ctrl[:] = ctrl

        # drag and wave force (ship)
        # frontal area is an approximation here for the part of ship submerged in fluid
        apply_fluid_forces_to_body(self.model, self.data, 'asv', 'asv', self.phase, self.ice_area_dict)

        # Apply drag to all ice floes
        for n in range(self.num_floes):
            name = f"ice_{n}"
            apply_fluid_forces_to_body(self.model, self.data, name, name, self.phase, self.ice_area_dict)
        
        mujoco.mj_step(self.model, self.data)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        # mujoco.mj_rnePostConstraint(self.model, self.data)
        
        mujoco.mju_zero(self.data.qfrc_applied)



    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)


    def get_floe_vertices(self):
        """
        gets vertices of all floes, returns a list of shape (num_floes, 4, 2)
        """
        obs = []
        # for n in range(self.num_floes):
        #     name = f"ice_{n}"
        #     floe_vertices = get_box_2d_vertices(model=self.model, data=self.data, body_name=name)
        #     obs.append(floe_vertices)
        return obs


    def _get_rew(self):
        reward = 0

        # decomposition of reward
        reward_info = {
            
        }
        return reward, reward_info


    def _get_obs(self):

        observation = np.zeros((100, 100)).astype(np.uint8)
        return observation


    def update_path(self, waypoints):
        for i, point in enumerate(waypoints):
            self.model.site(f"wp{i}").pos[:2] = point[:2]
            # print("update path point: ", point, "; updated position: ", self.model.site(f"wp{i}").pos)

        # Move the rest out of view
        for i in range(len(waypoints), 500):
            site_id = self.model.site(f"wp{i}").id
            self.data.site_xpos[site_id] = np.array([1000, 1000, 1000])


    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):

        # get ship state
        state = get_body_pose_2d(self.model, self.data, body_name='asv')

        # get vertices of all floes
        obs = self.get_floe_vertices()

        info = {
            'obs': obs, 
            'state': state
        }

        return info
