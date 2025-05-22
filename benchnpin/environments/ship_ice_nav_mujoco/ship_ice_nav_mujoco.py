from typing import Dict, Union

import numpy as np

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


    def apply_fluid_drag_to_body(self, model, data, body_name, joint_prefix, beta=3.0, Cd=0.8, area=0.02, angular_beta=0.2):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # x and y dof indices
        jnt_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_prefix + "_x")
        jnt_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_prefix + "_y")
        jnt_yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_prefix + "_yaw")

        dof_x = model.jnt_dofadr[jnt_x]
        dof_y = model.jnt_dofadr[jnt_y]
        dof_yaw = model.jnt_dofadr[jnt_yaw]

        vx = data.qvel[dof_x]
        vy = data.qvel[dof_y]
        v = np.array([vx, vy])
        v_mag = np.linalg.norm(v)
        v_dir = v / v_mag if v_mag > 1e-5 else np.zeros(2)

        F_linear = -beta * v
        F_quad = -0.5 * 1000 * Cd * area * v_mag**2 * v_dir
        F_total = F_linear + F_quad

        # Apply force in x and y directions
        total_force = np.array([F_total[0], F_total[1], 0]).astype(np.float64)
        point = np.zeros((3, 1)).astype(np.float64)

        # Angular velocity (yaw only)
        total_torque = np.zeros((3, 1)).astype(np.float64)
        omega_z = data.qvel[dof_yaw]
        total_torque = np.array([0, 0, -angular_beta * omega_z])
        total_torque = total_torque.reshape((3, -1)).astype(np.float64)

        # mass = model.body_mass[body_id]
        # print(mass)

        mujoco.mj_applyFT(model, data, force=total_force, torque=total_torque, body=body_id, point=point, qfrc_target=data.qfrc_applied)


    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulation.
        """
        self.data.ctrl[:] = ctrl

        # Apply drag to ASV and ice floes
        self.apply_fluid_drag_to_body(self.model, self.data, "asv", "free_joint", beta=5.0, area=0.04, angular_beta=5.0)

        # Apply drag to all ice floes
        for n in range(160):
            name = f"ice_{n}"
            self.apply_fluid_drag_to_body(self.model, self.data, name, name, beta=5.0, Cd=0.8, angular_beta=5.0)
        
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
