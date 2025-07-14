"""
An example script for running baseline policy for ship ice navigation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy
from benchnpin.baselines.ship_ice_nav.ppo.policy import ShipIcePPO
from benchnpin.baselines.ship_ice_nav.sac.policy import ShipIceSAC 

# Function to compute the PD control signal
def compute_pd_control(current_yaw, target_yaw, previous_error, dt, kp, kd):
    """
    Computes the PD control signal for yaw control.

    Args:
        current_yaw (float): The current yaw angle of the ASV.
        target_yaw (float): The desired yaw angle.
        previous_error (float): The error from the previous time step.
        dt (float): The time step.
        kp (float): Proportional gain.
        kd (float): Derivative gain.

    Returns:
        float: The control signal (rudder torque).
    """
    error = target_yaw - current_yaw
    # Make sure error is within -pi to pi
    error = np.arctan2(np.sin(error), np.cos(error))
    
    derivative = (error - previous_error) / dt
    control_signal = kp * error + kd * derivative
    return control_signal, error  # Return both control signal and current error


env = gym.make('ship-ice-mujoco-v0', render_mode = "human")
env = env.unwrapped

# PD controller gains
kp = 1.0  # Proportional gain
kd = 0.5  # Derivative gain

# Simulation parameters
target_yaw = np.pi / 2  # Target yaw angle (0 for alignment with x-axis)
forward_force = 15.0  # Constant force for forward motion
dt = env.model.opt.timestep  # Time step from the model

# Initialize previous error
previous_error = 0.0

# initialize planning policy
planner_type = 'lattice'             # set planner type here. 'lattice' or 'predictive'
policy = PlanningBasedPolicy(planner_type=planner_type, planner_config='lattice_config_mujoco.yaml')

# initialize RL policy
# policy = ShipIcePPO()
# policy = ShipIceSAC()

total_dist_reward = 0
total_col_reward = 0
total_scaled_col_reward = 0

total_episodes = 500
for eps_idx in range(total_episodes):

    observation, info = env.reset()
    obstacles = info['obs']
    state = info['state']
    ship_angular_vel = info['ship_angular_vel']
    ship_linear_vel = info['ship_linear_vel']
    curr_speed = np.linalg.norm(ship_linear_vel)
    current_yaw = state[2]

    # start a new rollout
    step_idx = 0
    while True:
        step_idx += 1
        
        # call planning based policy
        omega, v = policy.act(observation=(observation / 255).astype(np.float64), ship_pos=info['state'], obstacles=obstacles, 
                            goal=env.goal,
                            conc=env.cfg.concentration, 
                            action_scale=env.max_yaw_rate_step, 
                            speed=curr_speed)
        omega = omega * env.max_yaw_rate_step           # unnormalize to real-world scale
        env.update_path(policy.path)

        # call RL policy
        # action = policy.act(observation=observation, model_eps='470000')
        # action = policy.act(observation=observation, model_eps='130000')

        # # Compute PD control signal
        # rudder_control, previous_error = compute_pd_control(current_yaw, target_yaw, previous_error, dt, kp, kd)

        # forward_force = 40050000.0      # 15N forward force

        action = [v, omega]
        # print(action)
        
        observation, reward, terminated, truncated, info = env.step(action)
        obstacles = info['obs']
        if step_idx % 5 == 0:
            env.render()

        state = info['state']
        current_yaw = state[2]

        ship_angular_vel = info['ship_angular_vel']
        ship_linear_vel = info['ship_linear_vel']
        curr_speed = np.linalg.norm(ship_linear_vel)
        # print("desired angular vel: ", omega, "; actual angular vel: ", ship_angular_vel)
        # print("desired linear vel: ", v, "; actual linear vel: ", ship_linear_vel)

        # print("reward: ", reward, "; dist reward: ", info['dist reward'], "; col reward: ", info['collision reward'], "; col reward scaled: ", info['scaled collision reward'])
        # total_dist_reward += info['dist reward']
        # total_col_reward += info['collision reward']
        # total_scaled_col_reward += info['scaled collision reward']

        if terminated or truncated:
            policy.reset()
            break

print(total_dist_reward, total_col_reward, total_scaled_col_reward)
