
import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy
from benchnpin.baselines.ship_ice_nav.ppo.policy import ShipIcePPO
from benchnpin.baselines.ship_ice_nav.sac.policy import ShipIceSAC 

env = gym.make('ship-ice-mujoco-v0', render_mode = "human")
env = env.unwrapped

# select planner type
planner_type = 'lattice'             # set planner type here. 'lattice' or 'predictive'
# planner_type = 'straight'            # use 'straight' planner to test path tracking
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
        omega, v = policy.act(observation=(observation / 255).astype(np.float64), ship_pos=state, obstacles=obstacles, 
                            goal=env.goal,
                            conc=env.cfg.concentration, 
                            action_scale=env.max_yaw_rate_step, 
                            speed=curr_speed)
        env.update_path(policy.path)

        # call RL policy
        # action = policy.act(observation=observation, model_eps='470000')
        # action = policy.act(observation=observation, model_eps='130000')

        v_x, v_y = v * np.sin(current_yaw), -v * np.cos(current_yaw)

        action = [v_x,v_y, omega]
        
        observation, reward, terminated, truncated, info = env.step(action)
        obstacles = info['obs']
        if step_idx % 10 == 0:
            env.render()

        state = info['state']
        current_yaw = state[2]

        ship_angular_vel = info['ship_angular_vel']
        ship_linear_vel = info['ship_linear_vel']
        curr_speed = np.linalg.norm(ship_linear_vel)

        if terminated or truncated:
            policy.reset()
            break

print(total_dist_reward, total_col_reward, total_scaled_col_reward)
