"""
An example script for running baseline policy for ship ice navigation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy
from benchnpin.baselines.ship_ice_nav.ppo.policy import ShipIcePPO
from benchnpin.baselines.ship_ice_nav.sac.policy import ShipIceSAC 

env = gym.make('ship-ice-mujoco-v0', render_mode = "human")
env = env.unwrapped

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

    # start a new rollout
    while True:
        
        # call planning based policy
        omega = policy.act(observation=(observation / 255).astype(np.float64), ship_pos=info['state'], obstacles=obstacles, 
                            goal=env.goal,
                            conc=env.cfg.concentration, 
                            action_scale=env.max_yaw_rate_step)
        # env.update_path(policy.path)

        # call RL policy
        # action = policy.act(observation=observation, model_eps='470000')
        # action = policy.act(observation=observation, model_eps='130000')

        action = [0.8, omega]
        print(action)
        
        observation, reward, terminated, truncated, info = env.step(action)
        obstacles = info['obs']
        env.render()

        # print("reward: ", reward, "; dist reward: ", info['dist reward'], "; col reward: ", info['collision reward'], "; col reward scaled: ", info['scaled collision reward'])
        # total_dist_reward += info['dist reward']
        # total_col_reward += info['collision reward']
        # total_scaled_col_reward += info['scaled collision reward']

        if terminated or truncated:
            policy.reset()
            break

print(total_dist_reward, total_col_reward, total_scaled_col_reward)
