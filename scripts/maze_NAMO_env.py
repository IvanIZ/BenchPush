"""
An example script for running baseline policy for ship ice navigation
"""

import benchpush.environments
import gymnasium as gym
import numpy as np
from benchpush.baselines.maze_NAMO.ppo.policy import MazeNAMOPPO    
env = gym.make('maze-NAMO-v0')
env = env.unwrapped

# initialize planning policy
# planner_type = 'lattice'             # set planner type here. 'lattice' or 'predictive'
# policy = PlanningBasedPolicy(planner_type=planner_type)

# initialize RL policy
policy = MazeNAMOPPO()
# policy = ShipIceSAC()

total_dist_reward = 0
total_col_reward = 0
total_scaled_col_reward = 0
total_reward = 0

total_episodes = 500
for eps_idx in range(total_episodes):

    observation, info = env.reset()
    obstacles = info['obs']

    # start a new rollout
    step_c = 0
    while True:
        
        # call planning based policy
        # action = policy.act(observation=(observation / 255).astype(np.float64), ship_pos=info['state'], obstacles=obstacles, 
        #                     goal=env.goal,
        #                     conc=env.cfg.concentration, 
        #                     action_scale=env.max_yaw_rate_step)
        # env.update_path(policy.path)

        # call RL policy
        #action = policy.act(observation=observation, model_eps='90000')
        action = policy.act(observation=observation, model_eps='1030000')
        # print("action0: ", action)
        # print("step count: ", step_c)
        step_c += 1

        observation, reward, terminated, truncated, info = env.step(action)
        obstacles = info['obs']
        env.render()

        # haha print("reward: ", reward, "; dist increment reward: ", info['dist increment reward'], "; col reward: ", info['collision reward'], "; col reward scaled: ", info['scaled collision reward'])
        total_dist_reward += info['dist increment reward']
        total_col_reward += info['collision reward']
        total_scaled_col_reward += info['scaled collision reward']

        total_reward += reward
        print("total reward: ", total_reward)

        if terminated or truncated:
            # policy.reset()
            break

print(total_dist_reward, total_col_reward, total_scaled_col_reward)
