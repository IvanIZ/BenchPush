"""
An example script for running baseline policy for ship ice navigation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.maze_NAMO.ppo.policy import MazeNAMOPPO    
env = gym.make('maze-NAMO-mujoco-v0', render_mode = "human")
env = env.unwrapped

# initialize RL policy
policy = MazeNAMOPPO(model_path='models/maze')
# policy = ShipIceSAC()

total_dist_reward = 0
total_col_reward = 0
total_scaled_col_reward = 0
total_reward = 0

total_episodes = 500
for eps_idx in range(total_episodes):

    observation, info = env.reset()

    # start a new rollout
    step_c = 0
    while True:

        # call RL policy
        angular_v = policy.act(observation=observation)
        print("step count: ", step_c)
        step_c += 1
        
        angular_v = angular_v[0]
        action = np.array([0.1, angular_v])
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        total_reward += reward
        print("total reward: ", total_reward)

        if terminated or truncated:
            # policy.reset()
            break

print(total_dist_reward, total_col_reward, total_scaled_col_reward)
