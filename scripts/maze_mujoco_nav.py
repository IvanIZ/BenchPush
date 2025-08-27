"""
An example script for running baseline policy for ship ice navigation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.maze_mujoco.ppo.policy import MazeMujocoPPO
from benchnpin.baselines.maze_mujoco.sac.policy import MazeMujocoSAC
env = gym.make('maze-NAMO-mujoco-v0', render_mode = "human")
env = env.unwrapped

# initialize RL policy
# policy = MazeMujocoPPO(model_path='models/maze')
# policy = MazeMujocoPPO()

policy = MazeMujocoSAC()

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
        # angular_v = policy.act(observation=observation)
        # angular_v = policy.act(observation=observation, model_eps='1400000')
        angular_v = policy.act(observation=observation, model_eps='200000')
        # angular_v = policy.act(observation=observation, model_eps='beta3000')
        # print("step count: ", step_c)
        # print("policy output: ", angular_v)
        step_c += 1
        
        # action = np.array([0.1, angular_v])
        observation, reward, terminated, truncated, info = env.step(angular_v)
        env.render()

        # print("reward: ", reward, "; dist increment reward: ", info['dist increment reward'], "; col reward: ", info['collision reward'], "; col reward scaled: ", info['scaled collision reward'])
        print("reward: ", reward, "; dist increment reward: ", info['dist increment reward'], "; col reward scaled: ", info['scaled collision reward'])
        total_reward += reward
        # print("total reward: ", total_reward)

        if terminated or truncated:
            # policy.reset()
            break

print(total_dist_reward, total_col_reward, total_scaled_col_reward)
