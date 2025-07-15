
import benchnpin.environments
import gymnasium as gym
import random
import numpy as np

env = gym.make('area-clearing-mujoco-v0', render_mode = "human", cfg=cfg)
env.reset()

terminated = truncated = False
num_epochs = 20
num_steps_per_epoch = 40000

for i in range(num_epochs):

    env.reset()
    for t in range(num_steps_per_epoch):

        w = 0

        goal_pos = [w]
        action = goal_pos

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
