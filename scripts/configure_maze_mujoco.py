import benchnpin.environments
import gymnasium as gym
import random
import numpy as np

env = gym.make('maze-NAMO-mujoco-v0', render_mode = "human", cfg=cfg, disable_env_checker=True)

terminated = truncated = False
num_epochs = 10
num_steps_per_epoch = 40000

for i in range(num_epochs):

    _, info = env.reset()

    for t in range(num_steps_per_epoch):

        w = 0
        action = w

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(info['state'])

        if terminated or truncated:
            break
