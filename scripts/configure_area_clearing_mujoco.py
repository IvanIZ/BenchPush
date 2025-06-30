"""
An example script to show configurable parameters for Ship-Ice
"""
import benchnpin.environments
import gymnasium as gym
import random
import numpy as np


############### VVVVVVV Configurable Parameters for Ship-Ice VVVVVVV ####################
cfg = {
    "output_dir": "logs/",      # Specify directory for loggings
    "egocentric_obs": True,     # True egocentric observation, False for global observation
    "concentration": 0.1,       # Ice field concentration, options are 0.1, 0.2, 0.3, 0.4, 0.5
    "goal_y": 19,                # Initial distance from the goal line
    "render_scale": 30,         # Scalar applied to rendering window to fit the screen. Reducing this value makes rendering window smaller
}
############### ^^^^^^^ Configurable Parameters for Ship-Ice ^^^^^^^ ####################


env = gym.make('area-clearing-mujoco-v0', render_mode = "human", cfg=cfg)

terminated = truncated = False
num_epochs = 20
num_steps_per_epoch = 40000

for i in range(num_epochs):

    env.reset()
    for t in range(num_steps_per_epoch):

        v = 0.1
        w = 0

        goal_pos = [w]
        action = goal_pos

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
