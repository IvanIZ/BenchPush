"""
An example script to show configurable parameters for Ship-Ice
"""
import benchnpin.environments
import gymnasium as gym
import random

env = gym.make('box-delivery-mujoco-v0', render_mode = "human", frame_skip=2)
env.reset()

terminated = truncated = False
num_epochs = 20
num_steps_per_epoch = 2

for i in range(num_epochs):

    env.reset()
    for t in range(num_steps_per_epoch):

        x = random.uniform(0.5, 1.0)
        y = random.uniform(0.5, 2.3)

        goal_pos = [0]
        action = goal_pos

        observation, reward, terminated, truncated, info = env.step(action)
