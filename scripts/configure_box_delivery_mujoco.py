"""
An example script to show configurable parameters for Ship-Ice
"""
import benchnpin.environments
import gymnasium as gym
import random


############### VVVVVVV Configurable Parameters for Ship-Ice VVVVVVV ####################
cfg = {
    "output_dir": "logs/",      # Specify directory for loggings
    "egocentric_obs": True,     # True egocentric observation, False for global observation
    "concentration": 0.1,       # Ice field concentration, options are 0.1, 0.2, 0.3, 0.4, 0.5
    "goal_y": 19,                # Initial distance from the goal line
    "render_scale": 30,         # Scalar applied to rendering window to fit the screen. Reducing this value makes rendering window smaller
}
############### ^^^^^^^ Configurable Parameters for Ship-Ice ^^^^^^^ ####################


env = gym.make('box-delivery-mujoco-v0', render_mode = "human", cfg=cfg)
env.reset()

terminated = truncated = False
num_epochs = 20
num_steps_per_epoch = 2

for i in range(num_epochs):

    env.reset()
    for t in range(num_steps_per_epoch):

        x = random.uniform(0.5, 1.0)
        y = random.uniform(0.5, 2.3)

        goal_pos = [x, y]
        action = goal_pos

        observation, reward, terminated, truncated, info = env.step(action)
