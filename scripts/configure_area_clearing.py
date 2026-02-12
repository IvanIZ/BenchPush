"""
An example script to show configurable parameters for Area-Clearing
"""
import benchpush.environments
import gymnasium as gym


############### VVVVVVV Configurable Parameters for Area-Clearing VVVVVVV ####################
cfg = {
    "output_dir": "logs/",      # Specify directory for loggings
    "env": 'clear_env',         # Area structures. Options: 'clear_env_small', 'clear_env', walled_env', 'walled_env_with_columns'
    "num_obstacles": 15,        # Number of randomly positioned boxes to be removed
    "obstacle_size": 0.5,       # Size of each box
    "render_scale": 40,         # Scalar applied to rendering window to fit the screen. Reducing this value makes rendering window smaller
}
############### ^^^^^^^ Configurable Parameters for Area-Clearing ^^^^^^^ ####################


env = gym.make('area-clearing-v0', cfg=cfg)
env = env.unwrapped
env.activate_demo_mode()
env.reset()

terminated = truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(0)
    env.render()
