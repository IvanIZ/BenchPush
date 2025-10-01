"""
An example script to show configurable parameters for Maze
"""
import benchpush.environments
import gymnasium as gym


############### VVVVVVV Configurable Parameters for Maze VVVVVVV ####################
cfg = {
    "output_dir": "logs/",      # Specify directory for loggings
    "num_obstacles": 5,         # Number of cube obstacles to be randomly populated
    "obstacle_size": 0.5,       # Size of each cube
    "maze_version": 1,          # Maze version: 1 for U-Shape, 2 for Z-Shape
    "render_scale": 80,         # Scalar applied to rendering window to fit the screen. Reducing this value makes rendering window smaller
}
############### ^^^^^^^ Configurable Parameters for Maze ^^^^^^^ ####################


env = gym.make('maze-NAMO-v0', cfg=cfg)
env.reset()

terminated = truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(0)
    env.render()
