"""
An example script to show configurable parameters for Ship-Ice
"""
import benchpush.environments
import gymnasium as gym


############### VVVVVVV Configurable Parameters for Ship-Ice VVVVVVV ####################
cfg = {
    "output_dir": "logs/",      # Specify directory for loggings
    "egocentric_obs": True,     # True egocentric observation, False for global observation
    "concentration": 0.1,       # Ice field concentration, options are 0.1, 0.2, 0.3, 0.4, 0.5
    "goal_y": 19,                # Initial distance from the goal line
    "render_scale": 30,         # Scalar applied to rendering window to fit the screen. Reducing this value makes rendering window smaller
}
############### ^^^^^^^ Configurable Parameters for Ship-Ice ^^^^^^^ ####################


env = gym.make('ship-ice-v0', cfg=cfg)
env.reset()

terminated = truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(0)
    env.render()
