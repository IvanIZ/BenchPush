"""
An example script to show configurable parameters for Ship-Ice
"""
import benchnpin.environments
import gymnasium as gym


############### VVVVVVV Configurable Parameters for Ship-Ice VVVVVVV ####################
cfg = {
    "output_dir": "logs/",      # Specify directory for loggings
    "egocentric_obs": True,     # True egocentric observation, False for global observation
    "concentration": 0.2,       # Ice field concentration, options are 0.1, 0.2, 0.3, 0.4, 0.5
    "goal_y": 19,                # Initial distance from the goal line
    "render_scale": 30,         # Scalar applied to rendering window to fit the screen. Reducing this value makes rendering window smaller
}
############### ^^^^^^^ Configurable Parameters for Ship-Ice ^^^^^^^ ####################


env = gym.make('ship-ice-mujoco-v0', render_mode = "human", cfg=cfg)
env.reset()

terminated = truncated = False
while True:
    # forward_force = 40050000.0      # 15N forward force
    # rudder_control = 0.5     # no turning
    # action = [forward_force, rudder_control]

    forward_speed = 80.0      # 2 m/s
    angular_speed = 0.0     # no turning
    action = [forward_speed, angular_speed]

    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
