"""
An example script to show configurable parameters for Box-Delivery
"""
import benchpush.environments
import gymnasium as gym


############### VVVVVVV Configurable Parameters for Box-Delivery VVVVVVV ####################
cfg = {
    "output_dir": "logs/",      # Specify directory for loggings
    'render': {
        'show': True,           # if true display the environment
        'show_obs': False,       # if true show observation
    },
    'agent': {
            'action_type': 'position', # 'position', 'heading', 'velocity'
            'random_start': False
            },
    'boxes': {
        'num_boxes_small': 10, # number of boxes to include in the small environments
        'num_boxes_large': 20, # number of boxes to include in the large environments
    },
    'env': {
        'obstacle_config': 'small_empty', # options are small_empty, small_columns, large_columns, large_divider
    },
    'train': {
        'job_type': 'sam', 
    },
    "render_scale": 50,         # Scalar applied to rendering window to fit the screen. Reducing this value makes rendering window smaller
}
############### ^^^^^^^ Configurable Parameters for Box-Delivery ^^^^^^^ ####################


env = gym.make('box-delivery-v0', cfg=cfg)
env.reset()

terminated = truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(0)
    env.render()
