import benchnpin.environments
import gymnasium as gym
import random
from PIL import Image

# render 
render= True
if render == True:
    render_mode = "human"
else:
    render_mode = None

env = gym.make('box-delivery-mujoco-v0', render_mode=render_mode, frame_skip=5, disable_env_checker=False, render_width=480, render_height=480)
# 960*4
env = env.unwrapped
env.reset()

terminated = truncated = False
num_epochs = 20
num_steps_per_epoch = 20000

for i in range(num_epochs):

    env.reset()
    action = 96*48 + 48
    observation, reward, terminated, truncated, info = env.step(action)
    # input()

    for t in range(num_steps_per_epoch):

        action = 0

        observation, reward, terminated, truncated, info = env.step(action)
        # input(reward)
        # frame = env.mujoco_renderer.render(render_mode='rgb_array')
        # Image.fromarray(frame).save('snap_shot_' + str(t) + '.png')

        if terminated or truncated:
            break
