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

env = gym.make('area-clearing-mujoco-v0', render_mode =render_mode, disable_env_checker=True)
env = env.unwrapped
env.reset()

terminated = truncated = False
num_epochs = 20
num_steps_per_epoch = 40000

for i in range(num_epochs):

    env.reset()
    for t in range(num_steps_per_epoch):

        w = 0

        goal_pos = [w]
        action = goal_pos

        observation, reward, terminated, truncated, info = env.step(action)
        frame = env.mujoco_renderer.render(render_mode='rgb_array')
        Image.fromarray(frame).save('snap_shot_' + str(t) + '.png')

        if terminated or truncated:
            break
