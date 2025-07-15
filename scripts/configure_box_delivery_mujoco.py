import benchnpin.environments
import gymnasium as gym
import random

# render 
render= True
if render == True:
    render_mode = "human"
else:
    render_mode = None

env = gym.make('box-delivery-mujoco-v0', render_mode = render_mode, frame_skip=1,disable_env_checker=True)
env.reset()

terminated = truncated = False
num_epochs = 20
num_steps_per_epoch = 2

for i in range(num_epochs):

    env.reset()
    for t in range(num_steps_per_epoch):

        goal_pos = [0]
        action = goal_pos

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
