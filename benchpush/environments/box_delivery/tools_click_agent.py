import cv2
import numpy as np
import matplotlib.pyplot as plt
import benchpush.environments
import gymnasium as gym
import numpy as np
import pickle
from os.path import dirname
# from pynput import keyboard


class ClickAgent:
    def __init__(self, env):
        self.env = env
        self.window_name = 'window'
        self.reward_img_height = 12
        self.selected_action = None
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_callback)
        self.key_pressed = None
        self.map_scale = env.local_map_pixel_width
        plt.ion()
        plt.show()

    def mouse_callback(self, event):
        if event.button == 1:  # Left mouse button
            # self.selected_action = (max(0, int(event.ydata) - self.reward_img_height), int(event.xdata))
            self.selected_action = (int(event.ydata), int(event.xdata))
    
    def key_callback(self, event):
        self.key_pressed = event.key

    def update_display(self, state, last_reward, last_ministeps):
        state_img = state[:,:,0]
        self.ax.clear()
        self.ax.imshow(state_img)
        plt.draw()
        plt.pause(0.001)

    def run(self):
        state, _ = self.env.reset()
        last_reward = 0
        last_ministeps = 0

        done = False
        force_reset_env = False
        while True:
            self.update_display(state, last_reward, last_ministeps)

            # Read keyboard input
            if self.key_pressed is not None:
                if self.key_pressed == ' ':
                    force_reset_env = True
                elif self.key_pressed == 'q':
                    break

            if self.selected_action is not None:
                action = self.selected_action[0] * self.map_scale + self.selected_action[1]
                state, reward, done, _, info = self.env.step(action)
                last_reward = reward
                # last_ministeps = info['ministeps']
                self.selected_action = None
            else:
                #p.stepSimulation()  # Uncomment to make pybullet window interactive
                pass

            if done or force_reset_env:
                state, _ = self.env.reset()
                done = False
                force_reset_env = False
                last_reward = 0
                last_ministeps = 0
        plt.close()

def main():
    env = gym.make('box-delivery-v0')
    env = env.unwrapped
    env.cfg.render.show_obs = True # Show observations
    agent = ClickAgent(env)
    agent.run()
    env.close()

main()
