"""
An example script for running box pushing or ship ice environment
"""

import benchpush.environments
import gymnasium as gym
import numpy as np
import pickle

import os

env = gym.make('area-clearing-v0')
env = env.unwrapped
env.reset()

teleop_paths = []

# load previous demos
with open('area_clearing_metric_data-1.pkl', 'rb') as file:
    pickle_dict = pickle.load(file)
actions = pickle_dict['actions']
teleop_paths.append(actions)

print("Path 1 -------------------")
print("success_rate: ", pickle_dict['success_rate'])
print("efficiency_scores: ", pickle_dict['efficiency_scores'])
print("effort_scores: ", pickle_dict['effort_scores'])
print("rewards: ", pickle_dict['rewards'])

# load previous demos
with open('area_clearing_metric_data-2.pkl', 'rb') as file:
    pickle_dict = pickle.load(file)
actions = pickle_dict['actions']
teleop_paths.append(actions)

print("Path 2 -------------------")
print("success_rate: ", pickle_dict['success_rate'])
print("efficiency_scores: ", pickle_dict['efficiency_scores'])
print("effort_scores: ", pickle_dict['effort_scores'])
print("rewards: ", pickle_dict['rewards'])

# load previous demos
with open('area_clearing_metric_data-3.pkl', 'rb') as file:
    pickle_dict = pickle.load(file)
actions = pickle_dict['actions']
teleop_paths.append(actions)

print("Path 3 -------------------")
print("success_rate: ", pickle_dict['success_rate'])
print("efficiency_scores: ", pickle_dict['efficiency_scores'])
print("effort_scores: ", pickle_dict['effort_scores'])
print("rewards: ", pickle_dict['rewards'])

env.renderer.add_teleop_paths(teleop_paths)
env.renderer.teleop_path_thickness = 3

path = os.path.join('area_clearing_metrics.png')
if(env.renderer):
    env.renderer.render(save=True, path=path)
