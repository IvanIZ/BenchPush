"""
An example script for running box pushing or ship ice environment
"""

import benchpush.environments
import gymnasium as gym
import numpy as np
import pickle

env = gym.make('maze-NAMO-v0')
env = env.unwrapped
env.reset()

teleop_paths = []

# load previous demos
with open('maze_metric_data_p1-2.pkl', 'rb') as file:
    pickle_dict = pickle.load(file)
actions = pickle_dict['actions']
teleop_paths.append(actions)

print("Path 1 -------------------")
print("efficiency_scores: ", pickle_dict['efficiency_scores'])
print("effort_scores: ", pickle_dict['effort_scores'])
print("rewards: ", pickle_dict['rewards'])

# load previous demos
with open('maze_metric_data_p2-1.pkl', 'rb') as file:
    pickle_dict = pickle.load(file)
actions = pickle_dict['actions']
teleop_paths.append(actions)

print("Path 2 -------------------")
print("efficiency_scores: ", pickle_dict['efficiency_scores'])
print("effort_scores: ", pickle_dict['effort_scores'])
print("rewards: ", pickle_dict['rewards'])

# load previous demos
with open('maze_metric_data_p3-0.pkl', 'rb') as file:
    pickle_dict = pickle.load(file)
actions = pickle_dict['actions']
teleop_paths.append(actions)

print("Path 3 -------------------")
print("efficiency_scores: ", pickle_dict['efficiency_scores'])
print("effort_scores: ", pickle_dict['effort_scores'])
print("rewards: ", pickle_dict['rewards'])

env.renderer.add_teleop_paths(teleop_paths)

env.render()
