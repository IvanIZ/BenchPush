"""
A simple script for running ship planners to collect demonstration trajectory dataset for ship-ice navigation
"""

import benchpush.environments
import gymnasium as gym
import numpy as np

import numpy as np
from benchpush.baselines.ship_ice_nav.lattice import LatticePlanner
from benchpush.common.controller.dp import DP
import pickle

env = gym.make('ship-ice-v0')
env = env.unwrapped
lattice_planner = LatticePlanner()

observations = []
actions = []                # this is actually the states (i.e. 3 dof pose)
rewards = []
terminals = []              # This is true when episodes end due to termination conditions such as falling over.
timeouts = []               # This is true when episodes end due to reaching the maximum episode length


def record_transition(observation, state, reward, terminal, timeout):
    observations.append(observation)
    actions.append(state)
    rewards.append(reward)
    terminals.append(terminal)
    timeouts.append(timeout)

total_episodes = 5
episode_count = 0

path_lengths = []

while episode_count <= total_episodes:

    observation, info = env.reset()
    obstacles = info['obs']
    record_transition(observation, [info['state'][0], info['state'][1]], 0, False, False)

    # open-loop planning
    path = lattice_planner.plan(ship_pos=info['state'], goal=[0, lattice_planner.cfg.goal_y], obs=obstacles)

    env.update_path(path, scatter=False)

    # setup dp controller to track the planned path
    cx = path.T[0]
    cy = path.T[1]
    ch = path.T[2]
    dp = DP(x=info['state'][0], y=info['state'][1], yaw=info['state'][2],
            cx=cx, cy=cy, ch=ch, **lattice_planner.cfg.controller)
    state = dp.state

    # start a new rollout
    max_steps = 500
    for i in range(max_steps):

        print("data collection step: ", episode_count, " / ", total_episodes, end="\r")
        # print(episode_count, i, "; position: ", info['state'])

        # call ideal controller to get angular velocity control
        omega, _ = dp.ideal_control(info['state'][0], info['state'][1], info['state'][2])
        observation, reward, terminated, truncated, info = env.step(omega)
        
        # update setpoint
        x_s, y_s, h_s = dp.get_setpoint()
        dp.setpoint = np.asarray([x_s, y_s, np.unwrap([state.yaw, h_s])[1]])

        if i % 5 == 0:
            env.render()

        record_transition(observation, [info['state'][0], info['state'][1]], reward, terminated, truncated)
        if terminated or truncated or i + 1 == max_steps:
            path_lengths.append(i)
            break
    
    episode_count += 1
    

observations = np.array(observations).astype(np.float32)
actions = np.array(actions).astype(np.float32)
rewards = np.array(rewards).astype(np.float32)
terminals = np.array(terminals)
timeouts = np.array(timeouts)
path_lengths = np.array(path_lengths)

print("observation shape: ", observations.shape)
print("actions shape: ", actions.shape)
print("rewards shape: ", rewards.shape)
print("terminals shape: ", terminals.shape)
print("timeouts shape: ", timeouts.shape)
print("max path length: ", np.max(path_lengths), "; min path length: ", np.min(path_lengths), "; average path length: ", np.mean(path_lengths))

pickle_dict = {
    'observations': observations, 
    'actions': actions, 
    'rewards': rewards, 
    'terminals': terminals,
    'timeouts': timeouts
}
with open('ship_ice_demo.pkl', 'wb') as f:
    pickle.dump(pickle_dict, f)
