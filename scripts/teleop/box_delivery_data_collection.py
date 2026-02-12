"""
A simple script to run a teleoperation pipeline for demonstration dataset collection on box delivery environments
'A': large left turn; 'D' large right turn
'Z': small left turn; 'C' small right turn
'W': start moving
'X': stop turning (note: this does not stop linear motion)
'esc': exit teleoperation
"""
# TODO record high and low dimenstion states
import random

import benchpush.environments
import gymnasium as gym
import numpy as np
import pickle
from pynput import keyboard
from os.path import dirname

env = gym.make('box-delivery-v0')
env = env.unwrapped
env.cfg.teleop_mode = True
env.cfg.inactivity_cutoff_sam = 10000 # set to a large number to avoid inactivity cutoff
env.cfg.inactivity_cutoff = 10000 # set to a large number to avoid inactivity cutoff
env.reset()

observations = []
actions = []                # this is actually the states (i.e. 3 dof pose)
rewards = []
terminals = []              # This is true when episodes end due to termination conditions such as falling over.
timeouts = []               # This is true when episodes end due to reaching the maximum episode length

FORWARD = 0
STOP_TURNING = 1
LEFT = 2
RIGHT = 3
STOP = 4
BACKWARD = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7

command = STOP
manual_stop = False

def on_press(key):
    global command
    try:
        if key.char == 'w':  # Move up
            command = FORWARD
        elif key.char == 'x':  # Move down
            command = BACKWARD
        elif key.char == 'a':  # Move left
            command = LEFT
        elif key.char == 'd':  # Move right
            command = RIGHT
        elif key.char == 't':  # Stop moving
            command = STOP
        elif key.char == 'r':  # Stop turning
            command = STOP_TURNING
        elif key.char == 'z':  # Move left slowly
            command = SMALL_LEFT
        elif key.char == 'c':  # Move right slowly
            command = SMALL_RIGHT
    except AttributeError:
        pass


def on_release(key):
    global action, manual_stop
    if key == keyboard.Key.esc:  # Stop teleoperation when ESC is pressed
        manual_stop = True
        return False


def record_transition(observation, state, reward, terminal, timeout):
    observations.append(observation)
    actions.append(state)
    rewards.append(reward)
    terminals.append(terminal)
    timeouts.append(timeout)


def collect_demos():

    path_length = 0
    step_size = 0.1

    observation, info = env.reset()
    record_transition(observation, [info['state'][0], info['state'][1]], 0, False, False)
    prev_state = [info['state'][0], info['state'][1]]

    terminated = False
    truncated = False
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            t = 0
            transition_count = 1        # start from 1 as we recorded the reset step
            while listener.running:  # While the listener is active
                global command
                print("command: ", command, "; step: ", t, \
                    "; num completed: ", info['cumulative_cubes'],  end="\r")
                observation, reward, terminated, truncated, info = env.step(command)

                # command = OTHER
                if t % 5 == 0:
                    env.render()

                if (((info['state'][0] - prev_state[0])**2 + (info['state'][1] - prev_state[1])**2)**(0.5) >= step_size) or terminated or truncated:
                    record_transition(observation, [info['state'][0], info['state'][1]], reward, terminated, truncated)
                    prev_state = [info['state'][0], info['state'][1]]
                    transition_count += 1

                if terminated or truncated:
                    print("\nterminated: ", terminated, "; truncated: ", truncated)
                    path_length = transition_count
                    break

                t += 1


        except KeyboardInterrupt:
            print("Exiting teleoperation.")
        finally:
            env.close()
    
    # don't save the demo if this trial is truncated
    if truncated:
        print("\n Demo truncated. Ignored")
        return

    # don't save the demo if this trial is truncated
    global manual_stop
    if manual_stop:
        print("\nDemo manually stopped. Ignored")
        return
    ''' 
    global observations, actions, rewards, terminals, timeouts
    observations = np.array(observations).astype(np.float32)
    actions = np.array(actions).astype(np.float32)
    rewards = np.array(rewards).astype(np.float32)
    terminals = np.array(terminals)
    timeouts = np.array(timeouts)
    path_lengths = np.array([path_length])

    print("observation shape: ", observations.shape)
    print("actions shape: ", actions.shape)
    print("rewards shape: ", rewards.shape)
    print("terminals shape: ", terminals.shape)
    print("timeouts shape: ", timeouts.shape)
    print("current path length: ", path_length)


    try:
        # load previous demos
        with open('delivery_demo.pkl', 'rb') as file:
            pickle_dict = pickle.load(file)

        with open('delivery_demo_info.pkl', 'rb') as f:
            pickle_dict_info = pickle.load(f)
        
        # append current demonstration data
        pickle_dict['observations'] = np.concatenate((pickle_dict['observations'], observations))
        pickle_dict['actions'] = np.concatenate((pickle_dict['actions'], actions))
        pickle_dict['rewards'] = np.concatenate((pickle_dict['rewards'], rewards))
        pickle_dict['terminals'] = np.concatenate((pickle_dict['terminals'], terminals))
        pickle_dict['timeouts'] = np.concatenate((pickle_dict['timeouts'], timeouts))

        # append current meta-info data
        pickle_dict_info['path_lengths'] = np.concatenate((pickle_dict_info['path_lengths'], path_lengths))
        pickle_dict_info['demo_count'] = pickle_dict_info['demo_count'] + 1

    except:
        # if delivery_demo file not exist, create one with current demos
        pickle_dict = {
            'observations': observations, 
            'actions': actions, 
            'rewards': rewards, 
            'terminals': terminals,
            'timeouts': timeouts
        }

        pickle_dict_info = {
            'path_lengths': path_lengths,
            'demo_count': 1
        }

    print("Total Demonstration Data ======== \n")
    print("observation shape: ", pickle_dict['observations'].shape)
    print("actions shape: ", pickle_dict['actions'].shape)
    print("rewards shape: ", pickle_dict['rewards'].shape)
    print("terminals shape: ", pickle_dict['terminals'].shape)
    print("timeouts shape: ", pickle_dict['timeouts'].shape)

    print("max path lengths: ", np.max(pickle_dict_info['path_lengths']), "; min path length: ", np.min(pickle_dict_info['path_lengths']), "; average path length: ", np.mean(pickle_dict_info['path_lengths']))
    print("Total number of demos: ", pickle_dict_info['demo_count'])

    
    # save demo data
    with open('delivery_demo.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)

    # save demo info data
    with open('delivery_demo_info.pkl', 'wb') as f:
        pickle.dump(pickle_dict_info, f)
    '''


if __name__ == "__main__":
    collect_demos()
