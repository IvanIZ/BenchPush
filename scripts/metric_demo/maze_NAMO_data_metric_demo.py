"""
A simple script to run a teleoperation pipeline for demonstration dataset collection on box pushing environments
'A': large left turn; 'D' large right turn
'Z': small left turn; 'C' small right turn
'W': start moving
'X': stop turning (note: this does not stop linear motion)
'esc': exit teleoperation
"""

"""
NOTE (Feb 27) This is a temporary script to collect teleoperation path. 
This script save the teleoperated path into the pickle file under the 'action' key, along with the performance metrics for the paths.
This is intended for plotting
"""

import benchpush.environments
import gymnasium as gym
import numpy as np
import pickle
from pynput import keyboard
import time
from benchpush.common.metrics.maze_namo_metric import MazeNamoMetric

env = gym.make('maze-NAMO-v0')
env = env.unwrapped
metric = MazeNamoMetric(alg_name="PPO", robot_mass=env.cfg.robot.mass)

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
OTHER = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7
BACKWARD = 8

TELEOPT_FREQ = 20

angular_velocity = 0

command = STOP
manual_stop = False

def on_press(key):
    global command, angular_velocity
    try:
        if key.char == 'w':  # Move up
            command = FORWARD
        elif key.char == 'y':  # Stop turning
            command = STOP_TURNING
        elif key.char == 'x':  # Move down
            command = BACKWARD

        elif key.char == 'a':  # Move left
            command = LEFT
            angular_velocity += 0.1
            angular_velocity = np.clip(angular_velocity, a_max=1.0, a_min=-1.0)

        elif key.char == 'd':  # Move right
            command = RIGHT
            angular_velocity -= 0.1
            angular_velocity = np.clip(angular_velocity, a_max=1.0, a_min=-1.0)

        elif key.char == 'z':  # Move right
            command = SMALL_LEFT
            angular_velocity += 0.05
            angular_velocity = np.clip(angular_velocity, a_max=1.0, a_min=-1.0)

        elif key.char == 'c':  # Move right
            command = SMALL_RIGHT
            angular_velocity -= 0.05
            angular_velocity = np.clip(angular_velocity, a_max=1.0, a_min=-1.0)

    except AttributeError:
        pass


def on_release(key):
    global action, manual_stop
    if key == keyboard.Key.esc:  # Stop teleoperation when ESC is pressed
        manual_stop = True
        return False


def record_transition(observation, action, reward, terminal, timeout):
    observations.append(observation)
    actions.append(action)
    rewards.append(reward)
    terminals.append(terminal)
    timeouts.append(timeout)


def collect_demos():
    global angular_velocity
    path_length = 0
    total_reward = 0

    observation, info = env.reset()
    metric.reset(info)
    record_transition(0, [info['state'][0], info['state'][1]], 0, False, False)

    terminated = False
    truncated = False
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            t = 0
            transition_count = 1        # start from 1 as we recorded the reset step
            while listener.running:  # While the listener is active
                loop_start_time = time.time()

                global command
                _, reward, terminated, truncated, info = env.step(angular_velocity)
                metric.update(info=info, reward=reward, eps_complete=(terminated or truncated))
                record_transition(0, [info['state'][0], info['state'][1]], reward, terminated, truncated)

                print("increment reward: ", info['dist increment reward'], "; col reward scaled: ", info['scaled collision reward'])

                total_reward += reward

                if terminated or truncated:
                    print("\nterminated: ", terminated, "; truncated: ", truncated)
                    path_length = transition_count
                    break
                t += 1
                env.render()

                # loop frequency control for human operator
                loop_time = time.time() - loop_start_time
                if loop_time <= (1 / TELEOPT_FREQ):
                    time.sleep((1 / TELEOPT_FREQ) - loop_time)


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
        with open('maze_metric_data.pkl', 'rb') as file:
            pickle_dict = pickle.load(file)
        
        # append current demonstration data
        pickle_dict['observations'] = np.concatenate((pickle_dict['observations'], observations))
        pickle_dict['actions'] = np.concatenate((pickle_dict['actions'], actions))
        pickle_dict['rewards'] = np.concatenate((pickle_dict['rewards'], rewards))
        pickle_dict['terminals'] = np.concatenate((pickle_dict['terminals'], terminals))
        pickle_dict['timeouts'] = np.concatenate((pickle_dict['timeouts'], timeouts))

    except:
        # if pushing_demo file not exist, create one with current demos
        pickle_dict = {
            'observations': observations, 
            'actions': actions, 
            'rewards': rewards, 
            'terminals': terminals,
            'timeouts': timeouts, 
            'efficiency_scores': metric.efficiency_scores[0],
            'effort_scores': metric.effort_scores[0],
            'rewards': metric.rewards[0]
        }

    print("Total Demonstration Data ======== \n")
    print("observation shape: ", pickle_dict['observations'].shape)
    print("actions shape: ", pickle_dict['actions'].shape)
    print("rewards shape: ", pickle_dict['rewards'].shape)
    print("terminals shape: ", pickle_dict['terminals'].shape)
    print("timeouts shape: ", pickle_dict['timeouts'].shape)

    
    # save demo data
    with open('maze_metric_data.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)

    print(metric.efficiency_scores, metric.effort_scores, metric.rewards)

    
if __name__ == "__main__":
    collect_demos()
