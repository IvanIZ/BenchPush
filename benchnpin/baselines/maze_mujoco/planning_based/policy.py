#Benchnpin Environments Imports
import benchnpin.environments
import gymnasium as gym
#Non-learning Planners for Maze NAMO
from benchnpin.baselines.maze_mujoco.planning_based.RRT.rrt import RRTPlanner
#Base Policy Class
from benchnpin.baselines.base_class import BasePolicy
#Benchnpin Metrics 
from benchnpin.common.metrics.maze_namo_metric import MazeNamoMetric
#DP Controller
from benchnpin.common.controller.dp import DP

#data structures
from typing import List, Tuple
#testing imports will be removed later
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point

scale_factor = (2.845/10) # NOTE: this scales thresholds to be proportionately the same as in the 2d environment

MOVE_STEP_SIZE = 0.05 * scale_factor
TURN_STEP_SIZE = np.radians(15)

WAYPOINT_MOVING_THRESHOLD = 0.6 * scale_factor
WAYPOINT_TURNING_THRESHOLD = np.radians(10)
NOT_MOVING_THRESHOLD = 0.005 * scale_factor
NOT_TURNING_THRESHOLD = np.radians(0.05)
NONMOVEMENT_DIST_THRESHOLD = 0.05 * scale_factor
NONMOVEMENT_TURN_THRESHOLD = np.radians(0.05)
STEP_LIMIT = 4000

class PlanningBasedPolicy(BasePolicy):
    """
    A baseline policy for Maze-NAMO. 
    This policy is  either planning offline paths or online paths using a non-learning planner,
    then outputs actions to track the planned path.
    """

    def __init__(self, planner_type, cfg=None, planner_config=None) -> None:
        super().__init__()

        if planner_type not in ['RRT']: #only RRT is implemented for now
            raise Exception("Invalid planner type")
        self.planner_type = planner_type

        self.path = None

        self.RRTPlanner = RRTPlanner(config_file=None)
        
        self.cfg = cfg

        


    def plan_path(self, start, goal, observation, maze_vertices, obstacles=None, robot_radius = None):
        if self.planner_type == 'RRT':
           self.path = self.RRTPlanner.plan(start=start, goal=goal, maze_vertices=maze_vertices,
                                             movable_obstacles=obstacles, 
                                             robot_size= {'width': self.env.cfg.agent.width,
                                                          'length': self.env.cfg.agent.length},
                                             bounds_pad=0)
           self.path = np.array(self.path)
        #    self.path = self.prune_by_distance(self.path, min_dist=0.3/1.5)
           self.RRTPlanner.plot_env(self.path)
    
    def prune_by_distance(self, path, min_dist=0.5):
        """
        Only used for diffusion policy.
        Remove waypoints that are too close together
        """
        # Visualize pruning if needed
        # self.renderer.update_path(path.squeeze(0).cpu().numpy())
        # self.render()
        # input()
        # Always include start and end points
        # path = path / self.diffusion_policy.scale
        pruned = [path[0]]
        prev_point = path[0]
        final_point = path[-1]
    
        for i in range(1, path.shape[0] - 1):
            dist_from_prev = np.linalg.norm(path[i] - prev_point)
            dist_from_final = np.linalg.norm(path[i] - final_point)
            if dist_from_prev >= min_dist and dist_from_final >= min_dist:
                pruned.append(path[i])
                prev_point = path[i]

        pruned.append(path[-1]) # goal is always included
        pruned = np.stack(pruned, axis=0)
        # pruned = pruned[np.newaxis, :] #* self.diffusion_policy.scale
        return pruned

    def act(self, observation, **kwargs):
        robot_pos = kwargs.get('robot_pos', None)
        robot_radius = kwargs.get('robot_radius', None)
        goal = kwargs.get('goal', None)
        maze_vertices = kwargs.get('maze_vertices', None)
        obstacles = kwargs.get('obstacles', None)
        action_scale = kwargs.get('action_scale', 1.0)
        

        #plan path offline
        if self.path is None:
            self.plan_path(start=robot_pos[0:2], goal=goal, observation=observation, 
                           maze_vertices=maze_vertices, obstacles=obstacles,
                           robot_radius=robot_radius)
            
        #traverse the path using a DP controller
        # setup dp controller to track the planned path
        cx = self.path.T[0]
        cy = self.path.T[1]
        ch = np.zeros_like(cx)  # desired heading is a dummy value here
        self.dp = DP(x=robot_pos[0], y=robot_pos[1], yaw=robot_pos[2],
                cx=cx, cy=cy, ch=ch, **self.RRTPlanner.cfg.controller)
        self.dp_state = self.dp.state
        
        # call ideal controller to get angular velocity control
        omega, _ = self.dp.ideal_control(robot_pos[0], robot_pos[1], robot_pos[2])

        # update setpoint
        x_s, y_s, h_s = self.dp.get_setpoint()
        self.dp.setpoint = np.asarray([x_s, y_s, np.unwrap([self.dp_state.yaw, h_s])[1]])

        return omega / action_scale
        

    def evaluate(self,  num_eps: int, model_eps: str ='latest') -> Tuple[List[float], List[float], List[float], str]:
        env = gym.make('maze-NAMO-mujoco-v0', cfg=self.cfg, render_mode='human')
        self.env = env.unwrapped

        #algorithm name for metrics
        if self.planner_type == 'RRT':
            algo_name = 'RRT'
        
        #metric instance for evaluation
        metric = MazeNamoMetric(alg_name=algo_name, robot_mass= 1.07)
        
        #episodes
        for eps_idx in range(num_eps):
            print("Planning Based Progress: ", eps_idx, " / ", num_eps, " episodes")
            observation, info = self.env.reset()
            metric.reset(info)
            obstacles = info['obs']
            done = truncated = False

            #parameters that do not change per step
            goal = self.env.cfg.env.goal_position
            maze_vertices = self.env.maze_walls
            robot_radius = self.env.cfg.agent.robot_r

            while not (done or truncated):
                #paramters that change per step
                robot_pos = info['state']
                obstacles = info['obs']
                #compute the next action (angular velocity)

                room_length = self.env.cfg.env1.length
                room_width = self.env.cfg.env1.width

                maze_walls = [[(0,0),(room_width,0)] , [(0,0),(0,room_length)],
                    [(room_width,0),(room_width, room_length)], 
                    [(0,room_length),(room_width,room_length)],
                    [(room_width/2,0),(room_width/2,room_length - room_length/4.5)]]

                robot_position = (robot_pos[0], robot_pos[1])
                robot_heading = robot_pos[2]


                action = self.act(observation=(observation/255.0).astype(np.float64),
                                robot_pos=robot_pos, 
                                robot_radius=robot_radius,
                                goal=self.env.cfg.env.goal_position, 
                                maze_vertices= maze_walls,
                                obstacles= obstacles,
                                action_scale=self.env.max_yaw_rate_step)
                #render
                self.env.render()
               
                #take a step in the gym env
                observation, reward, done, truncated, info = self.env.step(action)
                #update the metric
                metric.update(info=info, reward=reward, eps_complete=(done or truncated))

            #reset the policy (path) for the next episode
            self.reset()
        
        self.env.close()
        
        #Save efficiency and effort plots
        metric.plot_scores(save_fig_dir=self.env.cfg.output_dir)

        return metric.success_rates, metric.efficiency_scores, metric.effort_scores, metric.rewards, algo_name

                



    def reset(self):
        self.path = None



if __name__ == "__main__":
    policy = PlanningBasedPolicy(planner_type='RRT')
    policy.evaluate(num_eps=10)

      


   
