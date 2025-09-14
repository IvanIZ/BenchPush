#Benchnpin Environments Imports
import benchnpin.environments
import gymnasium as gym
#Non-learning Planners for Maze NAMO
from benchnpin.baselines.maze_NAMO.planning_based.RRT.rrt import RRTPlanner
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
                                             robot_size= {'radius': robot_radius+0.8},
                                             bounds_pad=2.5)
           self.path = np.array(self.path)
                                         

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
        env = gym.make('maze-NAMO-v0', cfg=self.cfg)
        env = env.unwrapped

        #algorithm name for metrics
        if self.planner_type == 'RRT':
            algo_name = 'RRT'
        
        #metric instance for evaluation
        metric = MazeNamoMetric(alg_name=algo_name, robot_mass= env.cfg.robot.mass)
        
        #episodes
        for eps_idx in range(num_eps):
            print("Planning Based Progress: ", eps_idx, " / ", num_eps, " episodes")
            observation, info = env.reset()
            metric.reset(info)
            obstacles = info['obs']
            done = truncated = False

            #parameters that do not change per step
            goal = env.goal
            maze_vertices = env.maze_walls
            if env.cfg.robot.min_r is not None:
                robot_radius = env.cfg.robot.min_r
            else:
                robot_radius = env.cfg.agent.robot_r

            while not (done or truncated):
                #paramters that change per step
                robot_pos = info['state']
                obstacles = info['obs']
                #compute the next action (angular velocity)
                action = self.act(observation=(observation/255.0).astype(np.float64),
                                robot_pos=robot_pos, 
                                robot_radius=robot_radius,
                                goal=env.goal, 
                                maze_vertices= env.maze_walls,
                                obstacles= obstacles,
                                action_scale=env.max_yaw_rate_step)
                #render
                env.render()
               
                #take a step in the gym env
                observation, reward, done, truncated, info = env.step(action)
                #update the metric
                metric.update(info=info, reward=reward, eps_complete=(done or truncated))

            #reset the policy (path) for the next episode
            self.reset()
        
        env.close()
        
        #Save efficiency and effort plots
        metric.plot_scores(save_fig_dir=env.cfg.output_dir)

        return metric.success_rates, metric.efficiency_scores, metric.effort_scores, metric.rewards, algo_name

                



    def reset(self):
        self.path = None



if __name__ == "__main__":
    policy = PlanningBasedPolicy(planner_type='RRT')
    policy.evaluate(num_eps=10)

      


   
