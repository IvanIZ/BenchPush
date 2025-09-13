#Benchnpin Environments Imports
import benchnpin.environments
import gymnasium as gym
#Non-learning Planners for Maze NAMO
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

    def __init__(self, planner_type, cfg=None) -> None:
        super().__init__()

        if planner_type not in ['RRT', 'Astar']:
            raise Exception("Invalid planner type")
        self.planner_type = planner_type

        self.path = None

        self.cfg = cfg


    def plan_path(self, start, goal, observation, obstacles=None):
        pass

    def act(self, observation, **kwargs):
        pass
        

    def evaluate(self,  num_eps: int, model_eps: str ='latest') -> Tuple[List[float], List[float], List[float], str]:
        env = gym.make('maze-NAMO-v0', cfg=self.cfg)
        env = env.unwrapped

        #algorithm name for metrics
        if self.planner_type == 'RRT':
            algo_name = 'RRT'
        
        #metric instance for evaluation
        metric = MazeNamoMetric(algorithm_name=algo_name, robot_mass= env.cfg.robot.mass)
        
        #episodes
        for eps_idx in range(num_eps):
            print("Planning Based Progress: ", eps_idx, " / ", num_eps, " episodes")
            observation, info = env.reset()
            metric.reset(info)
            obstacles = info['obs']
            done = truncated = False

            #parameters per episode
            goal = info['goal']
            maze_vertices = info['maze_vertices']

            while not (done or truncated):
                #paramters per step
                robot_pos = info['state']
                obstacles = info['obs']
                #compute the next action (angular velocity)
                action = self.act(observation=(observation/255.0).astype(np.float64), 
                                    robot_pos=robot_pos, 
                                    goal=goal, 
                                    maze_vertices=maze_vertices,
                                    obstacles=obstacles,
                                    action_scale=env.max_yaw_rate_step)
                #take a step in the gym env
                observation, reward, done, truncated, info = env.step(action)
                #update the metric
                metric.update(info=info, reward=reward, eps_complete=(done or truncated))

           
        env.close()
        #Save efficiency and effort plots
        metric.plot_scores(save_fig_dir=env.cfg.output_dir)

        return metric.success_rates, metric.efficiency_scores, metric.effort_scores, metric.rewards, algo_name

                



    def reset(self):
        pass



if __name__ == "__main__":
    env = gym.make('maze-NAMO-v0')
    #unwrap env 
    env = env.unwrapped
    obs, info = env.reset()
    policy = PlanningBasedPolicy(planner_type='RRT')
    done = False
    total_reward = 0
    #print info in a line each
    for key, value in info.items():
        print(f"{key}")    
    #plot obstacles in key 'obs' from vertices
    plt.figure()
    for obs in env.obstacles:
        
        poly = Polygon(obs)
        x,y = poly.exterior.xy
        plt.plot(x, y)
    
    for wall in env.maze_walls:
        line = LineString(wall)
        x,y = line.xy
        plt.plot(x, y, color='black', linewidth=2)
    #start and goal
    plt.plot(env.start[0], env.start[1], 'go', markersize=10, label='Start')
    plt.plot(env.goal[0], env.goal[1], 'ro', markersize=10, label='Goal')
    plt.legend()
    plt.axis('equal')
    plt.savefig('maze_namo_env.png')
    plt.show()


   
