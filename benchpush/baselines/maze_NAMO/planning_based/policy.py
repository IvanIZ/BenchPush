#Benchnpin Environments Imports
import benchpush.environments
import gymnasium as gym
#Non-learning Planners for Maze NAMO
from benchpush.baselines.maze_NAMO.planning_based.RRT.rrt import RRTPlanner
#Base Policy Class
from benchpush.baselines.base_class import BasePolicy
#Benchnpin Metrics 
from benchpush.common.metrics.maze_namo_metric import MazeNamoMetric
#DP Controller
from benchpush.common.controller.dp import DP

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
                                             robot_size= {'radius': robot_radius*1.5})
           self.path = np.array(self.path)
           self.path = self.prune_by_distance(self.path, min_dist=0.3)
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
        print(path.shape)
        print(len(path))
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
        print(pruned.shape)
        print(len(pruned))
        return pruned
    
    def plot_env(self, nodes, parent, path, ax=None):
        
        """Plot the scene: walls (inflated), boxes.
        path and tree"""

        print("Plotting environment...")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        if self.RRTPlanner.scene.walls_infl:
            x, y = self.RRTPlanner.scene.walls_infl.exterior.xy
            ax.fill(x, y, color='lightgray', alpha=0.7, label='Inflated Walls')
            #holes
            for interior in self.RRTPlanner.scene.walls_infl.interiors:
                x, y = interior.xy
                ax.fill(x, y, color='white')
            
        if self.RRTPlanner.scene.boxes:
            for i, box in enumerate(self.RRTPlanner.scene.boxes):
                if isinstance(box, Polygon):
                    x, y = box.exterior.xy
                    ax.fill(x, y, color='orange', alpha=0.7, label='Box' if i == 0 else "")
                elif isinstance(box, LineString):
                    x, y = box.xy
                    ax.plot(x, y, color='orange', linewidth=4, label='Box' if i == 0 else "")
        # Plot RRT tree
        for i, node in enumerate(nodes):
            if parent[i] != -1:
                p_node = nodes[parent[i]]
                ax.plot([node[0], p_node[0]], [node[1], p_node[1]], color='blue', linewidth=0.5, alpha=0.5)
        
        # Plot path if available
        if path is not None and len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'r', linewidth=2, label='Planned Path')
        # Plot start and goal
        ax.plot(nodes[0][0], nodes[0][1], 'go', markersize=10, label='Start')
        ax.plot(nodes[-1][0], nodes[-1][1], 'ro', markersize=10, label='Goal')
        ax.set_aspect('equal')
        ax.set_xlim(scene.bounds[0], scene.bounds[1])
        ax.set_ylim(scene.bounds[2], scene.bounds[3])
        ax.set_title('RRT Planning Environment')
        ax.legend()
        plt.savefig('maze_namo_rrt_environment.png')
                                         

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
                print(obstacles)
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

      


   
