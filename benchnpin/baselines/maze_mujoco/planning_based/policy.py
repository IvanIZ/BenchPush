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
ANGLE_TOL                 = np.deg2rad(3) 
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
           self.path = self.prune_by_distance(self.path, min_dist=0.3/2)
           self.RRTPlanner.plot_env(self.path)
           self.path = self.get_path_headings()

    def get_path_headings(self):
        # compute waypoint headings
        headings = [None]
        for i in range(1, len(self.path)):
            x_diff = self.path[i][0] - self.path[i - 1][0]
            y_diff = self.path[i][1] - self.path[i - 1][1]
            waypoint_headings = self.restrict_heading_range(np.arctan2(y_diff, x_diff))
            headings.append(waypoint_headings)

        headings = np.array(headings).reshape(-1, 1)
        path = np.concatenate((self.path, headings), axis=1)
        return path 
    
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

    def act(self, **kwargs):
        robot_pos = kwargs.get('robot_pos', None)
        action_scale = kwargs.get('action_scale', 1.0)

        #plan path offline
        # if self.path is None:
        #     self.plan_path(start=robot_pos[0:2], goal=goal, observation=observation, 
        #                    maze_vertices=maze_vertices, obstacles=obstacles,
        #                    robot_radius=robot_radius)
            
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
        env = gym.make('maze-NAMO-mujoco-v0', cfg=self.cfg, render_mode=None)
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
                    [(room_width/2,0),(room_width/2,room_length - room_length/3)]]

                robot_position = (robot_pos[0], robot_pos[1])
                robot_heading = robot_pos[2]


                action = self.execute_robot_path(robot_initial_position=robot_position, 
                                robot_initial_heading=robot_heading,
                                observation=observation,
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

    def execute_robot_path(self, robot_initial_position, robot_initial_heading, **kwargs):
        observation = kwargs.get('observation', None)
        robot_pos = kwargs.get('robot_pos', None)
        robot_radius = kwargs.get('robot_radius', None)
        goal = kwargs.get('goal', None)
        maze_vertices = kwargs.get('maze_vertices', None)
        obstacles = kwargs.get('obstacles', None)
        action_scale = kwargs.get('action_scale', 1.0)
        if self.path is None:
            self.plan_path(start=robot_pos[0:2], goal=goal, observation=observation, 
                           maze_vertices=maze_vertices, obstacles=obstacles,
                           robot_radius=robot_radius)

            robot_position = robot_initial_position
            robot_heading = robot_initial_heading
            robot_is_moving = True
            self.robot_distance = 0

            self.robot_waypoint_index = 1
            self.robot_waypoint_positions = [(waypoint[0], waypoint[1]) for waypoint in self.path]
            self.robot_waypoint_headings = [waypoint[2] for waypoint in self.path]

            self.robot_prev_waypoint_position = self.robot_waypoint_positions[self.robot_waypoint_index - 1]
            self.robot_waypoint_position = self.robot_waypoint_positions[self.robot_waypoint_index]
            self.robot_waypoint_heading = self.robot_waypoint_headings[self.robot_waypoint_index]
            
            sim_steps = 0
            self.done_turning = False
            self.heading_diff = 0
            self.prev_heading_diff = 0

        if self.check_robot_moving(robot_initial_position, robot_initial_heading):
            # if not robot_is_moving:
            #     break

            # store pose to determine distance moved during simulation step
            robot_prev_position = robot_initial_position
            robot_prev_heading = robot_initial_heading

            # compute robot pose for new constraint
            # robot_new_position = robot_position
            # robot_new_heading = robot_heading
            self.heading_diff = self.heading_difference(robot_prev_heading, self.robot_waypoint_heading)
            if np.abs(self.heading_diff) > TURN_STEP_SIZE / 2 and np.abs(self.heading_diff - self.prev_heading_diff) > 0.001:
                pass
            else:
                self.done_turning = True
                if self.distance(robot_prev_position, self.robot_waypoint_position) < MOVE_STEP_SIZE:
                    robot_new_position = self.robot_waypoint_position

            # change robot pose (use controller)
            v, w, _ = self.make_controller(robot_prev_position, robot_prev_heading, self.robot_waypoint_position)
            # w = self.act(**kwargs)
            if not self.done_turning:
                v = 0
            # v_l, v_r = vw_to_wheels(v, w)

            # apply the control 'frame_skip' steps
            # self.do_simulation([v_l, v_r], self.frame_skip)
        return v, w
        
    def check_robot_moving(self, robot_position, robot_heading):
        robot_is_moving = True
        # get new robot pose
        # robot_position = get_body_pose_2d(self.model, self.data, self.robot_name_in_xml)[:2]
        # robot_heading = self.restrict_heading_range(quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7]))
        self.prev_heading_diff = self.heading_diff
        
        # stop moving if robot collided with obstacle
        # self.robot_hit_obstacle = self.robot_hits_static()
        # if self.distance(robot_prev_waypoint_position, robot_position) > MOVE_STEP_SIZE:
        #     if self.robot_hit_obstacle:
        #         robot_is_moving = False
                # break   # Note: self.robot_distance does not get updated

        # stop if robot reached waypoint
        if (self.distance(robot_position, self.robot_waypoint_positions[self.robot_waypoint_index]) < WAYPOINT_MOVING_THRESHOLD/2
                and np.abs(robot_heading - self.robot_waypoint_headings[self.robot_waypoint_index]) < WAYPOINT_TURNING_THRESHOLD):

            # update distance moved
            self.robot_distance += self.distance(self.robot_prev_waypoint_position, robot_position)

            # increment waypoint index or stop moving if done
            if self.robot_waypoint_index == len(self.robot_waypoint_positions) - 1:
                robot_is_moving = False
            else:
                self.robot_waypoint_index += 1
                self.robot_prev_waypoint_position = self.robot_waypoint_positions[self.robot_waypoint_index - 1]
                self.robot_waypoint_position = self.robot_waypoint_positions[self.robot_waypoint_index]
                self.robot_waypoint_heading = self.robot_waypoint_headings[self.robot_waypoint_index]
                self.done_turning = False
                self.path = self.path[1:]

        # sim_steps += 1
        # if sim_steps % 10 == 0 and self.cfg.render.show:
        #     self.render_env()

        # break if robot is stuck
        # if sim_steps > STEP_LIMIT:
        #     break

        # robot_angle = quat_z_yaw(*self.data.qpos[self.qpos_index_base+3:self.qpos_index_base+7])
        # robot_heading = self.restrict_heading_range(robot_angle)
        # self.robot_turn_angle = self.heading_difference(robot_initial_heading, robot_heading) 
        return robot_is_moving

    def make_controller(self, curr_pos, curr_yaw, goal_pos):
        """Simple proportional controller instead of PI controller due to ease of
        computation and faster training. Chosen Kp to be sufficiently large so that
        the steady state error for step input is sufficiently low"""

        # Current position and angle of robot  
        # pos = data.qpos[qpos_index : qpos_index + 2]
        # qw, qx, qy, qz = data.qpos[qpos_index + 3 : qpos_index + 7]
        # yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        # The distance to the set goal
        # vec  = goal_xy - pos
        vec = np.array(goal_pos) - np.array(curr_pos)
        dist = np.linalg.norm(vec)
        
        
        # Angle computations
        goal_head = np.arctan2(vec[1], vec[0])
        err_yaw   = np.arctan2(np.sin(goal_head - curr_yaw), np.cos(goal_head - curr_yaw))
        
        # Controller characteristics
        k_v, k_w = 0.2, 8.0

        # Rotation
        if abs(err_yaw) > ANGLE_TOL:
            return 0.0, k_w * err_yaw, dist

        # # Moving to required position
        # else:
            # return k_v * dist, 0.0, dist
        return k_v, k_w * err_yaw, dist

    def restrict_heading_range(self, heading):
        return np.mod(heading + np.pi, 2 * np.pi) - np.pi

    def heading_difference(self, heading1, heading2):
        return self.restrict_heading_range(heading1 - heading2)
    
    def distance(self, position1, position2):
        return np.linalg.norm(np.asarray(position1)[:2] - np.asarray(position2)[:2])


    def reset(self):
        self.path = None



if __name__ == "__main__":
    policy = PlanningBasedPolicy(planner_type='RRT')
    policy.evaluate(num_eps=10)

      


   
