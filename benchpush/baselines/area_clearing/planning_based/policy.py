import benchpush.environments
import gymnasium as gym
import numpy as np
import os

from shapely.geometry import Polygon, LineString
from shapely import shortest_line

from benchpush.common.controller.dp import DP
from benchpush.common.utils.utils import DotDict
from benchpush.common.evaluation.metrics import euclid_dist
from benchpush.common.metrics.task_driven_metric import TaskDrivenMetric

from benchpush.baselines.base_class import BasePolicy

from benchpush.baselines.area_clearing.planning_based.GTSPPlanner.transition_graph_lookup import TransitionGraphLookup
from benchpush.baselines.area_clearing.planning_based.GTSPPlanner.solve_gtsp import GTSPSolver

import numpy as np

EXTEND_BUFFER = 2

def extend_linestring_at_start(line: LineString, distance: float) -> LineString:
    """Extend a LineString by a small distance at the first end."""
    if len(line.coords) < 2:
        raise ValueError("LineString must have at least two points")

    # Get start and end points
    start, second = np.array(line.coords[0]), np.array(line.coords[1])
    end = np.array(line.coords[-1])

    # Compute direction vectors
    start_direction = start - second

    # Normalize and scale
    if np.linalg.norm(start_direction) == 0:
        start_extension = start
    else:
        start_extension = start + (start_direction / np.linalg.norm(start_direction)) * distance

    # Create new extended LineString
    new_coords = [tuple(start_extension)] + list(line.coords)
    return LineString(new_coords)


class PlanningBasedPolicy(BasePolicy):
    """
    A baseline policy for area clearing using . 
    This policy first plans a path using a ship planner and outputs actions to track the planned path.
    """

    def __init__(self, glns_executable_path, cfg=None) -> None:
        super().__init__()

        # get current directory of this script
        self.current_dir = os.path.dirname(__file__)
        # construct absolute path to the env_config folder
        gtsp_cfg_file = os.path.join(self.current_dir, 'gtsp_config.yaml')
        self.gtsp_cfg = DotDict.load_from_file(gtsp_cfg_file)

        self.cfg = cfg

        self.path = None
        self.dt = self.gtsp_cfg.controller.dt
        self.velocity = self.gtsp_cfg.controller.target_speed

        self.boundary_vertices = None
        self.walls = None
        self.static_obstacles = None

        self.boundary_polygon = None
        self.boundary_goals = None

        self.current_point_id = 0

        self.glns_executable_path = glns_executable_path


    def update_boundary_and_obstacles(self, boundary, walls, static_obstacles):
        self.boundary_vertices = boundary
        self.walls = walls
        self.static_obstacles = static_obstacles

        self.boundary_polygon = Polygon(self.boundary_vertices)

        self._compute_boundary_goals()

    def _compute_boundary_goals(self):
        if self.boundary_vertices is None:
            return None
        
        boundary_edges = []
        for i in range(len(self.boundary_vertices)):
            boundary_edges.append([self.boundary_vertices[i], self.boundary_vertices[(i + 1) % len(self.boundary_vertices)]])
        
        boundary_linestrings = [LineString(edge) for edge in boundary_edges]

        # remove walls from boundary
        for wall in self.walls:
            wall_polygon = LineString(wall)
            wall_polygon = wall_polygon.buffer(0.1)
            for i in range(len(boundary_linestrings)):
                boundary_linestrings[i] = boundary_linestrings[i].difference(wall_polygon)

        # convert multilinestrings to linestrings
        temp_boundary_linestrings = boundary_linestrings.copy()
        boundary_linestrings = []
        for line in temp_boundary_linestrings:
            if line.geom_type == 'MultiLineString':
                boundary_linestrings.extend([ls for ls in list(line.geoms) if ls.length > 0.1])
            elif line.geom_type == 'LineString':
                if line.length > 0.1:
                    boundary_linestrings.append(line)
            else:
                raise ValueError("Invalid geometry type to handle")

        self.boundary_goals = boundary_linestrings
    
    def plan_path(self, agent_pos, observation, obstacles=None):
        obstacles_to_push = []
        
        for obstacle in obstacles:
            obstacle_poly = Polygon(obstacle)
            is_in_boundary = self.boundary_polygon.intersects(obstacle_poly)
            if is_in_boundary:
                obstacles_to_push.append(obstacle)

        all_pushing_paths = []
        for obstacle in obstacles_to_push:
            obstacle_poly = Polygon(obstacle)
            # shrink the obstacle by a small amount
            obstacle_poly = obstacle_poly.buffer(-0.4)
            obs_pushing_paths = []
            for goal in self.boundary_goals:
                path = shortest_line(obstacle_poly, goal)
                path = extend_linestring_at_start(path, EXTEND_BUFFER)
                # round to 2 decimal places
                path = LineString(np.round(np.array(path.coords), 7))
                obs_pushing_paths.append(path)
            all_pushing_paths.append(obs_pushing_paths)

        transition_graph = TransitionGraphLookup.compute_transition_graph(agent_pos, obstacles_to_push, all_pushing_paths)

        gtsp_solver = GTSPSolver(self.glns_executable_path)
        output_tuple, time_found = gtsp_solver.solve_GTSP_Problem(obstacles_to_push, all_pushing_paths, transition_graph, agent_pos)
        final_nodes, transition_paths, transition_cost, transition_length, total_angle, transition_costs = output_tuple

        self.path = [agent_pos]
        for i in range(len(final_nodes)):
            point_1 = final_nodes[i].traversal_route[0]
            point_2 = final_nodes[i].traversal_route[-1]

            heading = np.arctan2(point_2[1] - point_1[1], point_2[0] - point_1[0])

            self.path.append([point_1[0], point_1[1], heading])
            self.path.append([point_2[0], point_2[1], heading])
        
        self.path = np.array(self.path)

    def act(self, observation, **kwargs):

        # parameters for planners
        agent_pos = kwargs.get('agent_pos', [0, 0, np.pi / 2])
        obstacles = kwargs.get('obstacles', None)

        omega = 0.0
        
        # plan a path
        if self.path is None:
            self.plan_path(agent_pos, observation, obstacles)

            # setup dp controller to track the planned path
            cx = self.path.T[0]
            cy = self.path.T[1]
            ch = self.path.T[2]
            self.dp = DP(x=agent_pos[0], y=agent_pos[1], yaw=agent_pos[2],
                    cx=cx, cy=cy, ch=ch, **self.gtsp_cfg.controller)
            self.dp_state = self.dp.state
        
            self.current_point_id = 1

        if(self.current_point_id >= len(self.path)):
            return 0.0, 0.0

        if(euclid_dist(agent_pos, self.path[self.current_point_id]) < 0.4):
            self.current_point_id += 1

            if(self.current_point_id >= len(self.path)):
                return 0.0, 0.0
            
            # update setpoint
            x_s, y_s, h_s = self.path[self.current_point_id]
            self.dp.setpoint = np.asarray([x_s, y_s, np.unwrap([self.dp_state.yaw, h_s])[1]])

        # call ideal controller to get angular velocity control
        omega, velocity = self.dp.ideal_control(agent_pos[0], agent_pos[1], agent_pos[2])

        global_velocity = np.linalg.norm(velocity)

        return global_velocity, omega


    def evaluate(self, num_eps: int, model_eps: str ='latest') -> list:
        env = gym.make('area-clearing-v0', cfg=self.cfg)
        env = env.unwrapped

        old_action_type = env.cfg.agent.action_type
        old_t_max = env.cfg.sim.t_max

        env.cfg.agent.action_type = 'velocity'
        env.cfg.sim.t_max = 2000

        metric = TaskDrivenMetric(alg_name="GTSP", robot_mass=env.cfg.agent.mass)

        rewards_list = []
        for eps_idx in range(num_eps):
            print("Progress: ", eps_idx, " / ", num_eps, " episodes")
            observation, info = env.reset()
            metric.reset(info)
            obstacles = info['obs']

            self.update_boundary_and_obstacles(info['boundary'], info['walls'], info['static_obstacles'])
            done = truncated = False
            eps_reward = 0.0

            while True:
                action = self.act(observation=(observation).astype(np.float64), agent_pos=info['state'], obstacles=obstacles)
                env.update_path(self.path)

                if env.cfg.agent.action_type == 'velocity':
                    scaled_action = [action[0] / env.target_speed, action[1] / env.max_yaw_rate_step]
                else:
                    raise Exception('Invalid action type. Must use velocity control')

                observation, reward, done, truncated, info = env.step(scaled_action)

                metric.update(info=info, reward=reward, eps_complete=(done or truncated))
                obstacles = info['obs']
                env.render()
                eps_reward += reward

                if done or truncated:
                    rewards_list.append(eps_reward)
                    self.reset()
                    break

        env.close()
        metric.plot_scores(save_fig_dir=env.cfg.output_dir)

        env.cfg.agent.action_type = old_action_type
        env.cfg.sim.t_max = old_t_max

        return metric.success_rates, metric.efficiency_scores, metric.effort_scores, metric.rewards, "GTSP"
    
    def reset(self):
        self.path = None