from benchpush.common.metrics.base_metric import BaseMetric
import numpy as np


class MazeNamoMetric(BaseMetric):
    """
    Reference to paper "Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments"
    Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8954627
    """

    def __init__(self, alg_name, robot_mass) -> None:
        super().__init__(alg_name=alg_name)

        self.eps_reward = 0

        # NOTE in contrast to the Interactive Gibson Benchmark, for robot ice navigation environment, we keep track of the mass motion distance 
        # instead of displacement. The concept of displacement is to penalize environment disturbance, 
        # which is more applicable to indoor environments, less suitable for an ice field.
        self.total_mass_dist = 0               # \sum_{i=1}^{k}m_il_i

        self.robot_mass = robot_mass          # m_0
        self.total_robot_dist = 0                    # l_0


    def compute_efficiency_score(self):
        """
        Compute 1_{success} * (L / robot_dist)
        """

        if not self.trial_success:
            return 0
        else:
            return self.L / self.total_robot_dist


    def compute_effort_score(self):
        """
        Compute (m_0 * l_0) / (\sum_{i=0}^k m_i * l_i)
        """

        effort = (self.robot_mass * self.total_robot_dist) / (self.robot_mass * self.total_robot_dist + self.total_mass_dist)
        return effort

    
    def update(self, info, reward, eps_complete=False):
        self.eps_reward += reward

        self.total_mass_dist = info['total_work']
        self.trial_success = info['trial_success']

        # compute robot motion distance
        robot_state = info['state']
        self.total_robot_dist += np.linalg.norm(np.array(self.robot_state[:2]) - np.array(robot_state[:2]))
        self.robot_state = robot_state
        
        if eps_complete:
            self.rewards.append(self.eps_reward)
            self.efficiency_scores.append(self.compute_efficiency_score())
            self.effort_scores.append(self.compute_effort_score())


    def reset(self, info):
        self.eps_reward = 0
        self.total_mass_dist = 0
        self.total_robot_dist = 0
        self.trial_success = False

        self.robot_state = info['state']
        goal_dt = info['goal_dt']
        m_to_pix_scale = info['m_to_pix_scale']
        
        # shortest obstacle-free path length for the robot
        robot_pixel_x = int(self.robot_state[0] * m_to_pix_scale) 
        robot_pixel_y = int(self.robot_state[1] * m_to_pix_scale)
        self.L = goal_dt[robot_pixel_y, robot_pixel_x] / m_to_pix_scale
