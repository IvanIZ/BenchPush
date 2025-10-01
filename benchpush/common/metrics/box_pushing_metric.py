from benchpush.common.metrics.base_metric import BaseMetric
import numpy as np


class BoxDeliveryMetric(BaseMetric):
    """
    Reference to paper "Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments"
    Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8954627
    """

    def __init__(self, alg_name, robot_mass) -> None:
        super().__init__(alg_name=alg_name)

        self.eps_reward = 0

        # NOTE in contrast to the Interactive Gibson Benchmark, for ship ice navigation environment, we keep track of the mass motion distance 
        # instead of displacement. The concept of displacement is to penalize environment disturbance, 
        # which is more applicable to indoor environments, less suitable for an ice field.
        self.total_box_dist = 0               # \sum_{i=1}^{k}m_il_i

        self.robot_mass = robot_mass          # m_0
        self.total_robot_dist = 0                    # l_0


    def compute_efficiency_score(self):
        """
        Compute 1_{success} * (L / ship_dist)
        """
        raise NotImplementedError


    def compute_effort_score(self):
        """
        Compute (m_0 * l_0) / (\sum_{i=0}^k m_i * l_i)
        """

        effort = (self.robot_mass * self.total_robot_dist) / (self.robot_mass * self.total_robot_dist + self.total_box_dist)
        return effort

    
    def update(self, info, eps_complete=False):

        self.total_box_dist = info['cumulative_cube_distance']
        # self.trial_success = info['trial_success']

        self.total_robot_dist = info['cumulative_distance']
        self.eps_reward = info['cumulative_reward']
        
        if eps_complete:
            self.rewards.append(self.eps_reward)
            # self.efficiency_scores.append(self.compute_efficiency_score())
            self.effort_scores.append(self.compute_effort_score())


    def reset(self, info):
        self.eps_reward = 0
        self.total_box_dist = 0
        self.total_robot_dist = 0
        # self.trial_success = False
