"""
An example script for running baseline policy for ship ice navigation
"""

import benchpush.environments
import gymnasium as gym
import numpy as np
from benchpush.baselines.area_clearing.planning_based.policy import PlanningBasedPolicy

# initialize RL policy
policy = PlanningBasedPolicy()
policy.evaluate(num_eps=5)