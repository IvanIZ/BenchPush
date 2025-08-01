import benchnpin.environments
import gymnasium as gym
from benchnpin.baselines.ship_ice_nav.planning_based.planners.lattice import LatticePlanner
from benchnpin.baselines.ship_ice_nav.planning_based.planners.predictive import PredictivePlanner
from benchnpin.baselines.base_class import BasePolicy
from benchnpin.common.metrics.ship_ice_metric import ShipIceMetric
from benchnpin.common.controller.dp import DP
from typing import List, Tuple
import numpy as np

class PlanningBasedPolicy(BasePolicy):
    """
    A baseline policy for autonomous ship navigation in ice-covered waters. 
    This policy first plans a path using a ship planner and outputs actions to track the planned path.
    """

    def __init__(self, planner_type, cfg=None, planner_config=None) -> None:
        super().__init__()

        if planner_type not in ['predictive', 'lattice', 'straight']:
            raise Exception("Invalid planner type. Choose a planner between 'lattice', 'predictive', or 'straight'.")
        self.planner_type = planner_type

        self.lattice_planner = LatticePlanner(planner_config)
        self.predictive_planner = PredictivePlanner()
        self.path = None

        self.cfg = cfg

    
    def plan_path(self, ship_pos, goal, observation, conc, obstacles=None):
        if self.planner_type == 'lattice':
            self.path = self.lattice_planner.plan(ship_pos=ship_pos, goal=goal, obs=obstacles)

        elif self.planner_type == 'predictive':
            occ_map = observation[0]
            footprint = observation[1]
            self.path = self.predictive_planner.plan(ship_pos=ship_pos, goal=goal, occ_map=occ_map, footprint=footprint, conc=conc)

        elif self.planner_type == 'straight':
            self.path = self.straight_planner(ship_pos, goal)

    
    def straight_planner(self, ship_pose, goal, dy=10):

        x, y, theta = ship_pose
        _, goal_y = goal

        # generate evenly-space y values, including goal_y if it lands on the grid
        y_vals = np.arange(y, goal_y + dy * 0.5, dy)

        # assemble the (N, 3) path
        path = np.column_stack((
            np.full_like(y_vals, x, dtype=float), 
            y_vals, 
            np.full_like(y_vals, theta, dtype=float)
        ))

        return path

    def act(self, observation, **kwargs):
        # tunables parameters
        CFG = dict(
            THRESH      = 10.0,     # cross‑track switch
            look_car    = 50.0,     # carrot distance when far
            d_back      = 15.0,     # slice behind (for straightness test)
            d_ahead     = 25.0,     # slice ahead

            # yaw‑rate PID
            kp = 0.10,  ki = 0.15,  kd = 2.0,
            i_cap = 10.0,  dead = 0.02,

            # straight‑line turn
            straight_ang = 0.100,   # max value
            yaw_big      = 0.50,    # only if yaw_err is greater than this
            omega_small  = 0.002,   # gentle turn rate

            # surge PI
            kp_v = 0.50,  ki_v = 0.05,  v_max = 2.5,
            omega_max = 0.02 # cap
        )
        dt                       = kwargs.get("dt", 0.005)
        ship_x, ship_y, ship_yaw = kwargs["ship_pos"]
        action_scale             = kwargs["action_scale"]

        # building path once
        if self.path is None:
            self.plan_path(kwargs["ship_pos"], kwargs["goal"],
                        observation,
                        kwargs.get("conc"),
                        kwargs.get("obstacles"))

        px, py = self.path[:, 0], self.path[:, 1]

        # nearest path index and cross‑track distance
        d2     = (px - ship_x)**2 + (py - ship_y)**2
        i_near = int(np.argmin(d2))
        ct_err = float(np.sqrt(d2[i_near]))

        # far case
        if ct_err > CFG['THRESH']:

            # carrot target
            dist, j = 0.0, i_near
            while dist < CFG['look_car'] and j + 1 < len(px):
                dist += np.hypot(px[j+1]-px[j], py[j+1]-py[j]); j += 1
            x_t, y_t = px[j], py[j]

            yaw_ref = np.arctan2(y_t - ship_y, x_t - ship_x)
            yaw_err = np.arctan2(np.sin(yaw_ref - ship_yaw),
                                np.cos(yaw_ref - ship_yaw))

            # straight‑line detector
            # building local slice 15 m back and 25 m ahead
            dist_b, k = 0.0, i_near
            while dist_b < CFG['d_back'] and k > 0:
                dist_b += np.hypot(px[k]-px[k-1], py[k]-py[k-1]); k -= 1
            dist_f, j2 = 0.0, i_near
            while dist_f < CFG['d_ahead'] and j2 + 1 < len(px):
                dist_f += np.hypot(px[j2+1]-px[j2], py[j2+1]-py[j2]); j2 += 1

            v_back = np.array([px[i_near] - px[k],  py[i_near] - py[k]])
            v_fwd  = np.array([px[j2]     - px[i_near], py[j2]     - py[i_near]])
            ang_seg = abs(np.arctan2(v_back[0]*v_fwd[1] - v_back[1]*v_fwd[0],
                                    np.dot(v_back, v_fwd)))

            straight_slice = (ang_seg < CFG['straight_ang'])
            big_yaw_err    = (abs(yaw_err) > CFG['yaw_big'])

            # yaw control
            if straight_slice and big_yaw_err:
                # gentle and fixed‑rate correction
                omega = np.sign(yaw_err) * CFG['omega_small']
            else:
                # full PID yaw‑rate
                if not hasattr(self, "_int_yaw"):
                    self._int_yaw, self._prev_yaw = 0.0, yaw_err

                if abs(yaw_err) > CFG['dead']:
                    self._int_yaw = np.clip(self._int_yaw + yaw_err*dt,
                                            -CFG['i_cap'], CFG['i_cap'])
                else:
                    self._int_yaw *= 0.8

                d_yaw          = (yaw_err - self._prev_yaw)/dt
                self._prev_yaw = yaw_err

                omega = (CFG['kp']*yaw_err +
                        CFG['ki']*self._int_yaw +
                        CFG['kd']*d_yaw)
                omega = np.clip(omega, -CFG['omega_max'], CFG['omega_max'])

        # near case
        else:
            dist_b, k = 0.0, i_near
            while dist_b < CFG['d_back'] and k > 0:
                dist_b += np.hypot(px[k]-px[k-1], py[k]-py[k-1]); k -= 1
            dist_f, j = 0.0, i_near
            while dist_f < CFG['d_ahead'] and j + 1 < len(px):
                dist_f += np.hypot(px[j+1]-px[j], py[j+1]-py[j]); j += 1
            dx, dy  = px[j]-px[k], py[j]-py[k]
            yaw_ref = np.arctan2(dy, dx)
            yaw_err = np.arctan2(np.sin(yaw_ref - ship_yaw),
                                np.cos(yaw_ref - ship_yaw))
            omega   = np.clip(yaw_err/dt, -CFG['omega_max'], CFG['omega_max'])

        # surge control
        if not hasattr(self, "_int_v"): self._int_v = 0.0
        self._int_v = np.clip(self._int_v + CFG['ki_v']*ct_err*dt, 0, CFG['v_max'])
        v_cmd = min(CFG['v_max'], CFG['kp_v']*ct_err + self._int_v)

        return omega/action_scale, 20.0*v_cmd

    def evaluate(self, num_eps: int, model_eps: str ='latest') -> Tuple[List[float], List[float], List[float], str]:
        env = gym.make('ship-ice-v0', cfg=self.cfg)
        env = env.unwrapped

        if self.planner_type == 'lattice':
            alg_name = "Lattice Planning"
        elif self.planner_type == 'predictive':
            alg_name = "Predictive Planning"
        metric = ShipIceMetric(alg_name=alg_name, ship_mass=env.cfg.ship.mass, goal=env.goal)

        for eps_idx in range(num_eps):
            print("Planning Based Progress: ", eps_idx, " / ", num_eps, " episodes")
            observation, info = env.reset()
            metric.reset(info)
            obstacles = info['obs']
            done = truncated = False

            while True:
                action = self.act(observation=(observation / 255).astype(np.float64), ship_pos=info['state'], obstacles=obstacles, 
                                    goal=env.goal,
                                    conc=env.cfg.concentration, 
                                    action_scale=env.max_yaw_rate_step)
                observation, reward, done, truncated, info = env.step(action)
                metric.update(info=info, reward=reward, eps_complete=(done or truncated))
                obstacles = info['obs']
                if done or truncated:
                    break

        env.close()
        metric.plot_scores(save_fig_dir=env.cfg.output_dir)
        return metric.efficiency_scores, metric.effort_scores, metric.rewards, alg_name

    
    def reset(self):
        self.path = None
