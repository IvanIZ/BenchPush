from typing import List

import numpy as np

from benchnpin.common.evaluation.metrics import euclid_dist, path_length


R = lambda theta: np.asarray([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])


class State:

    def __init__(self, x, y, yaw, input_lims, dt):
        self.x = x  # global x position [m]
        self.y = y  # global y position [m]
        self.yaw = yaw  # heading [rad]

        # our state vector in our linear system is [r, u, v]'
        self.r = 0  # yaw rate (deg/s)
        self.u = 0  # surge velocity
        self.v = 0  # sway velocity

        self.r_lim, self.u_lim, self.v_lim = input_lims

        self.dt = dt

    def limits(self, r, u, v):
        # Impose limits
        if abs(r) > self.r_lim:
            r = self.r_lim * np.sign(r)
        if abs(u) > self.u_lim:
            u = self.u_lim * np.sign(u)
        if abs(v) > self.v_lim:
            v = self.v_lim * np.sign(v)
        return r, u, v

    def update(self, r, u, v):
        r, u, v = self.limits(r, u, v)
        self.r = r
        self.u = u
        self.v = v

    def get_global_velocity(self):
        return R(self.yaw) @ [self.u, self.v]

    def update_pose(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

    def integrate(self):
        # rotate surge and sway velocities into global frame
        u_g, v_g = R(self.yaw) @ [self.u, self.v]
        yaw = (self.yaw + self.dt * self.r * np.pi / 180) % (2 * np.pi)
        x = self.x + self.dt * u_g
        y = self.y + self.dt * v_g
        return x, y, yaw


class TargetCourse:
    """
    Original code from https://github.com/AtsushiSakai/PythonRobotics
    """

    def __init__(self, cx, cy, ch, Lfc):
        self.cx = cx
        self.cy = cy
        self.ch = ch
        self.Lfc = Lfc
        self.path_length = path_length(np.asarray([cx, cy]).T, cumsum=True)
        self.setpoint_al = 0

    def init_setpoint(self, x, y):
        ind = self.search_target_index(x, y)[1]

        # search look ahead target point index
        while self.Lfc > euclid_dist((x, y), (self.cx[ind], self.cy[ind])):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        self.setpoint_al = self.path_length[min(len(self.path_length) - 1, ind)]
        return [self.cx[ind], self.cy[ind], self.ch[ind]], ind

    def update(self, cx, cy, ch):
        # get current setpoint
        ind = len(self.path_length[self.path_length <= self.setpoint_al])
        x, y = self.cx[ind], self.cy[ind]

        # update with new path
        self.cx = cx
        self.cy = cy
        self.ch = ch
        self.path_length = path_length(np.asarray([cx, cy]).T, cumsum=True)

        # get new setpoint by projecting old setpoint onto new path
        ind = self.search_target_index(x, y)[1]
        self.setpoint_al = self.path_length[min(len(self.path_length) - 1, ind)]

    def search_target_index(self, x, y):
        # search nearest point index
        dx = [x - icx for icx in self.cx]
        dy = [y - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        ind = np.argmin(d)

        return [self.cx[ind], self.cy[ind], self.ch[ind]], ind

    def advance(self, target_speed, dt):
        self.setpoint_al += target_speed * dt
        ind = len(self.path_length[self.path_length < self.setpoint_al])
        return [self.cx[ind], self.cy[ind], self.ch[ind]], ind
        

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.sum_error = 0
        self.prev_error = None

    def __call__(self, err, dt):
        d_err = (err - (self.prev_error or err)) / dt
        self.sum_error += err * dt
        self.prev_error = err

        return self.Kp * err + self.Ki * self.sum_error + self.Kd * d_err


class DP:
    def __init__(self,
                 dt: float, target_speed: float,
                 x: float, y: float, yaw: float,
                 cx: np.ndarray, cy: np.ndarray, ch: np.ndarray,
                 A: List = None, B: List = None, input_lims: List = None,
                 Lfc: float = None, PID_gains: List = None,
                 output_dir=None):
        self.dt = dt

        if A is None:
            self.A = np.zeros((3, 3))
        else:
            self.A = np.asarray(A)  # x_{k+1} = Ax_k + Bu_k, discretized dynamics
        
        if B is None:
            self.B = np.zeros(3)
        else:
            self.B = np.asarray(B)
        
        if input_lims is None:
            input_lims = [0, 0, 0]

        if PID_gains is None:
            PID_gains = [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]
        
        if Lfc is None:
            Lfc = 0.0
        

        self.target_speed = target_speed

        self.time = 0
        self.state = State(x, y, yaw, input_lims, dt)  # this is the current state
        self.target_course = TargetCourse(cx, cy, ch, Lfc)
        self.setpoint, _ = self.target_course.init_setpoint(self.state.x, self.state.y)
        self.input = (0, 0, 0)

        # initialize PD for each of yaw rate, surge velocity, and sway velocity
        self.pd = [PID(*PID_gains[0]),  # too high of gains will make ship swing around too much
                   PID(*PID_gains[1]),
                   PID(*PID_gains[2])]

    def get_setpoint(self, cur_speed=None):
        # the switch from one index to the next causes a spike
        # in the derivative term in the pid controller
        
        if cur_speed is None:
            return self.target_course.advance(self.target_speed, self.dt)[0]
        else:
            return self.target_course.advance(cur_speed, self.dt)[0]

        # return self.target_course.search_target_index(self.state.x, self.state.y)[0]


    def __call__(self, *pose):
        """
        Update the DP controller.

        Call the DP controller with input and return a computed control signal
        Input is of the form x, y, yaw
        """
        assert len(pose) == 3
        # unpack input
        x, y, yaw = pose

        self.state.update_pose(x, y, yaw)

        # get global error in heading, x, and y position
        self.setpoint[2] = np.unwrap([yaw, self.setpoint[2]])[1]
        e_x, e_y, e_yaw = np.asarray(self.setpoint) - np.asarray([x, y, yaw])

        # rotate error into body frame of vessel
        # need the inverse of the rotation matrix
        e_rot = R(yaw).T @ [e_x, e_y]

        # update PD for each yaw rate, surge v and sway v
        self.input = [PD(error, dt=self.dt) for error, PD in zip([e_yaw, *e_rot], self.pd)]
        state_next = self.A @ [self.state.r, self.state.u, self.state.v] + self.B * self.input
        self.state.update(*state_next)
        self.time += self.dt


    def ideal_control(self, *pose):
        """
        An ideal controller that directly outputs the velocity to perfectly
        track the setpoint

        Assumptions: 
        1. surge is set constant to the target speed
        2. sway is 0
        """
        assert len(pose) == 3
        
        # unpack input
        x, y, yaw = pose

        # compute the desired heading angle (global frame)
        x_d, y_d, yaw_d = np.asarray(self.setpoint)
        theta_d = np.arctan2(y_d - y, x_d - x)

        # compute the heading error (global frame)
        theta_e = theta_d - yaw

        # normalize the heading error to the range [-pi, pi]
        theta_e = np.arctan2(np.sin(theta_e), np.cos(theta_e))

        # angular velocity
        omega = 1.0 * theta_e
        omega = omega / self.dt

        # global velocity
        global_velocity = R(yaw) @ [self.target_speed, 0]

        return omega, global_velocity
