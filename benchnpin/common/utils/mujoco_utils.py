import numpy as np

# Some properties of Turtlebot3
R_WHEEL, L_AXLE, MAX_WSPD = 0.033, 0.160, 8.0
LEFT_ACT, RIGHT_ACT       = 0, 1
ANGLE_TOL                 = np.deg2rad(3) 


def inside_poly(x, y, poly):
    """Test whether the point (x, y) is inside a polygon"""
    
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i+1) % n]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0)*(y - y0)/(y1 - y0) + x0):
            inside = not inside
    return inside


def quat_z(theta):
    """convert theta to (qw,qx,qy,qz)"""
    return np.cos(theta/2), 0.0, 0.0, np.sin(theta/2)


def vw_to_wheels(v, w):
    """Differential drive formula to return left and right wheel speeds"""
    
    v_r = (2*v + w*L_AXLE) / (2*R_WHEEL)
    v_l = (2*v - w*L_AXLE) / (2*R_WHEEL)
    
    return np.clip(v_l, -MAX_WSPD, MAX_WSPD), np.clip(v_r, -MAX_WSPD, MAX_WSPD)


def make_controller(model, data, qpos_index, goal_xy):
    """Simple proportional controller instead of PI controller due to ease of
    computation and faster training. Chosen Kp to be sufficiently large so that
    the steady state error for step input is sufficiently low"""

    # Current position and angle of robot  
    pos = data.qpos[qpos_index : qpos_index + 2]
    qw, qx, qy, qz = data.qpos[qpos_index + 3 : qpos_index + 7]
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    
    #The distance to the set goal
    vec  = goal_xy - pos
    dist = np.linalg.norm(vec)
    
    
    #Angle computations
    goal_head = np.arctan2(vec[1], vec[0])
    err_yaw   = np.arctan2(np.sin(goal_head - yaw), np.cos(goal_head - yaw))

    #Controller characteristics (APPROX CAN BE UPDATED LATER)
    k_v, k_w = 4.0,4.0

    #Rotation
    if abs(err_yaw) > ANGLE_TOL:
        return 0.0, k_w * err_yaw, dist

    #Moving to required position
    else:
        return k_v * dist, 0.0, dist


def pushing(model, data, joint_id_boxes, threshold=1e-1):
    """
    Returns list of (cube_name, (x, y, z)) for all cubes that are currently moving.
    Assumes each cube has 1 free joint.
    """
    moving = []
    for joint_id in joint_id_boxes:
        
        #Velocity and address of the joint
        qadr = model.jnt_qposadr[joint_id]
        v    = data.qvel[qadr:qadr+3]
        
        # Don't want it to move for any random reason
        if np.linalg.norm(v) > threshold:
            pos = data.qpos[qadr:qadr+3].copy()
            moving.append(pos)

    return moving


def quat_z_yaw(qw, qx, qy, qz) -> float:
    """Return the yaw (rotation about Z) encoded in a quaternion."""
    # yaw = atan2(2(qw qz + qx qy), 1 − 2(qy²+qz²))
    return np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))


def corners_xy(centre_xy, yaw,corners_local_coordinates) -> np.ndarray:
    """4 x 2 array with world-space (x,y) vertices of a yawed cube."""
    R = np.array([[ np.cos(yaw), -np.sin(yaw)],
                  [ np.sin(yaw),  np.cos(yaw)]])
    return centre_xy + corners_local_coordinates @ R.T
