import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import trimesh
try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e

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


def get_body_pose_2d(model, data, body_name):
    """
    Get (x, y, theta) pose of a 2D body
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # Get world position (x, y)
    pos = data.xpos[body_id][:2]

    # Extract yaw angle (theta) from quaternion
    quat = data.xquat[body_id]
    siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
    cosy_cosp = 1 - 2 * (quat[2]**2 + quat[3]**2)
    theta = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([pos[0], pos[1], theta])


def get_box_2d_vertices(model, data, body_name):
        """
        Get the vertices in world coordinate of a box geometry
        NOTE this function assumes that the body only has one geometry. Suitable for checking cubes and ice floes
        """
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # World position of the box (x, y)
        pos = data.xpos[body_id][:2]  # take only x, y

        # Find the first box geom belonging to this body
        geom_id = None
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == body_id and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                geom_id = i
                break
        if geom_id is None:
            raise ValueError(f"No box geom found for body '{body_name}'")

        # Extract box half-sizes
        w, h = model.geom_size[geom_id][:2]  # use x, y half-lengths
        # print("id is: ", geom_id, "; size is: ", w, h)

        # Orientation (yaw) from the body’s quaternion
        # Convert quaternion to yaw angle
        quat = data.xquat[body_id]
        siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
        cosy_cosp = 1 - 2 * (quat[2]**2 + quat[3]**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Half sizes
        corners_local = np.array([
            [-w, -h],
            [-w,  h],
            [ w,  h],
            [ w, -h]
        ])

        # Rotation matrix
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])

        # Rotate and translate to world frame
        corners_world = corners_local @ R.T + pos
        return corners_world  # shape (4, 2)


def polygon_from_vertices(vertices_2d) -> sg.Polygon:
    """
    Build a Shapely polygon from a user-supplied list/ndarray of 2-D points.
    The list **must** describe a *simple* five-vertex outline (no holes).
    """
    poly = sg.Polygon(vertices_2d).convex_hull           # guarantee convexity
    if not poly.is_valid:
        raise ValueError("Invalid polygon shape")
    return so.orient(poly, sign=1.0)


def extrude_and_export(poly: sg.Polygon,
                       thickness: float = 0.2,
                       filename: str = 'ice_stl') -> None:
    """
    Extrude the 2-D polygon to the given `thickness` and write a binary STL.
    """
    mesh = trimesh.creation.extrude_polygon(poly, height=thickness)

    # Move the base so that Z = 0 is the *bottom* rather than the mid-plane.
    mesh.apply_translation([0, 0, thickness / 2])

    # Binary STL; overwrite if it exists
    mesh.export(filename)
