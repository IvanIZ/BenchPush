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


# def make_controller(model, data, qpos_index, goal_xy):
def make_controller(curr_pos, curr_yaw, goal_pos):
    """Simple proportional controller instead of PI controller due to ease of
    computation and faster training. Chosen Kp to be sufficiently large so that
    the steady state error for step input is sufficiently low"""

    # Current position and angle of robot  
    # pos = data.qpos[qpos_index : qpos_index + 2]
    # qw, qx, qy, qz = data.qpos[qpos_index + 3 : qpos_index + 7]
    # yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    
    # The distance to the set goal
    # vec  = goal_xy - pos
    vec = goal_pos - curr_pos
    dist = np.linalg.norm(vec)
    
    
    # Angle computations
    goal_head = np.arctan2(vec[1], vec[0])
    err_yaw   = np.arctan2(np.sin(goal_head - curr_yaw), np.cos(goal_head - curr_yaw))

    # Controller characteristics (APPROX CAN BE UPDATED LATER)
    # k_v, k_w = 4.0,4.0
    k_v, k_w = 0.2, 8.0

    # Rotation
    if abs(err_yaw) > ANGLE_TOL:
        return 0.0, k_w * err_yaw, dist

    # # Moving to required position
    # else:
        # return k_v * dist, 0.0, dist
    return k_v, k_w * err_yaw, dist


def pushing(model, data, joint_id_boxes, threshold=1e-1):
    """
    Returns list of (box_name, (x, y, z)) for all boxes that are currently moving.
    Assumes each box has 1 free joint.
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
    """4 x 2 array with world-space (x,y) vertices of a yawed box."""
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


def get_body_vel(model, data, body_name):
    """
    Get the 6 Dof velocity of a given mujoco body
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # Get the 6D velocity: first 3 = angular, last 3 = linear
    body_velocity = data.cvel[body_id]

    angular_velocity = body_velocity[:3]
    linear_velocity = body_velocity[3:]

    return linear_velocity, angular_velocity


def get_box_2d_vertices(model, data, body_name):
        """
        Get the vertices and position in world coordinate of a box geometry
        NOTE this function assumes that the body only has one geometry. Suitable for checking boxes and ice floes
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
        return corners_world, pos  # shape (4, 2), (2, )


def get_box_2d_area(model, data, body_name):
    """
    Get the vertices in world coordinate of a box geometry
    NOTE this function assumes that the body only has one geometry. Suitable for checking boxes and ice floes
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

    return (2 * w) * (h * 2)


def polygon_from_vertices(vertices_2d) -> sg.Polygon:
    """
    Build a Shapely polygon from a user-supplied list/ndarray of 2-D points.
    The list **must** describe a *simple* five-vertex outline (no holes).
    """
    poly = sg.Polygon(vertices_2d).convex_hull           # guarantee convexity
    if not poly.is_valid:
        raise ValueError("Invalid polygon shape")
    return so.orient(poly, sign=1.0)


def extrude_and_export(
        poly: sg.Polygon,
        h_min: float = 0.2,
        h_max: float = 1.0,
        filename: str = "ice_floe.stl",
        seed: int | None = None
) -> None:
    """
    Extrude a 2-D polygon into a 3-D ‘ice floe’ whose thickness varies
    point-to-point on the upper surface.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        The (convex) planform of the floe, in metres.
    h_min, h_max : float
        Minimum and maximum thicknesses to sample [m].
    filename : str
        Path for the binary STL that will be written (over-written if it exists).
    seed : int | None
        Fix the RNG seed for full reproducibility.
    """
    if not poly.is_valid or poly.is_empty:
        raise ValueError("Input polygon is invalid or empty")

    if seed is not None:
        np.random.seed(seed)

    # 1. Extrude once to the *maximum* possible thickness
    base_mesh = trimesh.creation.extrude_polygon(poly, height=h_max)

    # 2. Identify vertices that belong to the *top* cap
    #
    # `extrude_polygon` puts the mid-plane at z = 0, so the top cap
    # initially sits at +h_max/2.  Numerical noise → use a small tolerance.
    z_top      = base_mesh.vertices[:, 2].max()
    top_mask   = np.abs(base_mesh.vertices[:, 2] - z_top) < 1e-6
    n_top_verts = top_mask.sum()

    # 3. Draw a random thickness for *each* top-cap vertex.
    #    Their z-coordinate becomes that thickness (bottom is 0).
    random_thick = np.random.uniform(h_min, h_max, size=n_top_verts)
    base_mesh.vertices[top_mask, 2] = random_thick

    # 4. Slide the whole mesh down so that the bottom sits exactly at z = 0
    base_mesh.vertices[:, 2] -= base_mesh.vertices[:, 2].min()

    # 5. Export as binary STL (over-write OK)
    base_mesh.export(filename)


def wall_collision(data, model):
    """
    Check if the robot collides with a wall
    NOTE: assume all robot geoms names start with "robot", and all wall geoms names start with "wall"
    """
            
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2

        # Get geom names from name buffer
        name1 = model.geom(g1).name
        name2 = model.geom(g2).name

        if (name1.startswith("robot") and name2.startswith("wall")) or \
                (name2.startswith("robot") and name1.startswith("wall")):
                return True
    return False


def zero_body_velocity(model, data, body_name):
    """
    Set the velocity of body with the given body_name to zero
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    dof_start = model.body_dofadr[body_id]
    dof_count = model.body_dofnum[body_id]
    data.qvel[dof_start : dof_start + dof_count] = 0
    data.qacc[dof_start : dof_start + dof_count] = 0
