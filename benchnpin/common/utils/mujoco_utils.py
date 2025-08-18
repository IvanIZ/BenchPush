import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import trimesh
import math
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
    
    # Controller characteristics
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

def xy_positions_of_wheel_assembly_wrt_center(yaw,corners_local_coordinates) -> np.ndarray:
    """4 x 2 array with world-space (x,y) vertices of a yawed box."""
    R = np.array([[ np.cos(yaw), -np.sin(yaw)],
                  [ np.sin(yaw),  np.cos(yaw)]])
    return 0.8 * corners_local_coordinates @ R.T


def generating_box_xml(boxes, Z_BOX, wheels_on_boxes, wheels_mass, wheels_support_mass, wheels_sliding_friction, 
    wheels_torsional_friction, wheels_rolling_friction, wheels_support_damping_ratio, box_mass, box_sliding_friction, 
    box_torsional_friction, box_rolling_friction, box_half_size, num_boxes_with_wheels, wheels_axle_damping_ratio):

    box_xml = " <!-- Boxes -->\n"
    box_size = f"{box_half_size} {box_half_size} {box_half_size}"

    if not wheels_on_boxes:
        num_boxes_without_wheels = len(boxes)
    else:
        num_boxes_without_wheels = len(boxes) - num_boxes_with_wheels
    
    box_number=0
        
    while num_boxes_without_wheels > 0:

        Z_BOX = Z_BOX + 0.005
        (x, y, th)= boxes[box_number]

        qw, qx, qy, qz = quat_z(th)

        box_xml += f"""
    <body name="box{box_number}" pos="{x:.4f} {y:.4f} {Z_BOX:.3f}">
      <joint name="box{box_number}_joint" type="free" />
      <geom type="box" size="{box_size}" material="blue_mat" mass="{box_mass:.2f}"
            quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}" friction="{box_sliding_friction:.2f} {box_torsional_friction:.2f} {box_rolling_friction:.2f}" contype="1" conaffinity="1"/>
    </body>"""

        box_number += 1

        num_boxes_without_wheels -= 1

    if not wheels_on_boxes:
        return box_xml
      
    Z_BOX = Z_BOX + 0.030
    Z_wheel_support, Z_wheel = -0.0530, 0.0038

    # Wheel support position relative to wheel support assembly
    Wheel_support_assembly_pos = [-0.008,0.0093,0]

    # Wheel assembly positio relative to wheel support assembly
    Wheel_assembly_pos = [0,-0.0012,Z_wheel]
    # Wheel position relative to wheel assembly
    Wheel_pos = [-0.0016,-0.007,-0.007]

    def caster_block(i, idx, px, py):
      sx, sy, sz = Wheel_support_assembly_pos
      ax, ay, az = Wheel_assembly_pos
      wx, wy, wz = Wheel_pos
      return f"""
      <body name="Wheels assembly_{i}_{idx}" pos="{px:.3f} {py:.2f} {Z_wheel_support:.4f}">
      
        <joint name="Wheel_support_rotation_{i}_{idx}" type="hinge" axis="0 0 1" damping="{wheels_support_damping_ratio}"/>
        <geom  name="Wheel_support_{i}_{idx}" type="mesh" mesh="Wheels_support" rgba="0.3 0.13 0.08 1"
               contype="1" conaffinity="1" euler="1.5708 0 0" pos="{sx:.3f} {sy:.4f} {sz:.0f}" mass="{wheels_support_mass}"/>
        
        <body name="Wheels_actual_{i}_{idx}" pos="{ax:.1f} {ay:.4f} {az:.4f}">
          <joint name="Wheels_actual_joint_{i}_{idx}" type="hinge" axis="1 0 0" damping="{wheels_axle_damping_ratio}"/>
          <geom  name="Wheel_{i}_{idx}" type="mesh" mesh="Wheels" rgba="0.1 0.1 0.1 1"
                 contype="1" conaffinity="1" pos="{wx:.4f} {wy:.3f} {wz:.3f}"
                 friction="{wheels_sliding_friction:.2f} {wheels_torsional_friction:.2f} {wheels_rolling_friction:.2f}"
                 mass="{wheels_mass}"/>
        </body>
      </body>"""

    boxes_with_wheels = boxes[box_number:]


    for i, (x, y, th) in enumerate(boxes_with_wheels, start=box_number):
        
        qw, qx, qy, qz = quat_z(th)
        yaw = 2 * math.atan2(qz, qw)
  
        corners_local_coordinates = np.array([
            [-box_half_size, -box_half_size],
            [-box_half_size,  box_half_size],
            [ box_half_size,  box_half_size],
            [ box_half_size, -box_half_size]
        ])

        # Wheel positions relative to box center
        Wheel_support_pos = xy_positions_of_wheel_assembly_wrt_center(yaw,corners_local_coordinates)

        # Box header
        box_xml += f"""
    <body name="box{i}" pos="{x:.4f} {y:.4f} {Z_BOX:.3f}">
      <joint name="box{i}_joint" type="free" />
      <geom type="box" size="{box_size}" rgba="0.36 0.20 0.09 1" mass="{box_mass:.2f}"
            quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}"
            friction="{box_sliding_friction:.2f} {box_torsional_friction:.2f} {box_rolling_friction:.2f}"
            contype="1" conaffinity="1"/>"""

        # Four identical wheel assemblies (kept same order and names)
        for idx, (px, py) in enumerate(Wheel_support_pos, start=1):
            box_xml += caster_block(i, idx, px, py)

        # Close box body
        box_xml += """
    </body>"""


    return box_xml

def generating_agent_xml(agent_type, bumper_type, bumper_mass, robot_qpos, robot_rgb=(0.1, 0.1, 0.1)):

    if agent_type == "turtlebot_3":
        if bumper_type == 'curved_inwards':
            bumper_name= "TurtleBot3_Curved_Bumper_base"
        
        elif bumper_type == 'straight':
            bumper_name = "TurtleBot3_Straight_Bumper_base"
        
        elif bumper_type == 'curved_outwards':
            bumper_name = "TurtleBot3_Triangular_Bumper_base"
        
        else:
            raise ValueError("Invalid bumper type for Turtlebot3. Choose from 'curved_inwards', 'straight', or 'curved_outwards'.")

        agent_xml = f"""   <!-- Turtlebot3 robot -->\n
    <!-- robot -->
    <body name="base" pos="{robot_qpos}" euler="0 0 3.141592653589793">
      <joint type="free" name="base_joint"/>
      <!-- chassis -->
      <geom name="base_chasis" pos="-0.032 0 0.01" type="mesh" rgba="{robot_rgb[0]} {robot_rgb[1]} {robot_rgb[2]} 1" mesh="burger_base" friction="0.1 0.02 0.0001" mass="0.8" contype="1" conaffinity="1"/>
      <!-- small box sensor -->
      <geom name="base_sensor_1" size="0.015 0.0045 0.01" pos="-0.081 7.96327e-07 0.005" quat="0.707388 -0.706825 0 0" type="box" rgba="{robot_rgb[0]} {robot_rgb[1]} {robot_rgb[2]} 1" mass="0.05" contype="1" conaffinity="1"/>
      <!-- LDS sensor -->
      <geom name="base_sensor_2" pos="-0.032 0 0.182" quat="1 0 0 0" type="mesh" rgba="{robot_rgb[0]} {robot_rgb[1]} {robot_rgb[2]} 1" mesh="lds" mass="0.131" contype="1" conaffinity="1"/>
      <!-- Bumper -->
      <geom name="{bumper_name}" pos="-0.04 -0.09 0.01" quat="1 0 0 0" type="mesh" rgba="0.3 0.13 0.08 1" mesh="{bumper_name}" mass="{bumper_mass}" contype="1" conaffinity="1" friction="0.1 0.02 0.0001"/>
      
      <!-- Left wheel -->
      <body name="wheel_left_link_base" pos="0 0.08 0.033" quat="0.707388 -0.706825 0 0">
        <inertial pos="0 0 0" quat="-0.000890159 0.706886 0.000889646 0.707326" mass="0.0284989" diaginertia="2.07126e-05 1.11924e-05 1.11756e-05"/>
        <joint name="wheel_left_joint" pos="0 0 0" axis="0 0 1"/>
        <geom quat="0.707388 0.706825 0 0" type="mesh" rgba="{robot_rgb[0]+0.1} {robot_rgb[1]+0.1} {robot_rgb[2]+0.1} 1" mesh="left_tire" friction="1.2 0.01 0.001"/>
      </body>
      
      <!-- Right wheel -->
      <body name="wheel_right_link_base" pos="0 -0.08 0.033" quat="0.707388 -0.706825 0 0">
        <inertial pos="0 0 0" quat="-0.000890159 0.706886 0.000889646 0.707326" mass="0.0284989" diaginertia="2.07126e-05 1.11924e-05 1.11756e-05"/>
        <joint name="wheel_right_joint" pos="0 0 0" axis="0 0 1"/>
        <geom quat="0.707388 0.706825 0 0" type="mesh" rgba="{robot_rgb[0]+0.1} {robot_rgb[1]+0.1} {robot_rgb[2]+0.1} 1" mesh="right_tire" friction="1.2 0.01 0.001"/>
      </body>
    </body>"""
        actuator_xml = f"""
  <!-- Actuator -->
  <actuator>
    <velocity name="wheel_left_actuator" ctrllimited="true" ctrlrange="-22.0 22.0" gear="1" kv="1.0" joint="wheel_left_joint" />
    <velocity name="wheel_right_actuator" ctrllimited="true" ctrlrange="-22.0 22.0" gear="1" kv="1.0" joint="wheel_right_joint" />
  </actuator>
</mujoco>"""

    elif agent_type == "jackal":

        if bumper_type == 'curved_inwards':
            bumper_name= "jackal_bumper_curved_inwards"
            bumper_pos = "-0.2075 0.026 -0.283"
        
        elif bumper_type == 'straight':
            bumper_name = "jackal_straight_bumper"
            bumper_pos = "-0.2075 0.025 -0.252"
        
        elif bumper_type == 'curved_outwards':
            bumper_name = "jackal_bumper_curved"
            bumper_pos = "-0.2075 0.026 -0.283"
        
        else:
            raise ValueError("Invalid bumper type for Jackal. Choose from 'curved_inwards', 'straight', or 'curved_outwards'.")
        
        robot_qpos_x, robot_qpos_y = robot_qpos.split(" ")[0:2]

        agent_xml = f"""   <!-- Jackal robot -->\n
    <!-- robot -->
    <body name="jackal_base" pos="{robot_qpos_x} {robot_qpos_y} 0.015" euler="0 1.5707963267948966 1.5707963267948966">
    
      <joint type="free" name="base_joint_jackal"/>
      
      <!-- chassis -->
      <geom name="jackal_base" pos="0 0 0" type="mesh" rgba="0.08 0.08 0.09 1" mesh="jackal_base" friction="0.1 0.02 0.0001" contype="1" conaffinity="1" mass="14.0"/>
      
      <!-- Fender -->
      <geom name="jackal_fenders" pos="0 0 0" type="mesh" rgba="0.702 0.467 0.090 1" mesh="jackal_fenders" mass="0.5" contype="1" conaffinity="1" friction="0.1 0.02 0.0001"/>
      
      <!-- Antenna bracket -->
      <geom name="antenna_bracket" pos="0 0.272 0.19" type="mesh" rgba="0.702 0.467 0.090 1" mesh="antenna_bracket" mass="0.2" contype="1" conaffinity="1" euler="-1.5707963267948966 0 1.5707963267948966"/>
      
      <!-- Antenna 1 -->
      <geom name="Antenna_1" pos="0.05 0.265 0.190" type="mesh" rgba="0.1 0.1 0.1 1" mesh="Antenna" mass="0.13" contype="1" conaffinity="1" euler="0.8 1.5707963267948966 0"/>
      
      <!-- Antenna 2 -->
      <geom name="Antenna_2" pos="-0.07 0.265 0.190" type="mesh" rgba="0.1 0.1 0.1 1" mesh="Antenna" mass="0.13" contype="1" conaffinity="1" euler="0.8 1.5707963267948966 0"/>
      
      <!-- Bumper -->
      <geom name="{bumper_name}" pos="{bumper_pos}" type="mesh" rgba="0.3 0.13 0.08 1" mesh="{bumper_name}" mass="{bumper_mass}" contype="1" conaffinity="1" euler="-1.5707963267948966 0.0 -1.5707963267948966"/>
      
      <!-- Left wheel front -->
      <body name="wheel_left_front" pos="-0.19 0.1 -0.13" euler="0 1.5707963267948966 1.5707963267948966">
        
        <site name="wheel_1" pos="0 0 0" size="0.02" type="sphere"/>
        
        <joint name="wheel_left_jackal" axis="0 0 -1"/>
        <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="jackal_wheel" friction="1.2 0.01 0.001" mass="0.5"/>
      </body>
      
      <!-- Left wheel back -->
      <body name="wheel_left_back" pos="-0.19 0.1 0.13" euler="0 1.5707963267948966 1.5707963267948966">
        
        <site name="wheel_2" pos="0 0 0" size="0.02" type="sphere"/>
        <joint name="wheel_left_jackal_back" axis="0 0 -1"/>
        <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="jackal_wheel" friction="1.2 0.01 0.001" mass="0.5"/>
      </body>
      
      <!-- Right wheel front -->
      <body name="wheel_right_front" pos="0.19 0.1 -0.13" euler="0 1.5707963267948966 1.5707963267948966">
        
        <site name="wheel_3" pos="0 0 0" size="0.02" type="sphere"/>
        <joint name="wheel_right_jackal" axis="0 0 -1"/>
        <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="jackal_wheel" friction="1.2 0.01 0.001" mass="0.5"/>
        
      </body>
      
      <!-- Right wheel back -->
      <body name="wheel_right_back" pos="0.19 0.1 0.13" euler="0 1.5707963267948966 1.5707963267948966">
        
        <site name="wheel_4" pos="0 0 0" size="0.02" type="sphere"/>
        <joint name="wheel_right_jackal_back" axis="0 0 -1"/>
        <geom type="mesh" rgba="0.1 0.1 0.1 1" mesh="jackal_wheel" friction="1.2 0.01 0.001" mass="0.5"/>
        
      </body>
      
    </body>"""
        actuator_xml = f"""
  <!-- Actuator -->
  <actuator>
    <velocity name="wheel_left_actuator" ctrllimited="true" ctrlrange="-2.2 2.2" gear="1" kv="1.0" joint="wheel_left_jackal_back" />
    <velocity name="wheel_right_actuator" ctrllimited="true" ctrlrange="-2.2 2.2" gear="1" kv="1.0" joint="wheel_right_jackal_back" />
  </actuator>
</mujoco>"""
    else:
        raise ValueError("Only turtlebot_3 and jackal are supported agents")
    return agent_xml, actuator_xml

def large_divider_corner_vertices(cx, cy, hy, half_x):
    """
    Return two corner footprints as lists of (x,y) vertices, same format as your earlier 'corners'
    """
    corner_coordinates = [
    (0.0000, 0.0000),
    (0.3150, 0.0000),
    (0.1575, 0.0640),
    (0.0640, 0.1575),
    (0.0000, 0.3150)]

    def place_on_right_wall(y_anchor):
        # Mirror across X to the right wall (x -> half_x - x) and shift up to y_anchor (y -> y_anchor + y)
        return [(half_x - x, y_anchor + y) for (x, y) in corner_coordinates]
    
    def place_on_right_wall_mirror_y(y_anchor):
        return [(half_x - x, y_anchor - y) for (x, y) in corner_coordinates]

    corner_4 = place_on_right_wall(cy + hy)
    corner_5 = place_on_right_wall_mirror_y(cy - hy)

    return corner_4, corner_5


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
