"""
Creates a fresh MJCF file with:
- TurtleBot3 Burger (random start pose)
- N blue boxes (N from user, 10-20) with random yaw
- Randomly create enviornment type
"""

import numpy as np
from pathlib import Path
import os
import random

from benchnpin.common.utils.mujoco_utils import inside_poly, quat_z, quat_z_yaw, corners_xy

def precompute_static_vertices(keep_out, wall_thickness, room_length, room_width):
    """
    Gives list of static vertices that do not change during the simulation
    """
    # Walls vertices computation
    half_width, half_length= room_width/2, room_length/2
    X0, X1 = -half_length, half_length
    Y0, Y1 = -half_width, half_width
    t      = wall_thickness

    """Wall vertices are defined in the world frame, with a clearance"""
    wall_vertices = [
        ["wall_left",
         [(X0-t, Y0-t), (X0, Y0-t), (X0, Y1+t), (X0-t, Y1+t)]],
        ["wall_right",
         [(X1,   Y0-t), (X1+t, Y0-t), (X1+t, Y1+t), (X1, Y1+t)]],
        ["wall_bottom",
         [(X0, Y0-t), (X1, Y0-t), (X1, Y0), (X0, Y0)]],
        ["wall_top",
         [(X0, Y1), (X1, Y1), (X1, Y1+t), (X0, Y1+t)]],
    ]
    # Columns and Dividers
    # File_updating.changing_per_configuration returns 'keep_out', a list
    # of pillar footprints (each already a list of (x,y) tuples).
    # We simply wrap them with names here:
    def columns_from_keepout(keep_out):
        out = []
        for k, poly in enumerate(keep_out):
            shifted = [(x - half_length, y - half_width) for x, y in poly]
            out.append([f"Column_{k}", shifted])
        return out
    
    # Arena-shifted base polygon for corners
    base = [
        (0.0000, 0.0000),
        (0.3150, 0.0000),
        (0.1575, 0.0640),
        (0.0640, 0.1575),
        (0.0000, 0.3150)
    ]

    # helper to mirror and then arena-shift
    def shift(poly, mirror_x: bool, mirror_y: bool, half_x=half_length, half_y=half_width):
        out = []
        for x, y in poly:
            px = (2 * half_x - x) if mirror_x else x
            py = (2 * half_y - y) if mirror_y else y
            out.append((px - half_x, py - half_y))
        return out

    # three corners:  BL (0),  BR (mirror_x),  TL (mirror_y)
    corners = [
        ["Corner_BL", shift(base, False, False)],   # bottom-left (x=0, y=0)
        ["Corner_BR", shift(base, True,  False)],   # bottom-right (x=2Hx, y=0)
        ["Corner_TL", shift(base, False, True)],    # top-left   (x=0,  y=2Hy)
    ]


    return wall_vertices, columns_from_keepout(keep_out), corners


def dynamic_vertices(model, data, qpos_idx_robot: int, joint_ids_boxes: list[int], robot_full, box_half, room_length, room_width):
    """
    Returns the vertices of the robot and boxes in the world frame.
    """

    # robot vertices
    cx, cy = data.qpos[qpos_idx_robot:qpos_idx_robot+2]
    qw, qx, qy, qz = data.qpos[qpos_idx_robot+3:qpos_idx_robot+7]
    yaw = quat_z_yaw(qw, qx, qy, qz)

    local_robot = np.array([[-robot_full[0]/2, -robot_full[1]/2-0.03],
                            [ robot_full[0]/2, -robot_full[1]/2-0.03],
                            [ robot_full[0]/2,  robot_full[1]/2-0.03],
                            [-robot_full[0]/2,  robot_full[1]/2-0.03]])
    robot_vertices = ["robot",
                      corners_xy(np.array([cx, cy]), yaw, local_robot).tolist(), yaw, (cx, cy)]

    # boxes vertices
    local_box = np.array([[-box_half, -box_half],
                          [ box_half, -box_half],
                          [ box_half,  box_half],
                          [-box_half,  box_half]])

    boxes_vertices = []
    for jid in joint_ids_boxes:
        adr = model.jnt_qposadr[jid]
        bx, by = data.qpos[adr:adr+2]
        qw, qx, qy, qz = data.qpos[adr+3:adr+7]
        yaw = quat_z_yaw(qw, qx, qy, qz)
        verts = corners_xy(np.array([bx, by]), yaw, local_box).tolist()
        boxes_vertices.append([verts])

    return robot_vertices, boxes_vertices

def receptacle_vertices(receptacle_half, receptacle_local_dimension):
    """
    Returns the vertices of the receptacle in the world frame.
    """
    # Receptacle vertices
    x , y = receptacle_half

    d = receptacle_local_dimension
    local = np.array([
        [-d, -d],
        [ d, -d],
        [ d,  d],
        [-d,  d]
    ])
  
    Receptacle_vertices = corners_xy(np.array([x, y]), 0, local).tolist()

    return Receptacle_vertices

def changing_per_configuration(env_type: str, clearance_poly, 
                               ARENA_X, ARENA_Y, n_pillars, half):
    """ 
    Based on the configration, it would create code for pillars along with
    polygon to the area where nothing has to be placed.
    """

    def pillar(name, cx, cy, half):
        """Returns one pillar at centre cx,cy with half-size"""
        
        xh, yh, zh = half
        heavy_mass = 1e4
        Text = f"""
    <!-- pillar {name} -->
    <body name="{name}" pos="{cx:.3f} {cy:.3f} {zh:.3f}">
      <joint name="{name}_joint" type="free"/>
      <geom name="{name}" type="box" size="{xh:.3f} {yh:.3f} {zh:.3f}" mass="{heavy_mass:.1f}" 
      contype="1" conaffinity="1" rgba="0.647 0.165 0.165 0.4"/>
    </body>
"""
        Coordinates=[(cx-xh, cy-yh),(cx+xh, cy-yh),(cx+xh, cy+yh),(cx-xh, cy+yh)]
        return Text, Coordinates
    
    extra_xml = ""
    # list of polygons to exclude when sampling
    keep_out  = []

    def generating_centers(n_pillars, half, clearance_poly, ARENA_X, ARENA_Y):
      """
      Generates n_pillars centres within the clearance polygon
      and returns a list of centres.
      """
      centres = []
      rng = np.random.default_rng()
      attempts = 0
      while len(centres) < n_pillars and attempts < 50000:
        attempts += 1
        
        # This range is picked on clearance for robot and walls. Chosen the
        # lowest value
        cx = rng.uniform(0.600, ARENA_X[1]-0.600)
        cy = rng.uniform(0.600, ARENA_Y[1]-0.600)

        # test the 4 pillar corners
        corners = [(cx-half[0], cy-half[1]),
                    (cx+half[0], cy-half[1]),
                    (cx+half[0], cy+half[1]),
                    (cx-half[0], cy+half[1])]
        if all(inside_poly(x, y, clearance_poly) for x, y in corners):
            
          clearance = 0.30
          diameter   = 2 * half[0]
          
          # Check for clearance to other pillars
          if all(np.hypot(cx - px, cy - py) >= diameter + clearance
                  for px, py in centres):
              centres.append((cx, cy))                    
      return centres

    # small_columns
    if env_type == "small_columns":

      # Random number of pillars between 1-4
      centres=generating_centers(n_pillars, half, clearance_poly, ARENA_X, ARENA_Y)

      for k, (cx, cy) in enumerate(centres):
        xml, poly = pillar(f"small_col{k}", cx, cy, half)
        extra_xml += xml
        keep_out.append(poly)        

    # large_columns
    elif env_type == "large_columns":

        centres=generating_centers(n_pillars, half, clearance_poly, ARENA_X, ARENA_Y)

        for k, (cx, cy) in enumerate(centres):
            xml, poly = pillar(f"large_col{k}", cx, cy, half)
            extra_xml += xml
            keep_out.append(poly)

    # large_divider (one long wall)
    # NEED TO CONFIRM
    elif env_type == "large_divider":
        cx, cy = 0.975, 1.655
        half   = (0.55, 0.05, 0.30)
        xml, poly = pillar("divider", cx, cy, half)
        extra_xml += xml
        keep_out.append(poly)

    # small_empty
    elif env_type == "small_empty":
        # nothing to add
        pass

    else:
        raise ValueError("environment_type must be one of "
                         "'small_empty', 'small_columns', "
                         "'large_columns', 'large_divider'")
      
    return extra_xml, keep_out  


def intersects_keepout(x, y, keep_out):
    """To ensure that it doesn't lie in the keep out area due to pillars"""
    
    return any(inside_poly(x, y, poly) for poly in keep_out)
  

def sample_scene(n_boxes, keep_out, ROBOT_R, CLEAR, ARENA_X, ARENA_Y, clearance_poly):
    """returns robot pose + list of box poses (x,y,theta)"""

    # Robot iteration for its placement
    for _ in range(2000):
        rx = np.random.uniform(-ARENA_X[1]/2, ARENA_X[1]/2)
        ry = np.random.uniform(-ARENA_Y[1]/2, ARENA_Y[1]/2)
        if inside_poly(rx, ry, clearance_poly) and not intersects_keepout(rx, ry, keep_out):
            break
    robot_qpos = f"{rx:.4f} {ry:.4f} 0.01"

    # Boxes iteration for its placement
    boxes = []
    tries = 0
    while len(boxes) < n_boxes and tries < 10000:
        tries += 1
        x = np.random.uniform(-ARENA_X[1]/2, ARENA_X[1]/2)
        y = np.random.uniform(-ARENA_Y[1]/2, ARENA_Y[1]/2)
        if not inside_poly(x, y, clearance_poly):
            continue
        if intersects_keepout(x, y, keep_out):
            continue
        if np.hypot(x - rx, y - ry) < ROBOT_R:
            continue
        if any(np.hypot(x - cx, y - cy) < CLEAR for cx, cy, _ in boxes):
            continue
        theta = np.random.uniform(0, 2*np.pi)
        boxes.append((x, y, theta))
    if len(boxes) < n_boxes:
        print("Could only place", len(boxes), "boxes")

    return robot_qpos, boxes


def build_xml(robot_qpos, boxes, stl_model_path, extra_xml, Z_BOX, box_size, ARENA_X1, ARENA_Y1, goal_half, goal_center, adjust_num_pillars, robot_rgb, sim_timestep):
    """Building data for a different file"""

    if adjust_num_pillars is True:
        adjust_pillar_plane = f"""
    <!-- Pillar plane -->
    <body pos="-0.4 0 0">
      <geom type="box" size="0.4 {ARENA_Y1/2} 0.01" friction="0.5 0.05 0.0001"/>
    </body>
"""
    else:
        adjust_pillar_plane = ""
    header = f"""
<mujoco model="box_delivery_structured_env">
  <compiler angle="radian" autolimits="true" meshdir="{stl_model_path}"/>

  <option integrator="implicitfast" gravity="0 0 -9.81"
          timestep="{sim_timestep}" iterations="20" viscosity="1.5"/>

  <default>
    <joint limited="false" armature="0.01"/>
    <equality solref="0.0002 1" solimp="0.99 0.99 0.0001"/>
  </default>

  <asset>
    <material name="blue_mat" rgba="0.4 0.3 0.2 1"/>
    <mesh name="corner_full" file="corner_full.stl" scale="0.001 0.001 0.001"/>
    <mesh name="burger_base" file="burger_base.stl" scale="0.001 0.001 0.001"/>
    <mesh name="left_tire"   file="left_tire.stl"   scale="0.001 0.001 0.001"/>
    <mesh name="right_tire"  file="right_tire.stl"  scale="0.001 0.001 0.001"/>
    <mesh name="lds"         file="lds.stl"         scale="0.001 0.001 0.001"/>
    <mesh name="bumper"      file="TurtleBot3 Burger Bumper.STL" scale="0.001 0.001 0.001"/>
    <material name="mat_noreflect" rgba="0 0.3922 0 1" specular="0" shininess="0" reflectance="0"/>
  </asset>

  <visual>
    <quality shadowsize="4096"/>
    <headlight ambient="1 1 1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
  </visual>

  <worldbody>
  
    <!-- Floor -->
    <body pos="0 0 0">
      <geom name="floor" type="box" size="{ARENA_X1/2} {ARENA_Y1/2} 0.01" friction="0.5 0.05 0.0001" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Corner 1 -->
    <body name="corner_1" pos="{-ARENA_X1/2} {-ARENA_Y1/2} 0.05" quat="0 1 0 0">
      <geom name="corner_1" type="mesh" mesh="corner_full" rgba="0.0 0.3 1.0 1.0" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Corner 2 -->
    <body name="corner_2" pos="{ARENA_X1/2} {-ARENA_Y1/2} 0.05" quat="0 1 1 0">
      <geom name="corner_2" type="mesh" mesh="corner_full" rgba="0.0 0.3 1.0 1.0" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Corner 3 -->
    <body name="corner_3" pos="{-ARENA_X1/2} {ARENA_Y1/2} -0.01" quat="1 0 0 0">
      <geom name="corner_3" type="mesh" mesh="corner_full" rgba="0.0 0.3 1.0 1.0" contype="1" conaffinity="1"/>
    </body>
        
    <!-- Marked area -->
    <geom name="marked_area" type="box"
      pos="{goal_center[0]} {goal_center[1]} 0.01"
      size="{goal_half} {goal_half} 0.0005"
      material="mat_noreflect"
      contype="0"
      conaffinity="0"
      friction="0.5 0.05 0.0001"/>
      
    <!-- transporting area -->
    <geom name="transporting_area" type="box"
      pos="{ARENA_X1/2+0.05} {ARENA_Y1/2+0.07} 0.01"
      size="0.05 0.05 0.05"
      rgba="0.5 1 0.5 1"/>
      
    <!-- transporting area Y-walls -->
    <geom name="transporting_wall_y1" type="box"
      pos="{ARENA_X1/2+0.05} {ARENA_Y1/2+0.01} 1.0"
      size="0.05 0.01 1.0"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"/>
      
    <geom name="transporting_wall_y2" type="box"
      pos="{ARENA_X1/2+0.05} {ARENA_Y1/2+0.12} 1.0"
      size="0.05 0.01 1.0"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"/>
      
    <!-- transporting area X-walls -->
    <geom name="transporting_wall_x1" type="box"
      pos="{ARENA_X1/2-0.01} {ARENA_Y1/2+0.07} 1.0"
      size="0.01 0.05 1.0"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"/>
      
    <geom name="transporting_wall_x2" type="box"
      pos="{ARENA_X1/2+0.11} {ARENA_Y1/2+0.07} 1.0"
      size="0.01 0.05 1.0"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"/>
      
    <!-- X-walls: left and right sides -->
    <geom name="Wall_X1" type="box"
      pos="{-ARENA_X1/2-0.01} 0 0.15"
      size="0.01 {ARENA_Y1/2} 0.15"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>

    <geom name="Wall_X2" type="box"
      pos="{ARENA_X1/2+0.01} 0 0.15"
      size="0.01 {ARENA_Y1/2} 0.15"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>

    <!-- Y-walls: bottom and top -->
    <geom name="Wall_Y1" type="box"
      pos="0 {-ARENA_Y1/2-0.01} 0.15"
      size="{ARENA_X1/2} 0.01 0.15"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>

    <geom name="Wall_Y2" type="box"
      pos="0 {ARENA_Y1/2+0.01} 0.15"
      size="{ARENA_X1/2} 0.01 0.15"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>
      
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
      <geom name="base_bumper" pos="-0.04 -0.09 0.01" quat="1 0 0 0" type="mesh" rgba="0.3 0.13 0.08 1" mesh="bumper" mass="0.100" contype="1" conaffinity="1"/>
      
      <!-- Left wheel -->
      <body name="wheel_left_link" pos="0 0.08 0.033" quat="0.707388 -0.706825 0 0">
        <inertial pos="0 0 0" quat="-0.000890159 0.706886 0.000889646 0.707326" mass="0.0284989" diaginertia="2.07126e-05 1.11924e-05 1.11756e-05"/>
        <joint name="wheel_left_joint" pos="0 0 0" axis="0 0 1"/>
        <geom quat="0.707388 0.706825 0 0" type="mesh" rgba="{robot_rgb[0]+0.1} {robot_rgb[1]+0.1} {robot_rgb[2]+0.1} 1" mesh="left_tire" friction="1.2 0.01 0.001"/>
      </body>
      
      <!-- Right wheel -->
      <body name="wheel_right_link" pos="0 -0.08 0.033" quat="0.707388 -0.706825 0 0">
        <inertial pos="0 0 0" quat="-0.000890159 0.706886 0.000889646 0.707326" mass="0.0284989" diaginertia="2.07126e-05 1.11924e-05 1.11756e-05"/>
        <joint name="wheel_right_joint" pos="0 0 0" axis="0 0 1"/>
        <geom quat="0.707388 0.706825 0 0" type="mesh" rgba="{robot_rgb[0]+0.1} {robot_rgb[1]+0.1} {robot_rgb[2]+0.1} 1" mesh="right_tire" friction="1.2 0.01 0.001"/>
      </body>
    </body>

    {generate_waypoint_sites(50)}
      
"""
    
    #Data to be written for boxes
    box_xml = ""
    for i, (x, y, th) in enumerate(boxes):
        qw, qx, qy, qz = quat_z(th)
        box_xml += f"""
    <body name="box{i}" pos="{x:.4f} {y:.4f} {Z_BOX:.3f}">
      <joint name="box{i}_joint" type="free" />
      <geom type="box" size="{box_size}" material="blue_mat" mass="0.075"
            quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}" friction="0.4 0.015 0.002" contype="1" conaffinity="1"/>
    </body>"""

        
    #Data to be written for footers
    footer = f"""
  </worldbody>

  <!-- Actuator -->
  <actuator>
    <velocity name="wheel_left_actuator" ctrllimited="true" ctrlrange="-22.0 22.0" gear="1" kv="1.0" joint="wheel_left_joint" />
    <velocity name="wheel_right_actuator" ctrllimited="true" ctrlrange="-22.0 22.0" gear="1" kv="1.0" joint="wheel_right_joint" />
  </actuator>
</mujoco>
"""
    return header + box_xml + extra_xml +adjust_pillar_plane+ footer

def clearance_poly_generator(ARENA_X, ARENA_Y):
    """ Returns a polygon within which all items must be placed, with a clearance """
    return [
        (-ARENA_X[1]/2 + 0.100, -ARENA_Y[1]/2 + 0.300), (-ARENA_X[1]/2 + 0.315, -ARENA_Y[1]/2 + 0.300),
        (-ARENA_X[1]/2 + 0.415, -ARENA_Y[1]/2 + 0.100), (ARENA_X[1]/2 - 0.300, -ARENA_Y[1]/2 + 0.100),
        (ARENA_X[1]/2 - 0.300, -ARENA_Y[1]/2 + 0.300), (ARENA_X[1]/2 - 0.100, -ARENA_Y[1]/2 + 0.300),
        (ARENA_X[1]/2 - 0.100, ARENA_Y[1]/2 - 0.300), (ARENA_X[1]/2 - 0.300, ARENA_Y[1]/2 - 0.300),
        (ARENA_X[1]/2 - 0.300, ARENA_Y[1]/2 - 0.045), (-ARENA_X[1]/2 + 0.415, ARENA_Y[1]/2 - 0.100),
        (-ARENA_X[1]/2 + 0.415, ARENA_Y[1]/2 - 0.300), (-ARENA_X[1]/2 + 0.100, ARENA_Y[1]/2 - 0.300)
    ]

def generate_boxDelivery_xml(N,env_type,file_name,ROBOT_clear,CLEAR,Z_BOX,ARENA_X,ARENA_Y,
                  box_half_size, goal_half, goal_center,num_pillars, pillar_half, adjust_num_pillars,sim_timestep):
    
    # Name of input and output file otherwise set to default
    XML_OUT = Path(file_name)
    stl_model_path = os.path.join(os.path.dirname(__file__), 'models/')
    
    # Clearnaces and box sizes
    clearance_poly= clearance_poly_generator(ARENA_X, ARENA_Y)
    box_size = f"{box_half_size} {box_half_size} {box_half_size}"
    
    # Changing based on configration type
    extra_xml, keep_out = changing_per_configuration(env_type,clearance_poly, ARENA_X,ARENA_Y, num_pillars, pillar_half)
    
    # Finding the robot's q_pos and boxes's randomized data
    robot_qpos, boxes = sample_scene(N,keep_out,ROBOT_clear,CLEAR,ARENA_X,ARENA_Y, clearance_poly)
  
    # Building new environemnt and writing it down
    xml_string = build_xml(robot_qpos, boxes,stl_model_path,extra_xml,Z_BOX, box_size,ARENA_X[1],ARENA_Y[1], goal_half, goal_center, adjust_num_pillars, robot_rgb=(0.1, 0.1, 0.1),sim_timestep=0.01)
    XML_OUT.write_text(xml_string)
    
    return XML_OUT, keep_out, clearance_poly


def generate_waypoint_sites(num_sites=50):
    site_template = (
        '<site name="wp{0}" pos="0 0 5" size="0.02" rgba="0 1 0 1" type="sphere"/>'
    )
    return "\n".join([site_template.format(i) for i in range(num_sites)])


def transport_box_from_recept(model, data, joint_id_boxes, ARENA_X1, ARENA_Y1, goal_half, goal_center, box_half_size):
    """Teleport a box only if all its vertices are inside the goal box."""
    
    initial_len = len(joint_id_boxes)
    # half-edge of box
    HSIZE = box_half_size
    box_radius = box_half_size * np.sqrt(2)
    # local corner coordinates
    corners_local_coordinates = np.array([
        [-HSIZE, -HSIZE],
        [ HSIZE, -HSIZE],
        [ HSIZE,  HSIZE],
        [-HSIZE,  HSIZE]
    ])
    
    GOAL_HALF   = np.array([goal_half, goal_half])
    DROP_POS    = np.array([ARENA_X1/2+0.05, ARENA_Y1/2+0.07, 1.0])
    
    xmin, xmax = goal_center[0] - GOAL_HALF[0], goal_center[0] + GOAL_HALF[0]
    ymin, ymax = goal_center[1] - GOAL_HALF[1], goal_center[1] + GOAL_HALF[1]

    for jid in joint_id_boxes[:]:

        qadr = model.jnt_qposadr[jid]

        # joint centre (x,y) and quaternion
        centre_xy = data.qpos[qadr : qadr+2].copy()
        qw, qx, qy, qz = data.qpos[qadr+3 : qadr+7]

        # Sometimes the box gets squished against the wall
        if np.abs(centre_xy[0]) + box_radius > ARENA_X1/2:
            centre_xy[0] = np.sign(centre_xy[0]) * (ARENA_X1/2 - box_radius)
        if np.abs(centre_xy[1]) + box_radius > ARENA_Y1/2:
            centre_xy[1] = np.sign(centre_xy[1]) * (ARENA_Y1/2 - box_radius)

        # yaw from quaternion
        yaw = quat_z_yaw(qw, qx, qy, qz)

        # world vertices
        verts = corners_xy(centre_xy, yaw, corners_local_coordinates)

        # containment test – every vertex must satisfy the four inequalities
        inside = np.all((xmin <= verts[:,0]) & (verts[:,0] <= xmax) &
                        (ymin <= verts[:,1]) & (verts[:,1] <= ymax))

        if inside:
            data.qpos[qadr:qadr+3] = DROP_POS
            joint_id_boxes.remove(jid)

    # number of boxes that are transported
    final_len = len(joint_id_boxes)
    num_boxes = initial_len - final_len
    
    return joint_id_boxes, num_boxes
