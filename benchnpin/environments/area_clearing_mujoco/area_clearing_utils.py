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

def precompute_static_vertices(keep_out, room_width, room_length):
    """
    Gives list of static vertices that do not change during the simulation
    """
    # Walls vertices computation
    half_width, half_length= room_width/2, room_length/2
    X0, X1 = -half_width, half_width
    Y0, Y1 = -half_length, half_length
    t      = 4.0

    """Wall vertices are defined in the world frame, with a clearance"""
    Wall_vertices = [
        ["Wall_left",
         [(X0-t, Y0-t), (X0, Y0-t), (X0, Y1+t), (X0-t, Y1+t)]],
        ["Wall_right",
         [(X1,   Y0-t), (X1+t, Y0-t), (X1+t, Y1+t), (X1, Y1+t)]],
        ["Wall_bottom",
         [(X0, Y0-t), (X1, Y0-t), (X1, Y0), (X0, Y0)]],
        ["Wall_top",
         [(X0, Y1), (X1, Y1), (X1, Y1+t), (X0, Y1+t)]],
    ]
    # Columns and Dividers
    # File_updating.changing_per_configuration returns 'keep_out', a list
    # of pillar footprints (each already a list of (x,y) tuples).
    # We simply wrap them with names here:
    def columns_from_keepout(keep_out):
        out = []
        for k, poly in enumerate(keep_out):
            shifted = [(x - half_width, y - half_length) for x, y in poly]
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
    def shift(poly, mirror_x: bool, mirror_y: bool, half_x=half_width, half_y=half_length):
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
        ["Corner_TL", shift(base, True, True)],    # top-left   (x=0,  y=2Hy)
    ]


    return Wall_vertices, columns_from_keepout(keep_out), corners


def dynamic_vertices(model,
                     data,
                     qpos_idx_robot: int,
                     joint_ids_boxes: list[int],
                     robot_half   =(0.069, 0.0915),
                     box_half    =0.04):
    """
    Returns the vertices of the robot and boxes in the world frame.
    """

    # robot vertices
    cx, cy = data.qpos[qpos_idx_robot:qpos_idx_robot+2]
    cx, cy = cx-0.7875, cy-1.4225
    qw, qx, qy, qz = data.qpos[qpos_idx_robot+3:qpos_idx_robot+7]
    yaw  = quat_z_yaw(qw, qx, qy, qz)

    local_robot = np.array([[-robot_half[0], -robot_half[1]],
                            [ robot_half[0], -robot_half[1]],
                            [ robot_half[0],  robot_half[1]],
                            [-robot_half[0],  robot_half[1]]])
    Robot_vertices = ["robot",
                      corners_xy(np.array([cx, cy]), yaw, local_robot).tolist(),yaw,(cx, cy)]

    # boxes vertices
    local_box = np.array([[-box_half, -box_half],
                           [ box_half, -box_half],
                           [ box_half,  box_half],
                           [-box_half,  box_half]])

    Boxes_vertices = []
    for jid in joint_ids_boxes:
        adr   = model.jnt_qposadr[jid]
        bx, by = data.qpos[adr:adr+2]
        bx, by = bx-0.7875, by-1.4225
        qw, qx, qy, qz = data.qpos[adr+3:adr+7]
        yaw  = quat_z_yaw(qw, qx, qy, qz)
        verts = corners_xy(np.array([bx, by]), yaw, local_box).tolist()
        Boxes_vertices.append([verts])

    return Robot_vertices, Boxes_vertices

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
                               ARENA_X, ARENA_Y, n_pillars, pillar_half, internal_clearance_length):
    """ 
    Based on the configration, it would create code for pillars along with
    polygon to the area where nothing has to be placed.
    """

    def pillar(name, cx, cy, half):
        """Returns one pillar at centre cx,cy with half-size"""
        
        xh, yh, zh = pillar_half
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

    # Partially_closed_with_walls
    if env_type == "Partially_closed_with_static":

      cx = ARENA_X[1] / 2.0

      # lower and upper Y bounds
      y_min = internal_clearance_length + half[1] / 2.0
      y_max = ARENA_Y[1] - half[1] / 2.0 - internal_clearance_length

      # evenly space n_pillars points between y_min and y_max
      if n_pillars == 1:
        centers = [(cx, (y_min + y_max) / 2.0)]
      step = (y_max - y_min) / (n_pillars - 1)
      centers = [(cx, y_min + i * step) for i in range(n_pillars)]

      for k, (cx, cy) in enumerate(centers):
        xml, poly = pillar(f"small_col{k}", cx, cy, half)
        extra_xml += xml
        keep_out.append(poly)
      
    # small_empty
    elif env_type == "Fully_open_area" or env_type == "Partially_closed_with_walls":
        # nothing to add
        pass

    else:
        raise ValueError("environment_type must be one of "
                         "'Fully_open_area', 'Partially_closed_with_walls', "
                         "'Partially_closed_with_static'")
      
    return extra_xml, keep_out  


def intersects_keepout(x, y, keep_out):
    """To ensure that it doesn't lie in the keep out area due to pillars"""
    
    return any(inside_poly(x, y, poly) for poly in keep_out)
  

def sample_scene(n_boxes, keep_out, ROBOT_R, BOXES_clear, ARENA_X, ARENA_Y, internal_clearance_length):
    """returns robot pose + list of box poses (x,y,theta)"""

    # compute inset bounds once
    xmin = ARENA_X[0] + internal_clearance_length
    xmax = ARENA_X[1] - internal_clearance_length
    ymin = ARENA_Y[0] + internal_clearance_length
    ymax = ARENA_Y[1] - internal_clearance_length

    # Robot iteration for its placement
    for _ in range(2000):
        rx = np.random.uniform(xmin, xmax)
        ry = np.random.uniform(ymin, ymax)
        if not intersects_keepout(rx, ry, keep_out):
            break
    robot_qpos = f"{rx:.4f} {ry:.4f} 0.01"

    # Boxes iteration for its placement
    boxes = []
    tries = 0
    while len(boxes) < n_boxes and tries < 10000:
        tries += 1
        x = np.random.uniform(xmin, xmax)     # ← changed
        y = np.random.uniform(ymin, ymax)     # ← changed
        if intersects_keepout(x, y, keep_out):
            continue
        if np.hypot(x - rx, y - ry) < ROBOT_R:
            continue
        if any(np.hypot(x - cx, y - cy) < BOXES_clear for cx, cy, _ in boxes):
            continue
        theta = np.random.uniform(0, 2*np.pi)
        boxes.append((x, y, theta))

    if len(boxes) < n_boxes:
        print("Could only place", len(boxes), "boxes")

    return robot_qpos, boxes


def build_xml(robot_qpos, boxes, stl_model_path,extra_xml,Z_BOX, box_size, ARENA_X1, ARENA_Y1, env_type, wall_clearence_outer, wall_clearence_inner, robot_rgb):
    """Building data for a different file"""

    if env_type == "Partially_closed_with_walls" or env_type == "Partially_closed_with_static": 
      
      extra=0.0

      if env_type == "Partially_closed_with_walls":

        extra=0.2

      side_walls_code= f"""
    <!-- X-walls: left and right sides -->
    <geom name="Wall_X1" type="box"
      pos="{-wall_clearence_inner} {ARENA_Y1/2+extra/2} 0.15"
      size="0.01 {ARENA_Y1/2+extra} 0.15"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>

    <geom name="Wall_X2" type="box"
      pos="{ARENA_X1+wall_clearence_inner} {ARENA_Y1/2+extra/2} 0.15"
      size="0.01 {ARENA_Y1/2+extra} 0.15"
      rgba="1 1 1 0.1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>
"""
    else:
      side_walls_code = ""
    
    header = f"""
<mujoco model="box_delivery_structured_env">
  <compiler angle="radian" autolimits="true" meshdir="{stl_model_path}" inertiafromgeom="true"/>

  <option integrator="implicitfast" gravity="0 0 -9.81" timestep="0.002" iterations="50" viscosity="1.5"/>

  <default>
    <joint limited="false" armature="0.01"/>
    <equality solref="0.0002 1" solimp="0.99 0.99 0.0001"/>
  </default>

  <asset>

    <material name="blue_mat" rgba="0.4 0.3 0.2 1"/>

    <mesh name="corner_full" file="corner_full.stl" scale="0.001 0.001 0.001"/>
    <mesh name="burger_base" file="burger_base.stl"scale="0.001 0.001 0.001"/>
    <mesh name="left_tire"   file="left_tire.stl"  scale="0.001 0.001 0.001"/>
    <mesh name="right_tire"  file="right_tire.stl" scale="0.001 0.001 0.001"/>
    <mesh name="lds"         file="lds.stl"        scale="0.001 0.001 0.001"/>
    <mesh name="bumper"      file="TurtleBot3 Burger Bumper.STL" scale="0.001 0.001 0.001"/>
  </asset>

  <visual>
    <quality shadowsize="4096"/>
    <headlight ambient="1 1 1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
  </visual>

  <worldbody>
  
    <!-- Floor -->
    <body pos="{ARENA_X1/2} {ARENA_Y1/2} 0">
      <geom name="floor" type="box" size="{ARENA_X1/2+wall_clearence_outer+0.5} {ARENA_Y1/2+wall_clearence_outer+0.5} 0.01" friction="0.5 0.05 0.0001" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Marked area -->

    <site name="clearance_outline" 
      pos="{ARENA_X1/2} {ARENA_Y1/2} 0.001" 
      type="box"
      size="{ARENA_X1/2} {ARENA_Y1/2} 0.01" 
      rgba="0 1 0 1"/>
      
    <!-- X-walls: left and right sides -->
    <geom name="Wall_X1" type="box"
      pos="{-wall_clearence_outer-0.25} {ARENA_Y1/2} 0.15"
      size="0.25 {ARENA_Y1/2+wall_clearence_outer+0.25} 0.15"
      rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>

    <geom name="Wall_X2" type="box"
      pos="{ARENA_X1+wall_clearence_outer+0.25} {ARENA_Y1/2} 0.15"
      size="0.25 {ARENA_Y1/2+wall_clearence_outer+0.25} 0.15"
      rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>

    <!-- Y-walls: bottom and top -->
    <geom name="Wall_Y1" type="box"
      pos="{ARENA_X1/2} {-wall_clearence_outer-0.25} 0.15"
      size="{ARENA_X1/2+wall_clearence_outer+0.25} 0.25 0.15"
      rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"
      friction="0.45 0.01 0.003"/>

    <geom name="Wall_Y2" type="box"
      pos="{ARENA_X1/2} {ARENA_Y1+wall_clearence_outer+0.25} 0.15"
      size="{ARENA_X1/2+wall_clearence_outer+0.25} 0.25 0.15"
      rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"
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
      
"""
    
    #Data to be written for boxes
    box_xml = ""
    for i, (x, y, th) in enumerate(boxes):
        qw, qx, qy, qz = quat_z(th)
        box_xml += f"""
    <body name="box{i}" pos="{x:.4f} {y:.4f} {Z_BOX:.3f}">
      <joint name="box{i}_joint" type="free" />
      <geom type="box" size="{box_size} {box_size} {box_size}" material="blue_mat" mass="0.05"
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
    return header + box_xml + extra_xml + side_walls_code+ footer


def compute_inner_square(ARENA_X, ARENA_Y, internal_clearance_length):
  """
  Returns the four (x,y) corner points of the arena inset by `internal_clearance_length` on all sides.
  Points are ordered (bottom-left, bottom-right, top-right, top-left).
  """
  xmin = ARENA_X[0] + internal_clearance_length
  xmax = ARENA_X[1] - internal_clearance_length
  ymin = ARENA_Y[0] + internal_clearance_length
  ymax = ARENA_Y[1] - internal_clearance_length

  return [
      (xmin, ymin),
      (xmax, ymin),
      (xmax, ymax),
      (xmin, ymax)
    ]

def generate_boxDelivery_xml(N,env_type,file_name,ROBOT_clear,BOXES_clear,Z_BOX,ARENA_X,ARENA_Y,
                  box_half_size,num_pillars, pillar_half, wall_clearence_outer, wall_clearence_inner, internal_clearance_length):
    
    # Name of input and output file otherwise set to default
    XML_OUT = Path(file_name)
    stl_model_path = os.path.join(os.path.dirname(__file__), 'models/')
    
    # Changing based on configration type
    clearance_poly=compute_inner_square(ARENA_X, ARENA_Y, internal_clearance_length)
    extra_xml, keep_out = changing_per_configuration(env_type, ARENA_X,ARENA_Y, num_pillars, pillar_half, wall_clearence_outer, internal_clearance_length)
    
    # Finding the robot's q_pos and boxes's randomized data
    robot_qpos, boxes = sample_scene(N,keep_out,ROBOT_clear,BOXES_clear,ARENA_X,ARENA_Y, internal_clearance_length)
  
    # Building new environemnt and writing it down
    xml_string = build_xml(robot_qpos, boxes,stl_model_path,extra_xml,Z_BOX,box_half_size,ARENA_X[1],ARENA_Y[1],env_type, wall_clearence_outer, wall_clearence_inner, robot_rgb=(0.1, 0.1, 0.1))
    XML_OUT.write_text(xml_string)
    
    return XML_OUT, keep_out , clearance_poly
