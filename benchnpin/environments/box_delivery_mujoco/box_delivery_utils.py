"""
Creates a fresh MJCF file with:
- TurtleBot3 Burger (random start pose)
- N blue cubes (N from user, 10-20) with random yaw
- Randomly create enviornment type
"""

import numpy as np
from pathlib import Path
import os
import random

from benchnpin.common.utils.mujoco_utils import inside_poly, quat_z, quat_z_yaw, corners_xy


def changing_per_configuration(env_type: str):
    """ 
    Based on the configration, it would create code for pillars along with
    polygon to the area where nothing has to be placed.
    """

    def pillar(name, cx, cy, half):
        """Returns one pillar at centre cx,cy with half-size"""
        
        xh, yh, zh = half
        Text=f"""
    <!-- pillar {name} -->
    <geom type="box" size="{xh:.3f} {yh:.3f} {zh:.3f}" pos="{cx:.3f} {cy:.3f} {zh:.3f}""" 
        Coordinates=[(cx-xh, cy-yh),(cx+xh, cy-yh),(cx+xh, cy+yh),(cx-xh, cy+yh)]
        return Text, Coordinates
    
    #Area with clearance on walls and columns applied
    SMALL_POLY = np.array([
            (0.300, 0.600), (0.615, 0.600), (0.615, 0.300),
            (0.975, 0.300), (0.975, 0.600), (1.375, 0.600),
            (1.375, 2.245), (0.975, 2.245), (0.975, 2.600),
            (0.615, 2.600), (0.615, 2.245), (0.300, 2.245)
        ])    

    extra_xml = ""
    
    # list of polygons to exclude when sampling
    keep_out  = []

    # small_columns (4 columns 30 × 30 × 40 cm)
    if env_type == "small_columns":

        half = (0.15, 0.15, 0.20)
        
        # Random number of pillars between 1-4
        n_pillars = random.randint(1, 3)
        # Where the centers of the pillars would exists
        centres = []

        # rejection sample pillar centres until we have 4 non-overlapping
        # positions whose entire footprint falls inside SMALL_POLY
        rng = np.random.default_rng()
        attempts = 0
        while len(centres) < n_pillars and attempts < 50000:
            attempts += 1
            
            # This range is picked on clearance for robot and walls. Chosen the
            # lowest value and later verfiying how well it would work out
            cx = rng.uniform(0.600,1.075)
            cy = rng.uniform(0.600,1.975)

            # test the 4 pillar corners
            corners = [(cx-half[0], cy-half[1]),
                       (cx+half[0], cy-half[1]),
                       (cx+half[0], cy+half[1]),
                       (cx-half[0], cy+half[1])]
            if all(inside_poly(x, y, SMALL_POLY) for x, y in corners):
                
                clearance = 0.30
                diameter   = 2 * half[0]
                
                # Check for clearance to other pillars
                if all(np.hypot(cx - px, cy - py) >= diameter + clearance
                       for px, py in centres):
                    centres.append((cx, cy))                    

        for k, (cx, cy) in enumerate(centres):
            xml, poly = pillar(f"small_col{k}", cx, cy, half)
            extra_xml += xml
            keep_out.append(poly)        

    # large_columns (4 columns 16 × 16 × 10 cm)
    # NEED TO CONFIRM
    elif env_type == "large_columns":
        half = (0.16, 0.16, 0.10)  # m
        centres = [(0.6, 0.6), (0.6, 1.6), (1.0, 0.6), (1.0, 1.6)]
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
  

def sample_scene(n_cubes, keep_out,ROBOT_R,CLEAR,ARENA_X,ARENA_Y, clearance_poly):
    """returns robot pose + list of cube poses (x,y,theta)"""
    
    # Robot iteration for its placement
    for _ in range(200):
        rx = np.random.uniform(*ARENA_X)
        ry = np.random.uniform(*ARENA_Y)
        if inside_poly(rx, ry, clearance_poly) and not intersects_keepout(rx, ry, keep_out):
            break
    robot_qpos = f"{rx:.4f} {ry:.4f} 0.01"
    

    # Cubes iteration for its placement
    cubes = []
    tries = 0
    while len(cubes) < n_cubes and tries < 1000:
        tries += 1
        x = np.random.uniform(*ARENA_X)
        y = np.random.uniform(*ARENA_Y)
        if not inside_poly(x, y, clearance_poly):
            continue
        if intersects_keepout(x, y, keep_out):
            continue        
        if np.hypot(x - rx, y - ry) < ROBOT_R:
            continue
        if any(np.hypot(x - cx, y - cy) < CLEAR for cx, cy, _ in cubes):
            continue
        theta = np.random.uniform(0, 2*np.pi)
        cubes.append((x, y, theta))
    if len(cubes) < n_cubes:
        print("Could only place", len(cubes), "cubes")

    return robot_qpos, cubes


def build_xml(robot_qpos, cubes, stl_model_path,extra_xml,Z_CUBE, cube_size):
    """Building data for a different file"""
    
    header = f"""
<mujoco model="box_delivery_structured_env">
  <compiler angle="radian" autolimits="true" meshdir="{stl_model_path}"/>

  <option integrator="implicitfast" gravity="0 0 -9.81"
          timestep="0.002" iterations="50" viscosity="1.5"/>

  <default>
    <joint limited="false" armature="0.01"/>
    <equality solref="0.0002 1" solimp="0.99 0.99 0.0001"/>
  </default>

  <asset>

    <material name="blue_mat" rgba="0.0 0.3 1.0 1.0"/>

    <mesh name="corner_full" file="corner_full.stl" scale="0.001 0.001 0.001"/>
    <mesh name="burger_base" file="burger_base.stl"scale="0.001 0.001 0.001"/>
    <mesh name="left_tire"   file="left_tire.stl"  scale="0.001 0.001 0.001"/>
    <mesh name="right_tire"  file="right_tire.stl" scale="0.001 0.001 0.001"/>
    <mesh name="lds"         file="lds.stl"        scale="0.001 0.001 0.001"/>
  </asset>

  <visual>
    <quality shadowsize="4096"/>
    <headlight ambient="1 1 1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
  </visual>

  <worldbody>
  
    <!-- Floor -->
    <body pos="0.7875 1.4225 0">
      <geom type="box" size="0.7875 1.4225 0.01" friction="0.5 0.05 0.0001"/>
    </body>
    
    <!-- Corner 1 -->
    <body name="corner_1" pos="0.0 0.0 0.05" quat="0 1 0 0">
      <geom type="mesh" mesh="corner_full" rgba="0.0 0.3 1.0 1.0"/>
    </body>
    
    <!-- Corner 2 -->
    <body name="corner_2" pos="1.575 0.0 0.05" quat="0 1 1 0">
      <geom type="mesh" mesh="corner_full" rgba="0.0 0.3 1.0 1.0"/>
    </body>
    
    <!-- Corner 3 -->
    <body name="corner_3" pos="1.575 2.845 0.05" quat="0 0 1 0">
      <geom type="mesh" mesh="corner_full" rgba="0.0 0.3 1.0 1.0"/>
    </body>
    
    <!-- Marked area -->
    <geom type="box"
      pos="0.15 2.695 0.01"
      size="0.15 0.15 0.0005"
      rgba="0.5 1 0.5 1"
      contype="0"
      conaffinity="0"
      friction="0.5 0.05 0.0001"/>
    
    <!-- transporting area -->
    <geom type="box"
      pos="0.05 2.915 0.01"
      size="0.05 0.05 0.05"
      rgba="0.5 1 0.5 1"/>
    
    <!-- transporting area Y-walls-->
    <geom type="box"
      pos="0.05 2.855 1.0"
      size="0.05 0.01 1.0"
      rgba="1 1 1 0.1"/>
      
    <geom type="box"
      pos="0.05 2.965 1.0"
      size="0.05 0.01 1.0"
      rgba="1 1 1 0.1"/>
      
    <!-- transporting area X-walls-->
    <geom type="box"
      pos="-0.01 2.915 1.0"
      size="0.01 0.05 1.0"
      rgba="1 1 1 0.1"/>
      
    <geom type="box"
      pos="0.11 2.915 1.0"
      size="0.01 0.05 1.0"
      rgba="1 1 1 0.1"/>
      
    <!-- X-walls: left and right sides -->
    <geom type="box"
      pos="-0.01 1.4225 0.15"
      size="0.01 1.4225 0.15"
      rgba="1 1 1 0.1"/>

    <geom type="box"
      pos="1.585 1.4225 0.15"
      size="0.01 1.4225 0.15"
      rgba="1 1 1 0.1"/>

    <!-- Y-walls: bottom and top -->
    <geom type="box"
      pos="0.7875 -0.01 0.15"
      size="0.7875 0.01 0.15"
      rgba="1 1 1 0.1"/>

    <geom type="box"
      pos="0.7875 2.855 0.15"
      size="0.7875 0.01 0.15"
      rgba="1 1 1 0.1"/>
    
    <!-- robot -->
    <body name="base" pos="{robot_qpos}" euler="0 0 3.141592653589793">
      <joint type="free" name="base_joint"/>
      <!-- chassis -->
      <geom pos="-0.032 0 0.01" type="mesh" rgba="0.3 0.3 0.3 1" mesh="burger_base" friction="0.1 0.02 0.0001" mass="0.8"/>
      <!-- small box sensor -->
      <geom size="0.015 0.0045 0.01" pos="-0.081 7.96327e-07 0.005" quat="0.707388 -0.706825 0 0" type="box" rgba="0.3 0.3 0.3 1" mass="0.05"/>
      <!-- LDS sensor -->
      <geom pos="-0.032 0 0.182" quat="1 0 0 0" type="mesh" rgba="0.3 0.3 0.3 1" mesh="lds" mass="0.131"/>
      
      <!-- Left wheel -->
      <body name="wheel_left_link" pos="0 0.08 0.033" quat="0.707388 -0.706825 0 0">
        <inertial pos="0 0 0" quat="-0.000890159 0.706886 0.000889646 0.707326" mass="0.0284989" diaginertia="2.07126e-05 1.11924e-05 1.11756e-05"/>
        <joint name="wheel_left_joint" pos="0 0 0" axis="0 0 1"/>
        <geom quat="0.707388 0.706825 0 0" type="mesh" rgba="0.3 0.3 0.3 1" mesh="left_tire" friction="1.2 0.01 0.001"/>
      </body>
      
      <!-- Right wheel -->
      <body name="wheel_right_link" pos="0 -0.08 0.033" quat="0.707388 -0.706825 0 0">
        <inertial pos="0 0 0" quat="-0.000890159 0.706886 0.000889646 0.707326" mass="0.0284989" diaginertia="2.07126e-05 1.11924e-05 1.11756e-05"/>
        <joint name="wheel_right_joint" pos="0 0 0" axis="0 0 1"/>
        <geom quat="0.707388 0.706825 0 0" type="mesh" rgba="0.3 0.3 0.3 1" mesh="right_tire" friction="1.2 0.01 0.001"/>
      </body>
    </body>
      
"""
    
    #Data to be written for cubes
    cube_xml = ""
    for i, (x, y, th) in enumerate(cubes):
        qw, qx, qy, qz = quat_z(th)
        cube_xml += f"""
    <body name="cube{i}" pos="{x:.4f} {y:.4f} {Z_CUBE:.3f}">
      <joint name="cube{i}_joint" type="free" />
      <geom type="box" size="{cube_size}" material="blue_mat" mass="0.05"
            quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}" friction="0.01 0.05 0.0001"/>
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
    return header + cube_xml + extra_xml + footer


def generate_boxDelivery_xml(N,env_type,file_name,ROBOT_R,CLEAR,Z_CUBE,ARENA_X,ARENA_Y,
                  cube_half_size, clearance_poly):
    
    #Name of input and output file otherwise set to default
    XML_OUT = Path(file_name)
    stl_model_path = os.path.join(os.path.dirname(__file__), 'models/')
    
    #Clearnaces and cube sizes
    cube_size = f"{cube_half_size} {cube_half_size} {cube_half_size}"
    
    #Changing based on configration type
    extra_xml, keep_out = changing_per_configuration(env_type)
    
    # Finding the robot's q_pos and cubes's randomized data
    robot_qpos, cubes = sample_scene(N,keep_out,ROBOT_R,CLEAR,ARENA_X,ARENA_Y, clearance_poly)
  
    # Building new environemnt and writing it down
    xml_string = build_xml(robot_qpos, cubes,stl_model_path,extra_xml,Z_CUBE, cube_size)
    XML_OUT.write_text(xml_string)
    
    return XML_OUT, keep_out


def transporting(model, data, joint_id_boxes):
    """Teleport a cube only if all its vertices are inside the goal box."""
    
    # half-edge of your cube
    HSIZE = 0.03
    # local corner coordinates
    corners_local_coordinates = np.array([
        [-HSIZE, -HSIZE],
        [ HSIZE, -HSIZE],
        [ HSIZE,  HSIZE],
        [-HSIZE,  HSIZE]
    ])
    
    GOAL_CENTRE = np.array([0.15, 2.695])
    GOAL_HALF   = np.array([0.15, 0.15])
    DROP_POS    = np.array([0.05, 2.915, 1.0])
    
    xmin, xmax = GOAL_CENTRE[0] - GOAL_HALF[0], GOAL_CENTRE[0] + GOAL_HALF[0]
    ymin, ymax = GOAL_CENTRE[1] - GOAL_HALF[1], GOAL_CENTRE[1] + GOAL_HALF[1]

    for jid in joint_id_boxes[:]:
        qadr = model.jnt_qposadr[jid]

        # joint centre (x,y) and quaternion
        centre_xy = data.qpos[qadr : qadr+2]
        qw, qx, qy, qz = data.qpos[qadr+3 : qadr+7]

        # yaw from quaternion
        yaw = quat_z_yaw(qw, qx, qy, qz)

        # world vertices
        verts = corners_xy(centre_xy, yaw,corners_local_coordinates)

        # containment test – every vertex must satisfy the four inequalities
        inside = np.all((xmin <= verts[:,0]) & (verts[:,0] <= xmax) &
                        (ymin <= verts[:,1]) & (verts[:,1] <= ymax))

        if inside:
            data.qpos[qadr:qadr+3] = DROP_POS
            joint_id_boxes.remove(jid)
            name = model.names[jid]
