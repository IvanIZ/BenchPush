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


def changing_per_configuration(maze_version, maze_width, maze_len):
    """ 
    Based on the configration, it would create code for pillars along with
    polygon to the area where nothing has to be placed.
    """

    def wall(name, cx, cy, half):
        """Returns one pillar at centre cx,cy with half-size"""
        
        xh, yh, zh = half
        heavy_mass = 1e4
        Text = f"""
    <!-- pillar {name} -->
    <body name="{name}" pos="{cx:.3f} {cy:.3f} {zh:.3f}">
      <joint name="{name}_joint" type="free"/>
      <geom type="box" size="{xh:.3f} {yh:.3f} {zh:.3f}" mass="{heavy_mass:.1f}" 
      rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1"/>
    </body>
"""
        Coordinates=[(cx-xh, cy-yh),(cx+xh, cy-yh),(cx+xh, cy+yh),(cx-xh, cy+yh)]
        return Text, Coordinates
  

    extra_xml = ""
    
    # list of polygons to exclude when sampling
    keep_out  = []

    # U-Shape maze
    if maze_version == 1:

      wall_x = maze_width / 2
      wall_y = maze_len * (1/3)
      wall_width = 0.025
      wall_len = maze_len * (1/3)
      xml, poly = wall("u_shape_wall", wall_x, wall_y, (wall_width, wall_len, 0.1))
      extra_xml += xml
      keep_out.append(poly)

    # Z-Shape maze
    elif maze_version == 2:

      wall1_x = maze_width / 3
      wall1_y = maze_len * (1/3)
      wall1_width = 0.025
      wall1_len = maze_len * (1/3)
      xml, poly = wall("u_shape_wall1", wall1_x, wall1_y, (wall1_width, wall1_len, 0.1))
      extra_xml += xml
      keep_out.append(poly)

      wall2_x = maze_width * (2/3)
      wall2_y = maze_len * (2/3)
      wall2_width = 0.025
      wall2_len = maze_len * (1/3)
      xml, poly = wall("u_shape_wall2", wall2_x, wall2_y, (wall2_width, wall2_len, 0.1))
      extra_xml += xml
      keep_out.append(poly)


    else:
        raise ValueError("Invalid maze version")
      
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


def build_xml(robot_qpos, cubes, stl_model_path,extra_xml,Z_CUBE, cube_size, maze_width, maze_len, goal_half, goal_center, robot_rgb=(0.3, 0.3, 0.3)):
    """Building data for a different file"""

    header = f"""
<mujoco model="box_delivery_structured_env">
  <compiler angle="radian" autolimits="true" meshdir="{stl_model_path}"/>

  <option integrator="implicitfast" gravity="0 0 -9.81"
          timestep="0.005" iterations="50" viscosity="1.5"/>

  <default>
    <joint limited="false" armature="0.01"/>
    <equality solref="0.0002 1" solimp="0.99 0.99 0.0001"/>
  </default>

  <asset>

    <material name="blue_mat" reflectance="0.0" rgba="0.4 0.3 0.2 1"/>

    <texture name="floor_tex" type="2d" file="models/gray-floor.png" />
    <material name="floor_mat" reflectance="0.0" texrepeat="1 1" texture="floor_tex"/>

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
    <body pos="{maze_width/2} {maze_len/2}  0">
      <geom type="box" size="{maze_width/2} {maze_len/2}  0.01" friction="0.5 0.05 0.0001"/>
    </body>

    <camera name="bottom_cam" pos="{maze_width/2} -0.4 1.7" quat="9.39692621e-01 3.42020143e-01 2.09426937e-17 5.75395780e-17" fovy="60"/>    
    <camera name="top_cam" pos="{maze_width/2} {maze_len + 0.8} 1.3" quat="0.        0.        0.5       0.8660254" fovy="60"/>
    
    <!-- Goal Area -->
    <geom type="box"
      pos="{goal_center[0]} {goal_center[1]} 0.01"
      size="{goal_half} {goal_half} 0.0005"
      rgba="0.5 1 0.5 1"
      contype="0"
      conaffinity="0"
      friction="0.5 0.05 0.0001"/>
      
    <!-- X-walls: left and right sides -->
    <geom type="box"
      pos="0 {maze_len/2} 0.15"
      size="0.01 {maze_len/2} 0.15"
      rgba="1 1 1 0.1"/>

    <geom type="box"
      pos="{maze_width} {maze_len/2} 0.15"
      size="0.01 {maze_len/2} 0.15"
      rgba="1 1 1 0.1"/>

    <!-- Y-walls: bottom and top -->
    <geom type="box"
      pos="{maze_width/2} 0 0.15"
      size="{maze_width/2} 0.01 0.15"
      rgba="1 1 1 0.1"/>

    <geom type="box"
      pos="{maze_width/2} {maze_len} 0.15"
      size="{maze_width/2} 0.01 0.15"
      rgba="1 1 1 0.1"/>
    
    <!-- robot -->
    <body name="base" pos="{robot_qpos}" euler="0 0 3.141592653589793">
      <joint type="free" name="base_joint"/>
      <!-- chassis -->
      <geom pos="-0.032 0 0.01" type="mesh" rgba="{robot_rgb[0]} {robot_rgb[1]} {robot_rgb[2]} 1" mesh="burger_base" friction="0.1 0.02 0.0001" mass="0.8"/>
      <!-- small box sensor -->
      <geom size="0.015 0.0045 0.01" pos="-0.081 7.96327e-07 0.005" quat="0.707388 -0.706825 0 0" type="box" rgba="{robot_rgb[0]} {robot_rgb[1]} {robot_rgb[2]} 1" mass="0.05"/>
      <!-- LDS sensor -->
      <geom pos="-0.032 0 0.182" quat="1 0 0 0" type="mesh" rgba="{robot_rgb[0]} {robot_rgb[1]} {robot_rgb[2]} 1" mesh="lds" mass="0.131"/>
      <!-- Bumper -->
      <geom pos="-0.04 -0.09 0.01" quat="1 0 0 0" type="mesh" rgba="0.3 0.13 0.08 1" mesh="bumper" mass="0.100"/>
      
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


def generate_maze_xml(N, maze_version, file_name,ROBOT_R,CLEAR,Z_CUBE,ARENA_X,ARENA_Y,
                  cube_half_size, clearance_poly, goal_half, goal_center):
    
    #Name of input and output file otherwise set to default
    XML_OUT = Path(file_name)
    stl_model_path = os.path.join(os.path.dirname(__file__), 'models/')
    
    #Clearnaces and cube sizes
    cube_size = f"{cube_half_size} {cube_half_size} {cube_half_size}"
    
    #Changing based on configration type
    extra_xml, keep_out = changing_per_configuration(maze_version, ARENA_X[1], ARENA_Y[1])
    
    # Finding the robot's q_pos and cubes's randomized data
    robot_qpos, cubes = sample_scene(N, keep_out, ROBOT_R,CLEAR,ARENA_X,ARENA_Y, clearance_poly)
  
    # Building new environemnt and writing it down
    xml_string = build_xml(robot_qpos, cubes,stl_model_path,extra_xml,Z_CUBE, cube_size, ARENA_X[1], ARENA_Y[1], goal_half, goal_center, robot_rgb=(0.1, 0.1, 0.1))
    XML_OUT.write_text(xml_string)

    maze_width = ARENA_X[1]
    maze_len = ARENA_Y[1]
    thickness = 0.01
    wall_vertices = [
      [(0, 0), (thickness, 0), (thickness, maze_len), (0, maze_len)],                                               # right wall
      [(maze_width - thickness, 0), (maze_width, 0), (maze_width, maze_len), (maze_width - thickness, maze_len)],   # left wall
      [(0, maze_len - thickness), (maze_width, maze_len - thickness), (maze_width, maze_len), (0, maze_len)],       # top wall 
      [(0, 0), (maze_width, 0), (maze_width, thickness), (0, thickness)]                                            # bottom wall
    ]

    walls = keep_out + wall_vertices
    
    return XML_OUT, np.array(walls)


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
