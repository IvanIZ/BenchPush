import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import glfw
import argparse, math
from textwrap import dedent
from pathlib import Path
import torch,os

from typing import List
import numpy as np
import packcircles as pc
from skimage import draw
import pickle
import json

from benchnpin.common.geometry.polygon import generate_polygon, poly_area
from benchnpin.common.utils.mujoco_utils import polygon_from_vertices, extrude_and_export

ASV_MASS_TOTAL = 6000000.0     # kg
ICE_DENSITY    = 9000.0        # kg m⁻³ (Since ice is shown as an extermely thin plate for stimulation purposes)
RHO_WATER      = 1025.0        # kg m⁻³
CD_SHIP             = 0.5      # assuming like an airfoil
CD_ICE              = 1.1   # assuming like an airfoil
Cd_yaw_ship         = 10.0    
Cd_yaw_ice          = 2.0
DAMP_BETA_SHIP      = 1.0
ANG_DAMP_BETA_SHIP  = 1.0      # torque coefficient
DAMP_BETA_ICE       = 1.50
ANG_DAMP_BETA_ICE   = 1.50     # torque coefficient
STL_SCALE           = 0.4

# Random position of asv
ASV_Y0 = 100
ASV_X0 = random.uniform(50, 150)
ASV_POS = np.array([ASV_X0, ASV_Y0])

# rectangular area which has to be avoided for ice placement close to the ship
ICE_CLEAR_X = 50.0
ICE_CLEAR_Y = 45.0


# parameters for ice field generation
OBS_MIN_Y = 300
OBSTACLE = {
    'min_r': 6,                           # min and max radius of obstacles
    'max_r': 25,
    'min_y': OBS_MIN_Y,                             # boundaries of where obstacles can be placed
    'circular': False,                      # if True, obstacles are more circular, otherwise they are convex polygons
    'exp_dist': False                        # if True, obstacle radii are sampled from an exponential distribution
}

TOL = 0.01              # tolerance of deviation of actual concentration from desired concentration
SCALE = 8               # scales map by this factor for occupancy grid resolution
    
# Part 1- building the required file


def hfield_data_as_string(n=64, amp=0.2, kx=2*np.pi/200, ky=2*np.pi/80):
    """
    Creates a (nxn) sinusoidal heightfield
    """    
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    y = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    height = amp * np.sin(kx*X) * np.cos(ky*Y)
    return " ".join(f"{h:.6f}" for h in height.ravel())




def header_block(hfield, stl_model_path, sim_timestep, channel_len, channel_wid, num_floes):
    """Creates a header for the XML code"""

    # create mesh entries
    mesh_entries = []
    for i in range(num_floes):
        mesh_entries.append(
            f'<mesh name="ice_{i}_mesh" file="ice_floes/ice_{i}.stl" scale="1 1 1"/>'
        )

    #DAMPING FOR YAW ANGLE HAS TO BE CHANGED
    header= dedent(f"""\
        <mujoco model="asv_with_ice_random">
          <compiler angle="degree" meshdir="{stl_model_path}" inertiafromgeom="true"/>
          <option timestep="{sim_timestep}" gravity="0 0 -9.81" viscosity="1.5"/>

          <!-- Global material presets -->
          <asset>
            <mesh name="asv_mesh" file="cs_long.stl" scale="{STL_SCALE} {STL_SCALE} {STL_SCALE}"/>
            <mesh name="cs_long_bottom" file="cs_long_bottom.stl" scale="{STL_SCALE} {STL_SCALE} {STL_SCALE}"/>
            <mesh name="cs_long_top" file="cs_long_top.stl" scale="{STL_SCALE} {STL_SCALE} {STL_SCALE}"/>
            <mesh name="CONTAINER_Full" file="CONTAINER_Full.stl" scale="{STL_SCALE} {STL_SCALE} {STL_SCALE}"/>

            <texture name="ice_tex" type="2d" file="models/ice_texture.png" />
            <material name="ice_mat" texture="ice_tex"/>
            {' '.join(mesh_entries)}
            <texture name="water" type="2d" file="models/Background.png" />
            <material name="water_" texture="water"/>
            <hfield name="wave_field" nrow="640" ncol="640" size="{channel_wid/2} {channel_len/2} 4.0 2.0">
               {hfield}
            </hfield>
          </asset>
          
          <default>
          
             <geom solimp="0.95 0.99 0.01"/>
             
             
             <default class="asv_body">
               <geom density="250" friction="0.00714 0.00714 0.0001"/>
             </default>

             <default class="ice_floe">
                <geom density="{ICE_DENSITY}" friction="0.35 0.35 0.0001"/>
             </default>
          
          </default>
          
          
          <worldbody>

            <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"/>
            <geom name="sea_surface" type="plane" size="100000 100000 0.1" rgba="0 0.47 0.74 1" contype="0" conaffinity="0"/>
            <!-- visual water surface (uses the height-field) -->
            <geom name="hfield" type="hfield" hfield="wave_field" pos="0 0 0" size="1 1 2" rgba="0 0.48 0.9 1" contype="0" conaffinity="0" />
            <geom name="sea_surface2" type="plane" size="100000 100000 0.1" pos="0 0 -1" material="water_" contype="0" conaffinity="0"/>

            <camera name="overview_cam" pos="50 -200 200" euler="60 0 0" fovy="60"/>
            
            <!ASV->
            <body name="asv" pos="{ASV_X0} {ASV_Y0} 0" quat="0.7071 0 0 0.7071">
              <joint name="asv_x"   type="slide" axis="1 0 0"/>
              <joint name="asv_y"   type="slide" axis="0 1 0"/>
              <joint name="asv_yaw" type="hinge" axis="0 0 1" damping="10.0"/>
              <geom name="asv_cs_long_bottom" type="mesh" mesh="cs_long_bottom" mass="{ASV_MASS_TOTAL/2}" rgba="0.698 0.133 0.133 1" euler="0 0 -180"/>
              <geom name="asv_cs_long_top" type="mesh" mesh="cs_long_top" mass="{ASV_MASS_TOTAL/2}" rgba="0.45 0.47 0.50 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full1" type="mesh" mesh="CONTAINER_Full" pos="{-1*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}"  rgba="0.588 0.204 0.188 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full2" type="mesh" mesh="CONTAINER_Full" pos="{0.4*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}" rgba="0.169 0.329 0.553 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full3" type="mesh" mesh="CONTAINER_Full" pos="{1.8*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}" rgba="0.216 0.278 0.310 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full4" type="mesh" mesh="CONTAINER_Full" pos="{3.2*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}" rgba="0.439 0.502 0.565 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full5" type="mesh" mesh="CONTAINER_Full" pos="{4.6*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}" rgba="0.588 0.204 0.188 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full6" type="mesh" mesh="CONTAINER_Full" pos="{6.0*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}"  rgba="0.588 0.204 0.188 1" euler="0 0 -180"/>/>
              <geom name="asv_CONTAINER_Full7" type="mesh" mesh="CONTAINER_Full" pos="{7.4*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}"  rgba="0.588 0.204 0.188 1" euler="0 0 -180"/>/>
              <geom name="asv_CONTAINER_Full8" type="mesh" mesh="CONTAINER_Full" pos="{8.8*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}" rgba="0.216 0.278 0.310 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full9" type="mesh" mesh="CONTAINER_Full" pos="{10.2*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}" rgba="1.000 0.843 0.000 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full10" type="mesh" mesh="CONTAINER_Full" pos="{11.6*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}" rgba="0.588 0.204 0.188 1" euler="0 0 -180"/>
              <geom name="asv_CONTAINER_Full11" type="mesh" mesh="CONTAINER_Full" pos="{13.0*STL_SCALE*10} {-4*STL_SCALE*10} {1*STL_SCALE*10}" rgba="0.588 0.204 0.188 1" euler="0 0 -180"/>



              <camera name="asv_cam" pos="-0 0 25" euler="0 -90 -90" fovy="60"/>
            </body>

            {generate_waypoint_sites(500)}
    """)
    return header


ICE_MESH_BODY_TEMPLATE1 = """
    <body name="ice_{n}" pos="{x:.2f} {y:.2f} 0">
        <joint name="ice_{n}_x"   type="slide" axis="1 0 0"/>
        <joint name="ice_{n}_y"   type="slide" axis="0 1 0"/>
        <joint name="ice_{n}_yaw" type="hinge" axis="0 0 1" damping="750.0"/>
        <geom name="ice_{n}" class="ice_floe" type="mesh" mesh="ice_{n}_mesh" material="ice_mat"/>
    </body>
"""

def generate_ice_mesh_bodies(positions: list[tuple[float, float]]) -> str:
    """
    Given a list of (x, y) positions and num_ice_floes = len(positions),
    generate the <body> XML strings using STL mesh geometries.
    """
    body_strings = []
    for n, (x, y) in enumerate(positions):
        body_strings.append(
            ICE_MESH_BODY_TEMPLATE1.format(n=n, x=x, y=y)
        )
    return "\n".join(body_strings)

"""
def random_ice_bodies(concentration: float,
                      spacing: float = 5.0,
                      max_tries: int = 2000000000, 
                      icefield_len = 1000,
                      icefield_wid = 200) -> str:
    
    Returning code for placing ice floes until desired concentration is covered,
    Variables:
    spacing- minimum space between ice cubes
    asv_radius- radius along which no cube is place from the origion of ship
    

    channel_area = icefield_len * icefield_wid
    target_area  = concentration * channel_area
    placed_area  = 0.0

    bodies, positions, radii = [], [], []
    grid: dict[tuple[int, int], list[int]] = {}  # cell

    # cell size chosen so that two floes in non-neighbouring cells
    # are guaranteed to be farther apart than spacing + their radii
    r_max     = 25.0 * 2**0.5
    cell_size = spacing + r_max

    def cell_of(x: float, y: float) -> tuple[int, int]:
        return (int((x + icefield_len/2) // cell_size),
                int((y + icefield_wid/2) // cell_size))

    # main rejection-sampling loop
    tries = 0
    start = time.time()
    
    while placed_area < target_area and tries < max_tries:

        # random candidate
        sx, sy = random.uniform(6.0, 25.0), random.uniform(6.0, 25.0)
        sz     = 0.6
        radius = math.hypot(sx, sy)

        x = random.uniform(radius, icefield_wid - radius)
        y = random.uniform(radius, icefield_len - radius)

        # neighbour search
        cx, cy = cell_of(x, y)
        neighbours = [
            grid.get((cx + dx, cy + dy), [])
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        ]
        indices_to_check = [idx for bucket in neighbours for idx in bucket]

        too_close = False
        for idx in indices_to_check:
            px, py   = positions[idx]
            pr       = radii[idx]
            if math.hypot(x - px, y - py) < (pr + radius + spacing):
                too_close = True
                break
        
        asv_pos=ASV_POS
        # ASV clearance
        if (asv_pos[0] - ICE_CLEAR_X - radius < x < asv_pos[0] + ICE_CLEAR_X + radius
            and
            asv_pos[1] - ICE_CLEAR_Y - radius < y < asv_pos[1] + ICE_CLEAR_Y + radius):
            too_close = True

        # accept / reject
        if not too_close:
            n = len(positions)
            bodies.append(
                ICE_BODY_TEMPLATE.format(n=n, x=x, y=y, sx=sx, sy=sy, sz=sz)
            )
            positions.append((x, y))
            radii.append(radius)

            # insert into grid
            grid.setdefault((cx, cy), []).append(n)

            placed_area += (2 * sx) * (2 * sy)

        tries += 1
    
    end = time.time()
    total_time=end-start
    num_floes=len(bodies)
    
    print(f"[ice] only reached {placed_area/channel_area:.1%} "
              f"coverage after {tries:,} samples in {total_time:.3f}")

    return "\n".join(bodies), num_floes"""




def footer_block():
    """Setting up footer block"""
    
    footer = dedent("""\

          </worldbody>

          <actuator>
            <!-- <motor name="asv_forward" joint="asv_x"  ctrlrange="-6e7 9e7" gear="1"/> -->
            <!-- <motor name="asv_rudder"  joint="asv_yaw" ctrlrange="-6e7 9e7"   gear="5"/> -->

            <velocity name="asv_forward" joint="asv_x"  ctrlrange="-80 80" forcelimited="false" kv="1000000.0"/>
            <velocity name="asv_rudder"  joint="asv_yaw" ctrlrange="-10 10" forcelimited="false" kv="10000000000.0"/>
          </actuator>
        </mujoco>
    """)
    
    return footer


def generate_waypoint_sites(num_sites=500):
    site_template = (
        '<site name="wp{0}" pos="0 500 5" size="1" rgba="0 1 0 1" type="sphere"/>'
    )
    return "\n".join([site_template.format(i) for i in range(num_sites)])


def generate_shipice_xml(concentration, xml_file, sim_timestep, channel_len, channel_wid, icefield_len, icefield_wid, load_cached=True, trial_idx=0) -> str:

    # get current directory of this script
    current_dir = os.path.dirname(__file__)
    polygon_file = os.path.join(current_dir, 'models/ice_field' + str(concentration) + '.pkl')

    # generate random ice floes
    map_shape = (icefield_len, icefield_wid)
    ship_state = {
        # range for x, y, theta for generating a random ship starting position
        'range_x': [200, 300],  # if set to None, then the ship starts at the x position with lowest ice concentration
        'range_y': [0, 0],
        'range_theta': [np.pi / 2, np.pi / 2]
    }
    goal = (2, channel_len)            # NOTE: change to a tuple so planner doesn't complain

    ice_area_dict=dict()

    if load_cached:
        with open(polygon_file, 'rb') as f:
            data = pickle.load(f)

    else:
        data = generate_rand_exp(conc=concentration, map_shape=map_shape, ship_state=ship_state, goal=goal, max_trials=1, filename=polygon_file)

    trial_data = data['exp'][trial_idx]
    num_stl_floes = len(trial_data['obstacles'])

    # get current directory of this script
    current_dir = os.path.dirname(__file__)
    directory = os.path.join(current_dir, 'models/ice_floes')
    if directory:
        os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist

    positions = []
    for i in range(num_stl_floes):
        vertices = trial_data['obstacles'][i]['vertices']
        center = np.array(trial_data['obstacles'][i]['centre'])
        transformed_vertices = vertices - center
        positions.append(center)
        
        polygon = polygon_from_vertices(transformed_vertices)

        area_2d = polygon.area        # Shapely gives signed area directly
        ice_area_dict[f'ice_{i}'] = {'area': area_2d, 'vertices': vertices}

        out_file = os.path.join(directory, 'ice_' + str(i) + '.stl')
        extrude_and_export(polygon, h_min=0.4, h_max=1.0, filename=out_file)


    # get stl model path
    stl_model_path = os.path.join(os.path.dirname(__file__), 'models/')

    # Building the heightfield used for stimulation
    hfield = hfield_data_as_string()

    # ice_floe_text, num_floes = random_ice_bodies(concentration, icefield_len=icefield_len, icefield_wid=icefield_wid)

    ice_floe_text = generate_ice_mesh_bodies(positions)

    xml_text = header_block(hfield, stl_model_path, sim_timestep, channel_len=channel_len, channel_wid=channel_wid, num_floes=num_stl_floes) + ice_floe_text + "\n" + footer_block()
    Path(xml_file).write_text(xml_text)

    return num_stl_floes, ice_area_dict



def load_ice_field(concentration, xml_file, sim_timestep, channel_len, channel_wid, icefield_len, icefield_wid, load_cached=True, trial_idx=0) -> str:

    # get current directory of this script
    current_dir = os.path.dirname(__file__)
    polygon_file = os.path.join(current_dir, 'models/fields/field3.json')

    with open(polygon_file, 'rb') as f:
        data = json.load(f)

    trial_data = data
    num_stl_floes = len(trial_data['ice'])

    # get current directory of this script
    current_dir = os.path.dirname(__file__)
    directory = os.path.join(current_dir, 'models/ice_floes')
    if directory:
        os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist

    ice_dict = dict()

    positions = []
    print(num_stl_floes)
    ice_idx = 0
    for ice_key, ice_data in trial_data['ice'].items():
        vertices = np.array(ice_data['perimeter'])
        center = np.array(ice_data['location'])
        vertices = np.flip(vertices, axis=1)
        center = np.flip(center, axis=0)

        # offset positions
        center[1] = center[1] + OBS_MIN_Y
        positions.append(center)

        polygon = polygon_from_vertices(vertices)

        area_2d = polygon.area        # Shapely gives signed area directly
        ice_dict[f'ice_{ice_idx}'] = {'area': area_2d, 'vertices': vertices, 'vel': 0.0}

        out_file = os.path.join(directory, 'ice_' + str(ice_idx) + '.stl')
        extrude_and_export(polygon, h_min=0.4, h_max=1.0, filename=out_file)

        ice_idx += 1


    # get stl model path
    stl_model_path = os.path.join(os.path.dirname(__file__), 'models/')

    # Building the heightfield used for stimulation
    hfield = hfield_data_as_string()

    # ice_floe_text, num_floes = random_ice_bodies(concentration, icefield_len=icefield_len, icefield_wid=icefield_wid)

    ice_floe_text = generate_ice_mesh_bodies(positions)

    xml_text = header_block(hfield, stl_model_path, sim_timestep, channel_len=channel_len, channel_wid=channel_wid, num_floes=num_stl_floes) + ice_floe_text + "\n" + footer_block()
    Path(xml_file).write_text(xml_text)

    return num_stl_floes, ice_dict



# Part 2- Performing operations in the file

def apply_fluid_forces_to_body(model, data, body_name, joint_prefix, phase, ice_area_dict,angular_beta_ship=ANG_DAMP_BETA_SHIP, beta_ship=DAMP_BETA_SHIP, angular_beta_ice=ANG_DAMP_BETA_ICE, beta_ice=DAMP_BETA_ICE, wave_amp=0.2, g=9.81, kx=2*np.pi/200, ky=2*np.pi/80, max_omega=0.5):
    """Drag force and wave force in two dimensions"""

    if body_name=="asv":
        area=8
        beta = beta_ship
        angular_beta = angular_beta_ship
        thickness= 2.0
        Cd= CD_SHIP
        r_mean= np.sqrt(area)/2
        Cd_yaw= Cd_yaw_ship
    else:
        area = ice_area_dict.get(body_name)['area']
        beta = beta_ice
        angular_beta = angular_beta_ice
        thickness= 0.6
        Cd= CD_ICE
        Cd_yaw = Cd_yaw_ice
        r_mean= np.sqrt(area/np.pi)
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # The computations below are for drag
    jnt_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_prefix + "_x")
    jnt_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_prefix + "_y")
    jnt_yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_prefix + "_yaw")
    dof_x = model.jnt_dofadr[jnt_x]
    dof_y = model.jnt_dofadr[jnt_y]
    dof_yaw = model.jnt_dofadr[jnt_yaw]

    vx = data.qvel[dof_x]
    vy = data.qvel[dof_y]
    v = np.array([vx, vy])
    v_mag = np.linalg.norm(v)
    v_dir = v / v_mag if v_mag > 0.01 else np.zeros(2)

    F_linear = -beta * v
    F_quad = -0.5 * RHO_WATER * Cd * area * (v_mag**2) * v_dir
    Fxy_drag = F_linear + F_quad

    omega_z = data.qvel[dof_yaw]
    yaw_drag_linear = -angular_beta * omega_z
    yaw_drag_quad   = -0.5 * RHO_WATER * area * omega_z * abs(omega_z) * (r_mean**2) * Cd_yaw

    torque_z = yaw_drag_linear + yaw_drag_quad
    total_torque = np.array([0.0, 0.0,torque_z])
    
    # Clamping max omega just for not having errors
    if omega_z > max_omega:
        data.qvel[dof_yaw] = max_omega
    elif omega_z < -max_omega:
        data.qvel[dof_yaw] = -max_omega    
    
    # computation for wave force below
    pos = data.xpos[body_id]
    x, y = pos[0], pos[1]
    
    # Randomly generating kx and ky but not showing it in simulation due to not increasing time
    # gradients of sin(kx x + ky y + phase)
    dhdx = wave_amp * kx * np.cos(kx*x + ky*y + phase)
    dhdy = wave_amp * ky * np.cos(kx*x + ky*y + phase)
    Vdisp = area * thickness
    
    # Wave force computation
    Fxy_wave = -RHO_WATER * g * Vdisp * np.array([dhdx, dhdy])
    
    # Net computation
    total_Fxy = Fxy_drag +Fxy_wave
    # Only taking in two dimensions for now
    Fz_wave = 0
    
    # total force is acting at the origion of the body
    total_force = np.array([total_Fxy[0], total_Fxy[1], Fz_wave])    
    #point = np.zeros((3, 1))
    point = data.xpos[body_id].copy()

    mujoco.mj_applyFT(model, data, force=total_force, torque=total_torque, body=body_id, point=point, qfrc_target=data.qfrc_applied)


def init_body_dict(model, data):
    """
    Returns {body_name: [total_distance_traveled, current_pos, mass (kg), body_id]}
    """
    body_dict = {}
    
    for body_id in range(model.nbody):
        
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        if name is None:
            continue
        
        pos = data.xpos[body_id].copy()
        mass = model.body_mass[body_id]
        body_dict[name] = [0.0, pos, mass, body_id]
        
    return body_dict

def update_body_dict(model, data, state: dict):
    """
    Updates each body's path length & position. Returns updated dict and a list
    of names of bodies who are in motion
    """
    
    for name, (dist, prev_pos, mass, bid) in state.items():
        
        v_lin = data.cvel[bid][3:]
        speed = float(np.linalg.norm(v_lin))
        
        if speed > 0:
            
            cur_pos = data.xpos[bid].copy()
            state[name][0] += float(np.linalg.norm(cur_pos - prev_pos))
            state[name][1] = cur_pos
            
    return state


def rewards(model,data,asv_bid,asv_geom_ids,geom_to_body,state):
    """ To get the net reward of each step"""
    
    def cos_phi_to_goal(model, data,asv_bid):
        """
        cos(φ) between ASV forward axis (+x in local) and the fixed goal vector.
        """
            
        asv_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "asv")
            
        GOAL_DIR = np.array([0.0, 1.0, 0.0])
        
        # body-to-world rotation matrix (row-major, 9 elems)
        xmat = data.xmat[asv_bid].reshape(3, 3)
        fwd  = xmat[:, 0]
        
        return float(np.dot(fwd, GOAL_DIR) /
                     (np.linalg.norm(fwd)))

    def check_ship_collisions_fast(data,asv_geom_ids,geom_to_body):
        """
        If collision takes place between ship and ice cubes then gives the number of collision
        """
        
        num_of_collision=0
        
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
    
            if (g1 in asv_geom_ids) ^ (g2 in asv_geom_ids):
                other = g2 if g1 in asv_geom_ids else g1
                name = geom_to_body[other]
                if name.startswith("ice_"):
                    num_of_collision+=1
    
        return num_of_collision
    
    
    def goal_reached(state: dict, x_goal=500.0) -> bool:
        """
        Terminated if ASV’s current x ≥ x_goal.
        """
        
        asv_x = state["asv"][1][0]
        success=asv_x >= x_goal
        
        return success
    
    cos_phi=cos_phi_to_goal(model, data,asv_bid)
    num_of_collision=check_ship_collisions_fast(data,asv_geom_ids,geom_to_body)
    success=goal_reached(state)
    
    return cos_phi, num_of_collision, success
        

def compute_p_control(current_yaw, target_yaw):
    """Computes the PD control signal for yaw control
    TO BE USED LATER"""
    
    # PD controller gains
    kp = 1.5
    
    error = target_yaw - current_yaw
    # Making sure error is within -pi to pi
    error = np.arctan2(np.sin(error), np.cos(error))
    
    control_signal = kp * error
    
    return control_signal

def evaluating(body_state,start_xy,goal_x,success):
    """Evaluating different parameters"""

    def path_efficiency(body_state: dict,
                        start_xy: tuple[float, float],
                        goal_x: float,
                        success: bool) -> float:
        """
        0 if the ASV never reached x >= goal_x
        L_shortest / L_actual   otherwise
        """
        if not success:
            return 0.0
    
        x0, y0 = start_xy
        L_shortest = math.hypot(goal_x - x0, 0.0)
        L_actual   = body_state["asv"][0]
    
        return 0.0 if L_actual == 0 else L_shortest / L_actual
    
    def interaction_effort_score(body_state: dict) -> float:
        """
        Gives interaction effort score. Returns 0 if the denominator is zero
        """
        num   = body_state["asv"][2] * body_state["asv"][0]
        denom = sum( mass * dist for dist, _, mass, _ in body_state.values() )
    
        return 0.0 if denom == 0 else num / denom
    
    path_efficiency = path_efficiency(body_state, start_xy, goal_x, success)
    interaction_effort_score = interaction_effort_score(body_state)
    
    return path_efficiency, interaction_effort_score


def compute_poly_ob_concentration(polys, map_shape):
    im = np.zeros((map_shape[0] * SCALE, map_shape[1] * SCALE))
    area = 0
    for p in polys:
        area += p['area']
        rr, cc = p['pixels']
        im[rr, cc] = 1

    return area / (map_shape[1] * (map_shape[0] - OBSTACLE['min_y'])), im


def increase_concentration(obstacles, desired_concentration, map_shape):
    actual_concentration, im = compute_poly_ob_concentration(obstacles, map_shape)

    max_r = OBSTACLE['max_r']
    num_added = 0
    trials = 0

    while actual_concentration < desired_concentration - TOL and max_r - OBSTACLE['min_r'] > 0.05:
        if trials % 10 == 0:
            print('num trials', trials, 'current concentration', actual_concentration, 'max_r', max_r, 'added obs', num_added)
        trials += 1
        r = np.random.uniform(OBSTACLE['min_r'], max_r)
        slice_shape = int(max(1 / SCALE, r * 2) * SCALE)  # in map units

        # find all the slices that would a new obstacle using a sliding window approach
        new_obs_centres = []
        rand_offset_x = np.random.choice(np.arange(slice_shape))
        rand_offset_y = np.random.choice(np.arange(slice_shape))
        for i in range(rand_offset_y, im.shape[0] - slice_shape + 1, slice_shape):
            for j in range(rand_offset_x, im.shape[1] - slice_shape + 1, slice_shape):
                # skip if slice is beyond ice edge
                if OBSTACLE['min_y'] * SCALE <= i and i + slice_shape <= map_shape[0] * SCALE:
                    if im[i: i + slice_shape, j: j + slice_shape].sum() == 0:
                        new_obs_centres.append([(j + slice_shape / 2) / SCALE, (i + slice_shape / 2) / SCALE])

        if len(new_obs_centres) == 0:
            # decrease upper bound on r if no slices were found
            max_r = r
        else:
            # randomly choose one of these slices and generate an obstacle
            indices = np.random.choice(len(new_obs_centres), size=int(len(new_obs_centres) * 0.5), replace=False)
            for ind in indices:
                slice_choice = new_obs_centres[ind]
                x, y = slice_choice
                r = slice_shape / SCALE / 2
                vertices = generate_polygon(diameter=r * 2, origin=(x, y), circular=OBSTACLE['circular'])

                if vertices is not None:
                    # add ob to obstacles list
                    im_shape = (map_shape[0] * SCALE, map_shape[1] * SCALE)
                    rr, cc = draw.polygon(vertices[:, 1] * SCALE, vertices[:, 0] * SCALE, shape=im_shape)
                    if len(rr) == 0 and im[rr, cc].sum() > 0:
                        continue
                    obstacles.append({
                        'vertices': vertices,
                        'centre': (x, y),
                        'radius': r,
                        'pixels': (rr, cc),
                        'area': poly_area(vertices)
                    })

                    num_added += 1

                    # compute new poly concentration
                    actual_concentration, im = compute_poly_ob_concentration(obstacles, map_shape)

    print('added {} obstacles over {} trails!\ndesired concentration {}, actual concentration {}'
          .format(num_added, trials, desired_concentration, actual_concentration))

    return obstacles


def decrease_concentration(obstacles: List[dict], desired_concentration: float, map_shape):
    actual_concentration, _ = compute_poly_ob_concentration(obstacles, map_shape)
    num_deleted = 0
    while actual_concentration > desired_concentration + TOL:
        ind = np.random.choice(np.arange(len(obstacles)))
        obstacles = np.delete(obstacles, ind, axis=0)
        np.random.shuffle(obstacles)
        actual_concentration, _ = compute_poly_ob_concentration(obstacles, map_shape)
        num_deleted += 1

    print('deleted {} obstacles!\ndesired concentration {}, actual concentration {}'
          .format(num_deleted, desired_concentration, actual_concentration))
    return obstacles


def find_best_start_x(obstacles, map_shape, slice_shape=(10, 3)):
    # generate corresponding occupancy grid given obstacles
    im = np.zeros((map_shape[0] * SCALE, map_shape[1] * SCALE))
    for ob in obstacles:
        rr, cc = draw.polygon(ob['vertices'][:, 1] * SCALE, ob['vertices'][:, 0] * SCALE, shape=im.shape)
        im[rr, cc] = 1

    # find the slice that has the lowest concentration in obstacles using a sliding window approach
    # slice_shape[1] needs to be an odd number and be less than MAP_SHAPE[1]
    # we skip the first and last slice to avoid the ship starting too close to channel boundaries
    c = []
    for i in range((im.shape[1] - slice_shape[1] * SCALE) // SCALE):
        if i == 0:
            c.append(np.inf)
        sub_im = im[: slice_shape[0] * SCALE, i * SCALE: (i + slice_shape[1]) * SCALE]
        c.append(sub_im.sum() / np.multiply(*sub_im.shape))

    # get the indices for all the minimums
    min_idx = np.where(np.asarray(c) == np.min(c))[0]

    # return index closet to the middle if there are multiple mins
    if len(min_idx) != 0:
        best_idx = min_idx[np.argmin(np.abs((min_idx + (min_idx + slice_shape[1])) // 2 - map_shape[1] / 2))].item()

    else:
        best_idx = min_idx[0]

    return (best_idx + (best_idx + slice_shape[1])) // 2


def generate_rand_exp(conc, map_shape, ship_state, goal, max_trials, filename=None):
    # dict to store the experiments
    exp_dict = {
        'meta_data': {
            'concentration': conc,
            'map_shape': map_shape,
            'obstacle_config': OBSTACLE,
            'ship_state_config': ship_state,
            'goal': goal,
            'scale': SCALE,
        },
        'exp': {i: {'goal': None, 'ship_state': None, 'obstacles': None} for i in range(max_trials)} 
    }

    # approximate how many circles we need to pack environment assuming average radius
    if OBSTACLE['exp_dist']:
        avg_r = OBSTACLE['min_r'] * 1.5
    else:
        avg_r = (OBSTACLE['min_r'] + OBSTACLE['max_r']) / 2
    num_circ = (np.pi * (((map_shape[0] ** 2 + map_shape[1] ** 2) ** 0.5) / 2) ** 2) / (np.pi * avg_r ** 2)
    # approach is to first pack the environment with circles then convert circles
    # to polygons and then do rejection sampling to attain the desired concentration

    for i in range(max_trials):
        print("current trial: ", i, " / ", max_trials)

        # sample random radii
        if OBSTACLE['exp_dist']:
            radii = np.maximum(OBSTACLE['min_r'], np.minimum(
                OBSTACLE['max_r'], np.random.exponential(scale=avg_r, size=int(num_circ))
            ))
        else:
            radii = np.random.uniform(OBSTACLE['min_r'], OBSTACLE['max_r'], size=int(num_circ))
        gen = pc.pack(radii)  # this is deterministic! Randomness comes from radii
        circles = np.asarray([(x, y, r) for (x, y, r) in gen])
        circles[:, 1] += (-circles[:, 1].min())
        circles[:, 0] += map_shape[1]

        # remove the circles outside of environment boundaries
        circles = circles[np.logical_and(circles[:, 0] >= 0,
                                            circles[:, 0] <= map_shape[1])]
        circles = circles[np.logical_and(circles[:, 1] >= 0,
                                            circles[:, 1] <= map_shape[0])]
        # apply constraints specified in obstacle parameters
        circles = circles[np.logical_and(circles[:, 1] >= OBSTACLE.get('min_y', 0),
                                            circles[:, 1] <= map_shape[0])]

        np.random.shuffle(circles)

        # now generate polygons for each circle
        obstacles = []

        for (x, y, radius) in circles:
            vertices = generate_polygon(diameter=radius * 2, origin=(x, y), circular=OBSTACLE['circular'])

            if vertices is not None:
                # take intersection of vertices and environment boundaries
                vertices[:, 0][vertices[:, 0] < 0] = 0
                vertices[:, 0][vertices[:, 0] >= map_shape[1]] = map_shape[1]

                min_y = OBSTACLE.get('min_y', False) or 0
                max_y = map_shape[0]
                vertices[:, 1][vertices[:, 1] < min_y] = min_y
                vertices[:, 1][vertices[:, 1] > max_y] = max_y

                im_shape = (map_shape[0] * SCALE, map_shape[1] * SCALE)
                rr, cc = draw.polygon(vertices[:, 1] * SCALE, vertices[:, 0] * SCALE, shape=im_shape)
                obstacles.append({
                    'vertices': vertices,
                    'centre': (x, y),
                    'radius': radius,
                    'pixels': (rr, cc),
                    'area': poly_area(vertices)
                })

        # get concentration of ice field with polygon obstacles
        poly_concentration, _ = compute_poly_ob_concentration(obstacles, map_shape)
        if abs(conc - poly_concentration) > TOL:
            print('\ndesired concentration {}, actual concentration {}'.format(conc, poly_concentration))
            if conc > poly_concentration:
                # randomly add obstacles:
                obstacles = increase_concentration(obstacles, conc, map_shape)
            else:
                obstacles = decrease_concentration(obstacles, conc, map_shape)

        # add obstacles to dict
        exp_dict['exp'][i]['obstacles'] = obstacles

        # add goal to dict
        exp_dict['exp'][i]['goal'] = goal

        # generate ship starting state
        print(ship_state)
        if ship_state['range_x'] is None:
            x = find_best_start_x(obstacles)
        else:
            x = np.random.uniform(low=ship_state['range_x'][0], high=ship_state['range_x'][1])
        y = np.random.uniform(low=ship_state['range_y'][0], high=ship_state['range_y'][1])
        theta = np.random.uniform(low=ship_state['range_theta'][0], high=ship_state['range_theta'][1])

        # add to ship state to dict
        exp_dict['exp'][i]['ship_state'] = (x, y, theta)

    # save to disk
    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(exp_dict, f)

    return exp_dict
