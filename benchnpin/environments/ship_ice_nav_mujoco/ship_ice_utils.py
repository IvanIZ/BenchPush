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

# constant values defined below based on data referenced in research papers

concentration  = float(input("Enter a concentration value between (0.1-0.5):"))
TIMESTEP_SIM   = 0.005         # s
CHANNEL_LEN    = 1000.0        # m (X axis)
CHANNEL_WID    = 200.0         # m (Y axis)
ASV_MASS_TOTAL = 6000000.0     # kg
ICE_DENSITY    = 900.0         # kg m⁻³
RHO_WATER      = 1025.0        # kg m⁻³
CD             = 1.0
ANG_DAMP_BETA  = 0.3          # torque coefficient
STL_SCALE      = 0.3

# Random position of asv
ASV_X0 = -475.0
ASV_Y0 = random.uniform(-75.0, 75.0)
ASV_POS = np.array([ASV_X0, ASV_Y0])

# rectangular area which has to be avoided for ice placement close to the ship
ICE_CLEAR_X = 50.0
ICE_CLEAR_Y = 45.0

def path_of_file(stl_model_path):
    """Path of .stl models"""
    
    if stl_model_path==None:
        stl_model_path = os.path.join(os.path.dirname(__file__), 'models/')
    else:
        stl_model_path=os.path.join(stl_model_path,'models/')
        
    return stl_model_path
    
# Part 1- building the required file


def hfield_data_as_string(n=64, amp=0.2, kx=2*np.pi/200, ky=2*np.pi/80):
    """
    Creates a (n×n) sinusoidal heightfield
    """    
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    y = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    height = amp * np.sin(kx*X) * np.cos(ky*Y)
    return " ".join(f"{h:.6f}" for h in height.ravel())




def header_block(hfield,stl_model_path):
    """Creates a header for the XML code"""
    #DAMPING FOR YAW ANGLE HAS TO BE CHANGED
    header= dedent(f"""\
        <mujoco model="asv_with_ice_random">
          <compiler angle="degree" meshdir="{stl_model_path}" viscous="1.5"/>
          <option timestep="{TIMESTEP_SIM}" gravity="0 0 -9.81"/>

          <!-- Global material presets -->
          <asset>
            <mesh name="asv_mesh" file="cs_long.stl" scale="{STL_SCALE} {STL_SCALE} {STL_SCALE}"/>
            <texture name="ice_tex" type="2d" file="ice_type.png" />
            <material name="ice_mat" texture="ice_tex"/>
            <texture name="water" type="2d" file="Background.png" />
            <material name="water_" texture="water"/>
            <hfield name="wave_field" nrow="64" ncol="64" size="{CHANNEL_LEN/2} {CHANNEL_WID/2} 4.0 2.0">
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
                <joint damping="5"/>
             </default>
          
          </default>
          
          
          <worldbody>

            <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"/>
            <geom type="plane" size="{CHANNEL_LEN/2} {CHANNEL_WID/2} 0.1" rgba="0 0.47 0.74 1" contype="0" conaffinity="0"/>
            <!-- visual water surface (uses the height-field) -->
            <geom type="hfield" hfield="wave_field" pos="0 0 0" size="1 1 2" rgba="0 0.48 0.9 1" contype="0" conaffinity="0" />
            <geom type="plane" size="100000 100000 0.1" pos="0 0 -1" material="water_" contype="0" conaffinity="0"/>
            
            <!ASV->
            <body name="asv" pos="{ASV_X0} {ASV_Y0} 0" euler="1 0 0">
              <joint name="asv_x"   type="slide" axis="1 0 0"/>
              <joint name="asv_y"   type="slide" axis="0 1 0"/>
              <joint name="asv_yaw" type="hinge" axis="0 0 1" damping="10.0"/>
              <geom class="asv_body" type="mesh" mesh="asv_mesh" mass="{ASV_MASS_TOTAL}" euler="0 0 -180"/>
            </body>
    """)
    return header

ICE_BODY_TEMPLATE ="""
    <body name="ice_{n}" pos="{x:.2f} {y:.2f} 0">
      <joint name="ice_{n}_x"   type="slide" axis="1 0 0"/>
      <joint name="ice_{n}_y"   type="slide" axis="0 1 0"/>
      <joint name="ice_{n}_yaw" type="hinge" axis="0 0 1" damping="75.0"/>
      <geom class="ice_floe" type="box" size="{sx} {sy} {sz}" material="ice_mat"/>
    </body>"""

def random_ice_bodies(concentration: float,
                      spacing: float = 5.0,
                      max_tries: int = 2000000000) -> str:
    """
    Returning code for placing ice floes until desired concentration is covered,
    Variables:
    spacing- minimum space between ice cubes
    asv_radius- radius along which no cube is place from the origion of ship
    """
    global num_floes

    channel_area = CHANNEL_LEN * CHANNEL_WID
    target_area  = concentration * channel_area
    placed_area  = 0.0

    bodies, positions, radii = [], [], []
    grid: dict[tuple[int, int], list[int]] = {}  # cell

    # cell size chosen so that two floes in non-neighbouring cells
    # are guaranteed to be farther apart than spacing + their radii
    r_max     = 25.0 * 2**0.5
    cell_size = spacing + r_max

    def cell_of(x: float, y: float) -> tuple[int, int]:
        return (int((x + CHANNEL_LEN/2) // cell_size),
                int((y + CHANNEL_WID/2) // cell_size))

    # main rejection-sampling loop
    tries = 0
    start = time.time()
    
    while placed_area < target_area and tries < max_tries:

        # random candidate
        sx, sy = random.uniform(6.0, 25.0), random.uniform(6.0, 25.0)
        sz     = 0.6
        radius = math.hypot(sx, sy)

        x = random.uniform(-CHANNEL_LEN/2 + radius, CHANNEL_LEN/2 - radius)
        y = random.uniform(-CHANNEL_WID/2 + radius, CHANNEL_WID/2 - radius)

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

    return "\n".join(bodies)




def footer_block():
    """Setting up footer block"""
    
    footer = dedent("""\

          </worldbody>

          <actuator>
            <motor name="asv_forward" joint="asv_x"  ctrlrange="-2e7 2e7" gear="1"/>
            <motor name="asv_rudder"  joint="asv_yaw" ctrlrange="-1 1"   gear="5"/>
          </actuator>
        </mujoco>
    """)
    
    return footer



def build_xml(concentration,hfield,stl_model_path) -> str:
    return (
        header_block(hfield,stl_model_path) +
        random_ice_bodies(concentration) + "\n" +
        footer_block()
    )



# Part 2- Performing operations in the file


def apply_fluid_forces_to_body(model, data, body_name, joint_prefix, beta, Cd, area, angular_beta, phase, wave_amp=0.2, g=9.81, thickness=0.6, kx=2*np.pi/200, ky=2*np.pi/80, max_omega=5):
    """Drag force and wave force in two dimensions"""
    
    # clear out last frame’s forces only
    mujoco.mju_zero(data.qfrc_applied)        
    
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
    v_dir = v / v_mag if v_mag > 1e-3 else np.zeros(2)

    F_linear = -beta * v
    F_quad = -1.0 * RHO_WATER * Cd * area * v_mag**2 * v_dir
    Fxy_drag = F_linear + F_quad
    
    """
    # Angular velocity (yaw only)
    total_torque = np.zeros((3, 1)).astype(np.float64)
    omega_z = data.qvel[dof_yaw]
    total_torque = np.array([0, 0, -angular_beta * omega_z])
    total_torque = total_torque.reshape((3, -1))"""
    
    
    omega_z = data.qvel[dof_yaw]
    yaw_drag_linear = -angular_beta * omega_z
    yaw_drag_quad   = -0.5 * RHO_WATER * area * omega_z * abs(omega_z) * 0.1  # 0.1 is arbitrary moment-arm^2

    torque_z = yaw_drag_linear + yaw_drag_quad
    total_torque = np.array([0.0, 0.0, torque_z])
    
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
    total_Fxy = Fxy_drag
    # Only taking in two dimensions for now
    Fz_wave = 0
    
    # total force is acting at the origion of the body
    total_force = np.array([total_Fxy[0], total_Fxy[1], Fz_wave])    
    point = np.zeros((3, 1))

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

def main():
    
    # Building the heightfield used for stimulation
    hfield = hfield_data_as_string()

    # Models stored location
    stl_model_path = path_of_file(stl_model_path=None)
    
    # building XML and writing to disk
    xml_text = build_xml(concentration,hfield,stl_model_path)
    Path("asv_ice_planar_random_fixed.xml").write_text(xml_text)
    print("Wrote file with ice floes.")    

    # loading into MuJoCo
    model = mujoco.MjModel.from_xml_path("asv_ice_planar_random_fixed.xml")
    data  = mujoco.MjData(model)
    body_state = init_body_dict(model, data)
    
    # Get all geom-IDs on the body
    asv_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "asv")
    asv_geom_ids = {
        gid for gid, bid in enumerate(model.geom_bodyid)
        if bid == asv_bid
    }
    geom_to_body = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        for gid in range(model.ngeom)
    ]
    

    # control parameters- based on RL model LATER
    forward_force = 20050000.0
    phase = 0.0
    
    # launch the passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        while viewer.is_running():
            
            # thrust LATER NEED TO ADD ANGLE OF TURN AS WELL
            phase += 0.2*TIMESTEP_SIM
            data.ctrl[0] = forward_force
            
            
            # drag and wave force (ship)
            # frontal area is an apprximation here for the part of ship submerged in fluid
            apply_fluid_forces_to_body(model, data, 'asv', 'asv', ANG_DAMP_BETA,CD, 2.0, 5.0, phase)
    
            # drag and wave force (ice)
            # For now just approxed frontal area as a particular average 15 m^2 assuming half of the ice burg is submerged in fluid
            # Angular beta based on approximation to situation
            for n in range(num_floes):
                
                name = f'ice_{n}'
                apply_fluid_forces_to_body(model, data, name, name, ANG_DAMP_BETA,CD,30, 5.0, phase)            
            
            # step + render
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(TIMESTEP_SIM)            
            
            #Updating the quantities
            body_state = update_body_dict(model, data, body_state)
                
            #Reward function quantities
            
            cos_phi, num_of_collision, success= rewards(model,data,asv_bid,asv_geom_ids,geom_to_body,body_state)
            
            print(f"{'Rewards φ':>12} | {'col':>5} | {'succ':>5}")
            print("-"*12 + "-+-" + "-"*5 + "-+-" + "-"*5)
            print(f"{cos_phi:12.6f} | {num_of_collision:5d} | {str(success):5s}")
            
            # Still need to add terminated case
            if success:
                path_efficiency, interaction_effort_score  = evaluating(body_state, (ASV_X0, ASV_Y0), 500.0, success)
            
                print(f"Run finished!  Path-efficiency = {path_efficiency:.3f}   "
                      f"interaction_effort_score = {interaction_effort_score:.3f}")
                break
            
            
                    
if __name__ == "__main__":
    main()