from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray
from benchnpin.common.utils.utils import DotDict
from benchnpin.environments.box_delivery_mujoco.box_delivery_utils import generate_boxDelivery_xml, transporting, precompute_static_vertices, dynamic_vertices, receptacle_vertices, intersects_keepout
from benchnpin.common.utils.mujoco_utils import vw_to_wheels, make_controller, quat_z, inside_poly
from benchnpin.common.utils.sim_utils import get_color

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
from gymnasium import error, spaces
import matplotlib.pyplot as plt
from pathlib import Path

# SAM imports
from scipy.ndimage import distance_transform_edt, rotate as rotate_image
from cv2 import fillPoly
from skimage.draw import line
from skimage.measure import approximate_polygon
from skimage.morphology import disk, binary_dilation
import spfa

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

#Image segmentation indices
OBSTACLE_SEG_INDEX = 0
FLOOR_SEG_INDEX = 1
RECEPTACLE_SEG_INDEX = 3
CUBE_SEG_INDEX = 4
ROBOT_SEG_INDEX = 5
MAX_SEG_INDEX = 8


class BoxDeliveryMujoco(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        cfg=None,
        **kwargs,
    ):

        # get current directory of this script
        self.current_dir = os.path.dirname(__file__)

        # construct absolute path to the env_config folder
        base_cfg_path = os.path.join(self.current_dir, 'config.yaml')
        self.cfg = DotDict.load_from_file(base_cfg_path)

        if cfg is not None:
            # Update the base configuration with the user provided configuration
            for cfg_type in cfg:
                if type(cfg[cfg_type]) is DotDict or type(cfg[cfg_type]) is dict:
                    if cfg_type not in self.cfg:
                        self.cfg[cfg_type] = DotDict()
                    for param in cfg[cfg_type]:
                        self.cfg[cfg_type][param] = cfg[cfg_type][param]
                else:
                    self.cfg[cfg_type] = cfg[cfg_type]
        

        # Setting up the environment parameters
        self.room_length = self.cfg.env.room_length
        env_size = self.cfg.env.obstacle_config.strip()
        if 'small' in env_size:
            self.room_width = self.cfg.env.room_width_small
        else:
            self.room_width = self.cfg.env.room_width_large

        # Receptacle position and size
        self.receptacle_position= self.cfg.env.receptacle_position
        self.receptacle_size= self.cfg.env.receptacle_size

        # Pillar position and size
        self.num_pillars = None
        self.pillar_half = None
        self.pillar_type = None
        if env_size.strip()=="small_columns":
            self.num_pillars = self.cfg.small_pillars.num_pillars
            self.pillar_half = self.cfg.small_pillars.pillar_half
        elif env_size.strip()=="large_columns":
            self.num_pillars = self.cfg.large_pillars.num_pillars
            self.pillar_half = self.cfg.large_pillars.pillar_half

        # generate random environmnt
        xml_file = os.path.join(self.current_dir, 'turtlebot3_burger_updated.xml')
        _, self.initialization_keepouts, self.clearance_poly = generate_boxDelivery_xml (N=self.cfg.boxes.num_boxes, env_type=self.cfg.env.obstacle_config, file_name=xml_file,
                        ROBOT_clear=self.cfg.agent.robot_clear, CLEAR=self.cfg.boxes.clearance, goal_half= self.receptacle_size, goal_center= self.receptacle_position, Z_CUBE=0.02, ARENA_X=(0.0, self.room_width), 
                        ARENA_Y=(0.0, self.room_length), cube_half_size=0.04, num_pillars=self.num_pillars, pillar_half=self.pillar_half)
        
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            **kwargs,
        )


        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # environment
        self.local_map_pixel_width = self.cfg.env.local_map_pixel_width if self.cfg.train.job_type != 'sam' else self.cfg.env.local_map_pixel_width_sam
        self.local_map_width = max(self.room_length, self.room_width)
        self.local_map_pixels_per_meter = self.local_map_pixel_width / self.local_map_width
        self.wall_thickness = self.cfg.env.wall_thickness
        self.num_boxes = self.cfg.boxes.num_boxes

        # state
        self.num_channels = 4
        self.observation = None
        self.global_overhead_map = None
        self.small_obstacle_map = None
        self.configuration_space = None
        self.configuration_space_thin = None
        self.closest_cspace_indices = None
        self.observation_init= False

        # robot
        self.robot_hit_obstacle = False
        self.robot_info = self.cfg.agent
        self.robot_info['color'] = get_color('agent')
        self.robot_radius = ((self.robot_info.length**2 + self.robot_info.width**2)**0.5 / 2) * 1.2
        self.robot_half_width = max(self.robot_info.length, self.robot_info.width) / 2
        robot_pixel_width = int(2 * self.robot_radius * self.local_map_pixels_per_meter)
        self.robot_state_channel = np.zeros((self.local_map_pixel_width, self.local_map_pixel_width), dtype=np.float32)
        start = int(np.floor(self.local_map_pixel_width / 2 - robot_pixel_width / 2))
        for i in range(start, start + robot_pixel_width):
            for j in range(start, start + robot_pixel_width):
                # Circular robot mask
                if (((i + 0.5) - self.local_map_pixel_width / 2)**2 + ((j + 0.5) - self.local_map_pixel_width / 2)**2)**0.5 < robot_pixel_width / 2:
                    self.robot_state_channel[i, j] = 1

        # OBSERVATION SPACES
        self.show_observation = False
        self.observation_shape = (self.local_map_pixel_width, self.local_map_pixel_width, self.num_channels)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        # Define action space
        if self.cfg.agent.action_type == 'velocity':
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'heading':
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        elif self.cfg.agent.action_type == 'position':
            self.action_space = spaces.Box(low=0, high=self.local_map_pixel_width * self.local_map_pixel_width, dtype=np.float32)

        # get robot body & joint addresses
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        joint_adr = self.model.body_jntadr[self.base_body_id]
        self.qpos_index_base = self.model.jnt_qposadr[joint_adr]
        self.base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint")

        # Cube joint addresses
        joint_id_boxes=[]
        for i in range (self.cfg.boxes.num_boxes):
            joint_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube{i}_joint")
            joint_id_boxes.append(joint_id)
        self.joint_id_boxes = joint_id_boxes

    def save_local_map(self, obs_uint8, out_path) -> None:
        """
        Saves the local map observation as a 2x2 figure with each channel in a separate subplot.
        LATER REMOVING THIS FUCNTION AND USING RENDERING INSTEAD"""
    
        if obs_uint8.ndim != 3 or obs_uint8.shape[2] != 4:
            raise ValueError("Expected observation of shape (W, W, 4)")

        # 2×2 figure
        fig, ax = plt.subplots(2, 2, figsize=(4, 4))
        titles = ["Static", "Movable", "Goal DT", "Ergocenttric DT"]

        for k in range(4):
            i, j   = divmod(k, 2)
            ax_ij  = ax[i, j]
            ax_ij.imshow(obs_uint8[:, :, k], cmap="hot", vmin=0, vmax=255)
            ax_ij.set_xticks([]); ax_ij.set_yticks([])
            ax_ij.set_title(titles[k], fontsize=6, pad=2)

        plt.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def step(self, action):
        
        goal = action

        # navigate to the desired goal
        step_count = 0
        while True:
            step_count += 1

            v, w, dist = make_controller(self.model, self.data, self.qpos_index_base, goal)

            if dist < 0.02:
                    
                # Arrived at location
                self.data.ctrl[:] = 0.0
                print(f"Reached goal {goal}")
                break

            # otherwise drive as normal
            v_l, v_r = vw_to_wheels(v, w)

            # apply the control 'frame_skip' steps
            self.do_simulation([v_l, v_r], self.frame_skip)

            # teleporting boxes inside the receptacle
            transporting(self.model, self.data, self.joint_id_boxes, self.room_width, 
                        self.room_length,goal_half= self.receptacle_size, goal_center= self.receptacle_position, cube_half_size=0.04)

            if self.render_mode == "human" and step_count % 10 == 0:
                self.render()

        # get observation
        observation=self.generate_observation()
        reward = 0
        info = {}

        self.save_local_map(observation, "snapshots/step_023.png")

        return observation, reward, False, False, info
    

    # Observation generation functions

    def generate_observation(self, done=False):
        """ Generates the observation for the environment."""

        if done:
            return None
        
        # Getting the robot and boxes vertices
        Robot_properties, Boxes_vertices=dynamic_vertices(self.model,self.data, self.qpos_index_base,self.joint_id_boxes)

        # Initialize the global overhead map if not already done
        if not self.observation_init:

            self.update_configuration_space()
            self.global_overhead_map = self.create_padded_room_zeros()
            self.observation_init = True

        # Update the global overhead map with the current robot and boundaries
        self.update_global_overhead_map(Robot_properties, Boxes_vertices)

        Robot_postition= Robot_properties[3]
        Robot_angle = Robot_properties[2]

        # Create the robot state channel
        channels = []
        
        channels.append(self.get_local_map(self.global_overhead_map, Robot_postition, Robot_angle))
        channels.append(self.robot_state_channel)
        channels.append(self.get_local_distance_map(self.create_global_shortest_path_to_receptacle_map(self.receptacle_position), Robot_postition, Robot_angle))
        channels.append(self.get_local_distance_map(self.create_global_shortest_path_map(Robot_postition), Robot_postition, Robot_angle))
        observation = np.stack(channels, axis=2)
        observation = (observation * 255).astype(np.uint8)
        return observation
    
    def get_local_map(self, global_map, robot_position, robot_heading):
        """Takes a global map and returns a local map centered around the robot's position."""
        
        # sqrt 2 to provide some clearance around the image when the frame is rotated
        W        = self.local_map_pixel_width
        crop_W   = self.round_up_to_even(W * np.sqrt(2))
        rot_deg  = 90 - np.degrees(robot_heading)

        # robot (x,y) to pixel indices in the global map
        ci = int(np.floor(-robot_position[1] * self.local_map_pixels_per_meter + global_map.shape[0] / 2))
        cj = int(np.floor( robot_position[0] * self.local_map_pixels_per_meter + global_map.shape[1] / 2))

        # clip to valid pixel indices
        half = crop_W // 2
        i0, i1 = ci - half, ci + half
        j0, j1 = cj - half, cj + half
        
        # how many pixels to pad
        pad_top    = max(0, -i0)
        pad_left   = max(0, -j0)
        pad_bottom = max(0,  i1 - global_map.shape[0])
        pad_right  = max(0,  j1 - global_map.shape[1])

        # pad the global map if needed
        if pad_top or pad_left or pad_bottom or pad_right:
            pad_val = FLOOR_SEG_INDEX / MAX_SEG_INDEX
            global_map = np.pad(global_map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=pad_val)
            # shift window after padding
            i0 += pad_top;  i1 += pad_top
            j0 += pad_left; j1 += pad_left

        # crop the global map and rotate it
        crop = global_map[i0:i1, j0:j1]
        rot  = rotate_image(crop, rot_deg, order=0)
        mid  = rot.shape[0] // 2
        local_map = rot[mid - W // 2 : mid + W // 2,
                        mid - W // 2 : mid + W // 2]
        return local_map
    
    def get_local_distance_map(self, global_map, robot_position, robot_heading):
        """Takes a global distance map and returns a local distance map centered around the robot's position."""

        local_map = self.get_local_map(global_map, robot_position, robot_heading)
        local_map -= local_map.min() 
        return local_map

    def create_padded_room_zeros(self):
        """ Creates a padded room of zeros with the size of the local map."""

        return np.zeros((
            int(2 * np.ceil((self.room_width * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2)),
            int(2 * np.ceil((self.room_length * self.local_map_pixels_per_meter + self.local_map_pixel_width * np.sqrt(2)) / 2))
        ), dtype=np.float32)
    
    def create_global_shortest_path_to_receptacle_map(self,receptacle_position):
        """ Creates a global shortest path map to the receptacle."""

        # Create a padded room of zeros and compute the shortest path to the receptacle
        global_map = self.create_padded_room_zeros() + np.inf
        rx, ry = receptacle_position[:2]
        pixel_i, pixel_j = self.position_to_pixel_indices(rx, ry, self.configuration_space.shape)
        pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
        shortest_path_image, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))

        # Scale the shortest path image to the local map pixel width
        shortest_path_image /= self.local_map_pixels_per_meter
        global_map = np.minimum(global_map, shortest_path_image)

        # Scale the global map to the local map pixel width and invert if needed
        global_map /= (np.sqrt(2) * self.local_map_pixel_width) / self.local_map_pixels_per_meter
        global_map *= self.cfg.env.shortest_path_channel_scale

        # Invert the global map if needed
        if self.cfg.env.invert_receptacle_map:
            global_map += 1-self.configuration_space
            global_map[global_map==(1-self.configuration_space)] = 1

        return global_map
    
    def create_global_shortest_path_map(self, robot_position):
        """ Creates a global shortest path map from the robot's position."""

        pixel_i, pixel_j = self.position_to_pixel_indices(robot_position[0], robot_position[1], self.configuration_space.shape)
        pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
        global_map, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
        global_map /= self.local_map_pixels_per_meter
        global_map /= (np.sqrt(2) * self.local_map_pixel_width) / self.local_map_pixels_per_meter
        global_map *= self.cfg.env.shortest_path_channel_scale
        return global_map
    
    def update_configuration_space(self):
        """
        Obstacles are dilated based on the robot's radius to define a collision-free space
        """

        # Create a padded room of zeros to draw the walls and columns
        obstacle_map = self.create_padded_room_zeros()
        small_obstacle_map = np.zeros((self.local_map_pixel_width+20, self.local_map_pixel_width+20), dtype=np.float32)
        
        # Precompute static vertices for walls and columns
        Wall_vertices, columns_from_keepout, corners=precompute_static_vertices(self.initialization_keepouts, self.room_width, self.room_length)
        print(Wall_vertices)

        # Iterating through each wall vertice and keepout columns
        for wall_vertices_each_wall in Wall_vertices+columns_from_keepout+ corners:

            # get world coordinates of vertices
            vertices_np = np.array([[v[0], v[1]] for v in wall_vertices_each_wall[1]])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * self.local_map_pixels_per_meter).astype(np.int32)
            vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] = small_obstacle_map.shape[0] - vertices_px[:, 1]

            # draw the boundary on the small_obstacle_map
            fillPoly(small_obstacle_map, [vertices_px], color=1)
        
        start_i, start_j = int(obstacle_map.shape[0] / 2 - small_obstacle_map.shape[0] / 2), int(obstacle_map.shape[1] / 2 - small_obstacle_map.shape[1] / 2)
        obstacle_map[start_i:start_i + small_obstacle_map.shape[0], start_j:start_j + small_obstacle_map.shape[1]] = small_obstacle_map

        # Dilate obstacles and walls based on robot size
        selem = disk(np.floor(self.robot_radius * self.local_map_pixels_per_meter))
        self.configuration_space = 1 - binary_dilation(obstacle_map, selem).astype(np.float32)
        
        # Dilate obstacles and walls based on robot size just for the thin configuration space
        selem_thin = disk(np.floor(self.robot_half_width * self.local_map_pixels_per_meter))
        self.configuration_space_thin = 1 - binary_dilation(obstacle_map, selem_thin).astype(np.float32)

        # closest_cspace_indices is used to find the closest valid configuration space indices for a given position
        self.closest_cspace_indices = distance_transform_edt(1 - self.configuration_space, return_distances=False, return_indices=True)
        # Converting the value as in 1 for obstacle and 0 for free space
        self.small_obstacle_map = 1 - small_obstacle_map


    def update_global_overhead_map(self,Robot_vertices, Boxes_vertices):
        """Updates the global overhead map with the current robot and boundaries."""

        # Makes a copy of immovable obstacles
        small_overhead_map = self.small_obstacle_map.copy()

        # draw the boundary on the small_overhead_map
        small_overhead_map[small_overhead_map == 1] = FLOOR_SEG_INDEX / MAX_SEG_INDEX

        def draw_object(vertices, seg_index):

            # get world coordinates of vertices
            vertices_np = np.array([[v[0], v[1]] for v in vertices])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * self.local_map_pixels_per_meter).astype(np.int32)
            vertices_px[:, 0] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] += int(self.local_map_width * self.local_map_pixels_per_meter / 2) + 10
            vertices_px[:, 1] = small_overhead_map.shape[0] - vertices_px[:, 1]

            fillPoly(small_overhead_map, [vertices_px], color=seg_index/MAX_SEG_INDEX)
        
        # Draw the robot
        robot_vertices= Robot_vertices[1]
        draw_object(robot_vertices, ROBOT_SEG_INDEX)
        
        # Draw the boxes
        for box_vertices in Boxes_vertices:
            draw_object(box_vertices[0], CUBE_SEG_INDEX)
        
        # Draw the receptacle
        receptacle_vertice=receptacle_vertices(self.receptacle_position, self.receptacle_size)
        draw_object(receptacle_vertice, RECEPTACLE_SEG_INDEX)

        start_i, start_j = int(self.global_overhead_map.shape[0] / 2 - small_overhead_map.shape[0] / 2), int(self.global_overhead_map.shape[1] / 2 - small_overhead_map.shape[1] / 2)
        self.global_overhead_map[start_i:start_i + small_overhead_map.shape[0], start_j:start_j + small_overhead_map.shape[1]] = small_overhead_map
    
    def shortest_path(self, source_position, target_position, check_straight=False, configuration_space=None):

        if configuration_space is None:
            configuration_space = self.configuration_space

        # convert positions to pixel indices
        source_i, source_j = self.position_to_pixel_indices(source_position[0], source_position[1], configuration_space.shape)
        target_i, target_j = self.position_to_pixel_indices(target_position[0], target_position[1], configuration_space.shape)

        # check if there is a straight line path
        if check_straight:
            rr, cc = line(source_i, source_j, target_i, target_j)
            if (1 - self.configuration_space_thin[rr, cc]).sum() == 0:
                return [source_position, target_position]

        # run SPFA
        source_i, source_j = self.closest_valid_cspace_indices(source_i, source_j) # NOTE does not use the cspace passed into this method
        target_i, target_j = self.closest_valid_cspace_indices(target_i, target_j)
        _, parents = spfa.spfa(configuration_space, (source_i, source_j))

        # recover shortest path
        parents_ij = np.stack((parents // parents.shape[1], parents % parents.shape[1]), axis=2)
        parents_ij[parents < 0, :] = [-1, -1]
        i, j = target_i, target_j
        coords = [[i, j]]
        while not (i == source_i and j == source_j):
            i, j = parents_ij[i, j]
            if i + j < 0:
                break
            coords.append([i, j])

        # convert dense path to sparse path (waypoints)
        coords = approximate_polygon(np.asarray(coords), tolerance=1)

        # remove unnecessary waypoints
        new_coords = [coords[0]]
        for i in range(1, len(coords) - 1):
            rr, cc = line(*new_coords[-1], *coords[i+1])
            if (1 - configuration_space[rr, cc]).sum() > 0:
                new_coords.append(coords[i])
        if len(coords) > 1:
            new_coords.append(coords[-1])
        coords = new_coords

        # convert pixel indices back to positions
        path = []
        for coord in coords[::-1]:
            position_x, position_y = self.pixel_indices_to_position(coord[0], coord[1], configuration_space.shape)
            path.append([position_x, position_y])
        
        if len(path) < 2:
            path = [source_position, target_position]
        else:
            path[0] = source_position
            path[-1] = target_position
        
        return path

    def shortest_path_distance(self, source_position, target_position, configuration_space=None):
        """ Calculates the distance of the shortest path between two positions in the configuration space."""

        path = self.shortest_path(source_position, target_position, configuration_space=configuration_space)

        return sum(self.distance(path[i - 1], path[i]) for i in range(1, len(path)))
    
    def closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]
    
    # Helper functions
    def round_up_to_even(self, x):
        return int(np.ceil(x / 2) * 2)

    def distance(self, position1, position2):
        return np.linalg.norm(np.asarray(position1)[:2] - np.asarray(position2)[:2])

    def restrict_heading_range(self, heading):
        return np.mod(heading + np.pi, 2 * np.pi) - np.pi

    def heading_difference(self, heading1, heading2):
        return self.restrict_heading_range(heading1 - heading2)

    def position_to_pixel_indices(self, x, y, image_shape):
        """ Converts a position (x, y) in meters to pixel indices in the local map."""

        pixel_i = np.floor(image_shape[0] / 2 - y * self.local_map_pixels_per_meter).astype(np.int32)
        pixel_j = np.floor(image_shape[1] / 2 + x * self.local_map_pixels_per_meter).astype(np.int32)
        pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
        pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
        return pixel_i, pixel_j

    def pixel_indices_to_position(self, pixel_i, pixel_j, image_shape):
        """ Converts pixel indices (i, j) in the local map to a position (x, y) in meters."""

        position_x = (pixel_j - image_shape[1] / 2) / self.local_map_pixels_per_meter
        position_y = (image_shape[0] / 2 - pixel_i) / self.local_map_pixels_per_meter

        return position_x, position_y

    # Reward functions

    def _get_rew(self):
        reward = 0

        # decomposition of reward
        reward_info = {
            
        }
        return reward, reward_info
    
    def _sample_pillar_centres(self):
        """Return new (cx, cy) pairs for every pillar and the matching keep-outs."""

        if self.num_pillars is None or self.pillar_half is None:
            return [], []                        # small_empty or divider only
        centres = []
        rng = np.random.default_rng()
        tries = 0
        while len(centres) < self.num_pillars and tries < 50_000:
            tries += 1
            cx = rng.uniform(0.60, self.room_width  - 0.60)
            cy = rng.uniform(0.60, self.room_length - 0.60)

            # four corners of this pillar
            hx, hy, _ = self.pillar_half
            corners = [(cx-hx, cy-hy), (cx+hx, cy-hy),
                        (cx+hx, cy+hy), (cx-hx, cy+hy)]
            if not all(inside_poly(x, y, self.clearance_poly) for x, y in corners):
                continue
            if any(np.hypot(cx-px, cy-py) < 2*hx+0.30 for px, py in centres):
                continue
            centres.append((cx, cy))

        keep_out = [[(cx-hx, cy-hy), (cx+hx, cy-hy),
                        (cx+hx, cy+hy), (cx-hx, cy+hy)] for cx, cy in centres]
        return centres, keep_out

    def _get_obs(self):

        observation = np.zeros((100, 100)).astype(np.uint8)
        return observation


    def reset_model(self):
        """
        Randomly sample non-overlapping (x, y, theta) for robot and boxes.
        Teleport them in simulation using sim.data.qpos.

        Args:
            robot_r (float): Robot radius, used for clearance check.
            clearance (float): Minimum distance between any two boxes.
        """
        positions = []

        # pillars
        pillar_centres, pillar_keepouts = self._sample_pillar_centres()
        self.initialization_keepouts = pillar_keepouts        # for c-space
        
        # teleport each pillar body to its new random centre
        for k, (cx, cy) in enumerate(pillar_centres):
            prefix = "small_col" if "small" in self.cfg.env.obstacle_config else "large_col"
            name = f"{prefix}{k}"

            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                     f"{name}_joint")
        
            if j_id != -1:
                adr = self.model.jnt_qposadr[j_id]
                self.data.qpos[adr:adr+3]   = [cx, cy, self.pillar_half[2]]
                self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]     # zero yaw
                self.data.qvel[adr:adr+6]   = 0

        def is_valid(pos, radius):
            # unpack immediately
            x, y, _ = pos

            # 1) check against previously placed robot/boxes
            for (px, py, _), pr in positions:
                dist = np.hypot(x - px, y - py)
                if dist < (pr + radius + self.cfg.boxes.clearance):
                    return False

            # 2) check against pillar keep-outs
            if intersects_keepout(x, y, self.initialization_keepouts):
                return False

            # 3) finally check your overall clearance polygon
            if not inside_poly(x, y, self.clearance_poly):
                return False

            return True

        # Define bounds of the placement area (slightly inside the walls)
        x_min, x_max = 0.2,self.room_width - 0.2
        y_min, y_max = 0.2,self.room_length - 0.2

        # Sample robot pose
        while True:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            theta = np.random.uniform(-np.pi, np.pi)
            if is_valid((x, y, theta), self.cfg.agent.robot_clear):
                positions.append(((x, y, theta), self.cfg.agent.robot_clear))
                break

        # Set robot pose
        base_qpos_addr = self.model.jnt_qposadr[self.base_joint_id]
        self.data.qpos[base_qpos_addr:base_qpos_addr+3] = [x, y, 0.01]  # x, y, z
        self.data.qpos[base_qpos_addr+3:base_qpos_addr+7] = quat_z(theta)

        self.data.qvel[base_qpos_addr:base_qpos_addr+6] = 0

        # Assume box is square with radius from center to corner (diagonal/2)
        box_r = np.sqrt(0.04 ** 2 + 0.04 ** 2)

        for i in range(self.num_boxes):
            while True:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                theta = np.random.uniform(-np.pi, np.pi)
                if is_valid((x, y, theta), box_r):
                    positions.append(((x, y, theta), box_r))
                    break
            
            qadr = self.model.jnt_qposadr[self.joint_id_boxes[i]]
            self.data.qpos[qadr:qadr+2] = np.array([x, y])
            self.data.qpos[qadr+3:qadr+7] = quat_z(theta)
            self.data.qvel[qadr:qadr+6] = 0

        observation = self._get_obs()
        return observation


    def _get_reset_info(self):
        return {
        }
