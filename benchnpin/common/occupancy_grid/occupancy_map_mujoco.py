import numpy as np
import cv2
import math
from skimage.draw import draw
from skimage.measure import block_reduce
from benchnpin.common.geometry.polygon import poly_centroid
from scipy.ndimage import rotate

class OccupancyGrid:

    def __init__(self, grid_width, grid_height, map_width, map_height, local_width=6, local_height=6, ship_body=None, meter_to_pixel_scale=50) -> None:
        """
        grid_width, grid_height, map_width, map_height are in meter units
        ship body info at starting position
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.map_width = map_width
        self.map_height = map_height
        self.occ_map_width = int(self.map_width / self.grid_width)         # number of grids in x-axis
        self.occ_map_height = int(self.map_height / self.grid_height)      # number of grids in y-axis
        self.meter_to_pixel_scale = meter_to_pixel_scale

        self.local_width = local_width
        self.local_height = local_height
        self.local_window_height = int(self.local_height * meter_to_pixel_scale)            # local window height (unit: cell)
        self.local_window_width = int(self.local_width * meter_to_pixel_scale)            # local window width (unit: cell)

        # print("Occupancy map resolution: ", grid_width, "; occupancy map dimension: ", (self.occ_map_width, self.occ_map_height))

        self.occ_map = np.zeros((self.occ_map_height, self.occ_map_width))
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        self.obstacle_centroids = np.zeros((self.occ_map_height, self.occ_map_width))
        self.swath = np.zeros((self.occ_map_height, self.occ_map_width))


    def compute_occ_img(self, obstacles, ice_binary_w=235, ice_binary_h=774, local_range=None, ship_state=None):
        meter_to_pixel_scale = self.meter_to_pixel_scale

        raw_ice_binary = np.zeros((ice_binary_h, ice_binary_w))

        for obstacle in obstacles:

            if local_range is not None:
                robot_x, robot_y = ship_state[:2]
                range_x, range_y = local_range
                center_x, center_y = np.abs(poly_centroid(obstacle))
                if abs(robot_x - center_x) > range_x or abs(robot_y - center_y) > range_y:
                    continue

            obstacle = np.asarray(obstacle) * meter_to_pixel_scale

            # get pixel coordinates on costmap that are contained inside obstacle/polygon
            rr, cc = draw.polygon(obstacle[:, 1], obstacle[:, 0], shape=raw_ice_binary.shape)

            # skip if 0 area
            if len(rr) == 0 or len(cc) == 0:
                continue

            raw_ice_binary[rr, cc] = 1.0
        
        # occ_val = np.sum(raw_ice_binary) / (raw_ice_binary.shape[0] * raw_ice_binary.shape[1])
        # print("occ concentration: ", occ_val)
        
        return raw_ice_binary
    
    def compute_occ_img_walls(self, walls, width , height):
        meter_to_pixel_scale = self.meter_to_pixel_scale
        raw_wall_binary = np.zeros((height, width))

        for wall in walls:
            vertices = wall * meter_to_pixel_scale

            # get pixel coordinates on costmap that are contained inside obstacle/polygon
            rr, cc = draw.polygon(vertices[:, 1], vertices[:, 0], shape=raw_wall_binary.shape)

            # skip if 0 area
            if len(rr) == 0 or len(cc) == 0:
                continue

            raw_wall_binary[rr, cc] = 1.0

        return raw_wall_binary


    def compute_con_gridmap(self, raw_ice_binary, save_fig_dir=None):
        """
        Compute concentration grid map
        """
        meter_to_pixel_scale_y = self.meter_to_pixel_scale
        meter_to_pixel_scale_x = self.meter_to_pixel_scale

        
        block_size = (int(self.grid_height * meter_to_pixel_scale_y), int(self.grid_width * meter_to_pixel_scale_x))
        occ_map = block_reduce(raw_ice_binary, block_size, np.mean)

        self.occ_map = occ_map
        return occ_map


    def ego_view_obstacle_map(self, raw_ice_binary, ship_state, vertical_shift):

        global_obstacle_map = self.compute_con_gridmap(raw_ice_binary=raw_ice_binary)

        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        local_obstacle_map = np.zeros((self.local_window_height, self.local_window_width))

        robot_x, robot_y = ship_state[:2]
        window_x, window_y = robot_x, robot_y + vertical_shift      # shifting the local window upward

        window_x = int(window_x * meter_to_grid_scale_x)                              # center of local window on global window (unit: cell)
        window_y = int(window_y * meter_to_grid_scale_y)

        for local_i in range(self.local_window_height):
            for local_j in range(self.local_window_width):

                global_i = int(local_i + window_y - (self.local_window_height / 2))
                global_j = int(local_j + window_x - (self.local_window_width / 2))

                # check out-of-bound
                if global_i < 0 or global_i >= global_obstacle_map.shape[0] or global_j < 0 or global_j >= global_obstacle_map.shape[1]:
                    continue

                local_obstacle_map[local_i, local_j] = global_obstacle_map[global_i, global_j]
        
        self.local_obstacle_map = local_obstacle_map  
        return local_obstacle_map

    def ego_view_map_maze(self, agent_state, agent_vertices, obstacles, maze_walls , global_dist_map):
        #get local window of movable agent
        #the center of the local window is the agent's position
        #This method is different from ego_view_obstacle_map() because it calls compute_occ_img() instead of compute_con_gridmap()
        global_obstacle_map = self.compute_occ_img(obstacles , ice_binary_w=self.occ_map_width, ice_binary_h=self.occ_map_height)
        
        global_wall_map = self.compute_occ_img_walls(maze_walls, self.occ_map_width, self.occ_map_height)

        robot_global_footprint = self._compute_global_footprint_maze(agent_state, agent_vertices)
        #robot position in meter with respect to the global map
        robot_x, robot_y = agent_state[:2]
        #center the local window on the agent's position 
        window_x, window_y = int(robot_x * self.meter_to_pixel_scale), int(robot_y * self.meter_to_pixel_scale)

        inflation = max(self.local_window_width, self.local_window_height) // 2
        inflated_local_height = self.local_window_height + inflation
        inflated_local_width = self.local_window_width + inflation

        #initialize the local robot footprint as zeros (out of bound = 0.0)
        local_footprint = np.zeros((inflated_local_height, inflated_local_width))
        #initialize the local obstacle map as zeros (out of bound = 0.0)
        local_obstacle_map = np.zeros((inflated_local_height, inflated_local_width))
        #initialize the local wall map as zeros (out of bound = 0.0)
        local_wall_map = np.zeros((inflated_local_height, inflated_local_width))
        #initialize the local global distance map as ones (out of bound = 1.0) higher cost for out of bound
        local_dist_map = np.ones((inflated_local_height, inflated_local_width))

        for local_i in range(inflated_local_height):
            for local_j in range(inflated_local_width):
                global_i = int(local_i + window_y - (inflated_local_height / 2))
                global_j = int(local_j + window_x - (inflated_local_width / 2))
                # check out-of-bound
                if global_i < 0 or global_i >= global_obstacle_map.shape[0] or global_j < 0 or global_j >= global_obstacle_map.shape[1]:
                    continue #zero padding for out of bound
                
                # print("global: ", robot_global_footprint)
                # print("local: ", local_footprint)
                local_footprint[local_i, local_j] = robot_global_footprint[global_i, global_j]   
                local_obstacle_map[local_i, local_j] = global_obstacle_map[global_i, global_j]
                local_wall_map[local_i, local_j] = global_wall_map[global_i, global_j]
                local_dist_map[local_i, local_j] = global_dist_map[global_i, global_j]

        corrected_heading = agent_state[2] - np.pi / 2
        rotate_deg = corrected_heading * (180 / np.pi)       # convert to degree
        local_footprint = rotate(local_footprint, rotate_deg, reshape=False, cval=0, order=1)
        local_obstacle_map = rotate(local_obstacle_map, rotate_deg, reshape=False, cval=0, order=1)
        local_wall_map = rotate(local_wall_map, rotate_deg, reshape=False, cval=0, order=1)
        local_dist_map = rotate(local_dist_map, rotate_deg, reshape=False, cval=1, order=1)

        half_inflate = inflation // 2
        local_footprint = local_footprint[half_inflate:half_inflate+self.local_window_height, half_inflate:half_inflate+self.local_window_width]
        local_obstacle_map = local_obstacle_map[half_inflate:half_inflate+self.local_window_height, half_inflate:half_inflate+self.local_window_width]
        local_wall_map = local_wall_map[half_inflate:half_inflate+self.local_window_height, half_inflate:half_inflate+self.local_window_width]
        local_dist_map = local_dist_map[half_inflate:half_inflate+self.local_window_height, half_inflate:half_inflate+self.local_window_width]

        self.local_footprint = local_footprint
        self.local_obstacle_map = local_obstacle_map
        self.local_wall_map = local_wall_map
        self.local_dist_map = local_dist_map


        return local_footprint, local_obstacle_map, local_wall_map, local_dist_map




    def compute_ship_footprint(self, body, ship_vertices, padding=0.25):
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        # # apply padding as in a* search
        # ship_vertices = np.asarray(
        #     [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in ship_vertices]
        # )

        # ship vertices in meter
        heading = body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(body.position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)

        rr, cc = draw.polygon(r=r, c=c)
        self.footprint[rr, cc] = 1.0


    def compute_goal_image(self, goal_y):
        self.goal_img = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_y = self.occ_map_height / self.map_height
        goal_y_idx = int(goal_y * meter_to_grid_scale_y)
        self.goal_img[goal_y_idx] = 1.0

    def compute_goal_point_image(self, goal):
        goal_x, goal_y = goal
        self.goal_img = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height
        goal_x_idx = int(goal_x * meter_to_grid_scale_x)
        goal_y_idx = int(goal_y * meter_to_grid_scale_y)
        self.goal_img[goal_y_idx, goal_x_idx] = 1.0
    
    def compute_ship_footprint_planner(self, ship_state, ship_vertices, padding=0.25):
        """
        NOTE this function computes current ship footprint similarily to self.compute_ship_footprint()
        but is intended for generating observations for planners 
        :param ship_state: (x, y, theta) where x, y are in meter and theta in radian
        :param ship_vertices: original unscaled, unpadded ship vertices
        """
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # # apply padding as in a* search
        # ship_vertices = np.asarray(
        #     [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in ship_vertices]
        # )

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)
        
        # it is possible that the ship state is outside of the grid map
        if len(r) == 0 or len(c) == 0:
            # print("ship outside the costmap!!!")
            return
        
        rr, cc = draw.polygon(r=r, c=c)
        
        self.footprint[rr, cc] = 1.0



    def _compute_global_footprint(self, ship_state, ship_vertices, padding=0.25):
        """
        This function computes a global footprint map. 
        Values correspondences: free space 0.5, robot 1.0, out-of-bound 0.0
        """
        global_footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        global_footprint = global_footprint + 0.5                   # free space 0.5, robot 1.0, out-of-bound 0.0
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)
        
        # it is possible that the ship state is outside of the grid map
        if len(r) == 0 or len(c) == 0:
            return None
        
        rr, cc = draw.polygon(r=r, c=c)
        
        global_footprint[rr, cc] = 1.0
        return global_footprint


    def _compute_global_footprint_maze(self, ship_state, ship_vertices, padding=0.25):
        """
        This function computes a global footprint map. 
        Values correspondences: robot 1.0, elsewhere 0.0
        """
        global_footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)
        
        # it is possible that the ship state is outside of the grid map
        if len(r) == 0 or len(c) == 0:
            print(self.occ_map_height, self.occ_map_width)
            return None
        
        rr, cc = draw.polygon(r=r, c=c)
        
        global_footprint[rr, cc] = 1.0
        return global_footprint


    def ego_view_footprint(self, ship_state, ship_vertices, vertical_shift=2): 
        """
        This function computes a ego-centric footprint crop, based on the global footprint from _compute_global_footprint()
        Values correspondences: free space 0.5, robot 1.0, out-of-bound 0.0
        """
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        local_footprint = np.zeros((self.local_window_height, self.local_window_width))

        robot_x, robot_y = ship_state[:2]
        window_x, window_y = robot_x, robot_y + vertical_shift      # shifting the local window upward

        window_x = int(window_x * meter_to_grid_scale_x)                              # center of local window on global window (unit: cell)
        window_y = int(window_y * meter_to_grid_scale_y)

        global_footprint = self._compute_global_footprint(ship_state, ship_vertices)

        for local_i in range(self.local_window_height):
            for local_j in range(self.local_window_width):

                global_i = int(local_i + window_y - (self.local_window_height / 2))
                global_j = int(local_j + window_x - (self.local_window_width / 2))

                # check out-of-bound
                if global_i < 0 or global_i >= global_footprint.shape[0] or global_j < 0 or global_j >= global_footprint.shape[1]:
                    continue

                local_footprint[local_i, local_j] = global_footprint[global_i, global_j]
        
        self.local_footprint = local_footprint  
        return local_footprint

    
    def global_goal_dist_transform(self, goal_y):
        """
        Compute a global normalized goal-line distance transform image
        """
        global_edt = np.zeros((self.occ_map_height, self.occ_map_width))
        grid_to_meter_scale_y = self.map_height / self.occ_map_height

        for i in range(self.occ_map_height):
            for j in range(self.occ_map_width):
                
                # compute distance to goal in meters
                dist_to_goal = goal_y - i * grid_to_meter_scale_y
                if dist_to_goal < 0: 
                    dist_to_goal = 0

                # normalize to prevent covariance shift
                nomalized_dist_to_goal = dist_to_goal / goal_y

                global_edt[i, j] = nomalized_dist_to_goal

        return global_edt

    def global_goal_point_dist_transform(self, goal, walls,wall_radius = 0.5,connectivity= "8-connectivity"):
        """
        Compute a global normalized goal-point distance transform image using wavefront algorithm
        """
        grid_to_meter_scale = self.map_width / self.occ_map_width
        goal_x, goal_y = goal
        goal_x_idx = int(goal_x / grid_to_meter_scale)
        goal_y_idx = int(goal_y / grid_to_meter_scale)

        global_edt = np.zeros((self.occ_map_height, self.occ_map_width))
        global_edt[goal_y_idx, goal_x_idx] = 1.0

        # compute occupancy of the walls
        global_wall_binary = self.compute_occ_img_walls(walls, self.occ_map_width, self.occ_map_height)
        
        # neighbors 
        if connectivity == "4-connectivity":
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        elif connectivity == "8-connectivity":
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # wavefront algorithm
        queue = [(goal_y_idx, goal_x_idx)]
        visited = set()
        visited.add((goal_y_idx, goal_x_idx))

        while queue:
            y, x = queue.pop(0)
            for dy, dx in neighbors:
                new_y, new_x = y + dy, x + dx
                # check if neighbor is out-of-bound
                if new_y < 0 or new_y >= self.occ_map_height or new_x < 0 or new_x >= self.occ_map_width:
                    continue
                # check if neighbor is already visited
                if (new_y, new_x) in visited:
                    continue
                # check if neighbor is inside the wall
                if global_wall_binary[new_y, new_x] == 1.0:
                    continue
                # otherwise, update the distance, add to visited and queue
                visited.add((new_y, new_x))
                global_edt[new_y, new_x] = global_edt[y, x] + 1
                queue.append((new_y, new_x))

        #normalize the distance
        normalized_global_edt = global_edt / global_edt.max()

        #add walls to the normalized distance map as 1.0
        normalized_global_edt[global_wall_binary == 1.0] = 1.0

        return normalized_global_edt, global_edt
        



            
    
    def ego_view_goal_dist_transform(self, goal_y, ship_state, vertical_shift=2):
        """
        Compute an ego-centric local crop of a global goal-line distance transform from global_goal_dist_transform()
        """
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        global_edt = self.global_goal_dist_transform(goal_y=goal_y)
        local_edt = np.ones((self.local_window_height, self.local_window_width))

        robot_x, robot_y = ship_state[:2]
        window_x, window_y = robot_x, robot_y + vertical_shift      # shifting the local window upward

        window_x = int(window_x * meter_to_grid_scale_x)                              # center of local window on global window (unit: cell)
        window_y = int(window_y * meter_to_grid_scale_y)

        for local_i in range(self.local_window_height):
            for local_j in range(self.local_window_width):

                global_i = int(local_i + window_y - (self.local_window_height / 2))
                global_j = int(local_j + window_x - (self.local_window_width / 2))

                # check out-of-bound
                if global_i < 0 or global_i >= global_edt.shape[0] or global_j < 0 or global_j >= global_edt.shape[1]:
                    continue

                local_edt[local_i, local_j] = global_edt[global_i, global_j]
        
        self.local_edt = local_edt  
        return local_edt


    def global_orientation_map(self, ship_state, head, tail):
        """
        Compute a global orientation map. This is a grayscale map with a single line indicating the orientation of the ship
        """
        global_orientation = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])

        # get global position for ship head and tail
        head_pos = np.array(head) @ R.T + np.array(position)
        tail_pos = np.array(tail) @ R.T + np.array(position)

        # convert to grid coordinate
        head_pix = np.array([head_pos[0] * meter_to_grid_scale_x, head_pos[1] * meter_to_grid_scale_y]).astype(np.uint16)       # (x, y)
        tail_pix = np.array([tail_pos[0] * meter_to_grid_scale_x, tail_pos[1] * meter_to_grid_scale_y]).astype(np.uint16)       # (x, y)

        cv2.line(global_orientation, head_pix, tail_pix, color=0.5, thickness=1)

        head_pix = np.clip(head_pix, a_min=[0, 0], a_max=[global_orientation.shape[1] - 1, global_orientation.shape[0] - 1])
        global_orientation[head_pix[1], head_pix[0]] = 1.0              # mark the head

        return global_orientation


    def ego_view_orientation_map(self, ship_state, head, tail, vertical_shift):
        """
        Compute an ego-centric local crop of the global orientation map, computed from global_orientation_map()
        """
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        local_orientation = np.zeros((self.local_window_height, self.local_window_width))

        robot_x, robot_y = ship_state[:2]
        window_x, window_y = robot_x, robot_y + vertical_shift      # shifting the local window upward

        window_x = int(window_x * meter_to_grid_scale_x)                              # center of local window on global window (unit: cell)
        window_y = int(window_y * meter_to_grid_scale_y)

        global_orientation = self.global_orientation_map(ship_state, head, tail)

        for local_i in range(self.local_window_height):
            for local_j in range(self.local_window_width):

                global_i = int(local_i + window_y - (self.local_window_height / 2))
                global_j = int(local_j + window_x - (self.local_window_width / 2))

                # check out-of-bound
                if global_i < 0 or global_i >= global_orientation.shape[0] or global_j < 0 or global_j >= global_orientation.shape[1]:
                    continue

                local_orientation[local_i, local_j] = global_orientation[global_i, global_j]
        
        self.local_orientation = local_orientation  
        return local_orientation