import math

class GTSPNodeSet:
    def __init__(self):
        self.all_nodes = dict()
        self.gtsp_set_lookup = dict()

        self.gtsp_sets = list()

        self.node_counter = 1
        self.gtsp_set_counter = 0

        self.current_set = None
        self.tour = None
    
    def open_new_set(self):
        self.current_set = list()

    def add_node(self, path):
        node = GTSPNode(self.node_counter, self.gtsp_set_counter, path)
        self.all_nodes[self.node_counter] = node
        self.gtsp_set_lookup[self.node_counter] = self.gtsp_set_counter
        self.current_set.append(self.node_counter)
        self.node_counter += 1

        return node

    def close_set(self):
        current_set = self.current_set.copy()
        self.gtsp_sets.append(current_set)
        self.gtsp_set_counter += 1
        self.current_set = None
    
    def add_pushing_paths_to_nodeset(self, obs_pushing_path_set, callback=None):
        self.open_new_set()

        for path in obs_pushing_path_set:
            node = self.add_node(path.coords)
            if(callback is not None):
                callback(node)

        self.close_set()

    def add_artificial_pose_to_nodeset(self, pose, callback=None):
        position = (pose[0], pose[1])
        self.open_new_set()

        node = self.add_node((position, position))

        self.close_set()

        return node

class GTSPNode:
    def __init__(self, node_id, set_id, traversal_route):
        self.node_id = node_id
        self.set_id = set_id
        self.traversal_route = traversal_route

        x_diff = traversal_route[1][0] - traversal_route[0][0]
        y_diff = traversal_route[1][1] - traversal_route[0][1]
        self.dir_x = math.copysign(1, x_diff) if x_diff != 0 else 0
        self.dir_y = math.copysign(1, y_diff) if y_diff != 0 else 0
    
    def serialize(self):
        # Serialize only necessary info about each cell
        cells_ser = []
        for cell in self.cells:
            cell_ser = [[cell.bottom_left.x, cell.bottom_left.y], [cell.top_right.x, cell.top_right.y]]
            cells_ser.append(cell_ser)
        return [self.dir_x, self.dir_y, cells_ser]