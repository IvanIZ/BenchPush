import math
import numpy as np

from collections import defaultdict

from benchpush.baselines.area_clearing.planning_based.GTSPPlanner.gtsp_nodeset import GTSPNodeSet

# import json

# Define property constants

NAME = 'NAME'
TYPE = 'TYPE'
COMMENT = 'COMMENT'
DIMENSION = 'DIMENSION'
GTSP_SETS = 'GTSP_SETS'
EDGE_WEIGHT_TYPE = 'EDGE_WEIGHT_TYPE'
EDGE_WEIGHT_FORMAT = 'EDGE_WEIGHT_FORMAT'

EDGE_WEIGHT_SECTION = 'EDGE_WEIGHT_SECTION'

GTSP_SET_SECTION = 'GTSP_SET_SECTION'

END_STR = 'EOF'

class GTSPFileProperties:

    def __init__(self):

        # Using properties for easier usage
        # Would be nice to have explicit setters/functional generator
        self.file_name = None
        self.file_type = None
        self.comment = None
        self.dimension = None
        self.gtsp_sets = None
        self.edge_weight_type = None
        self.edge_weight_format = None

        self.init_tour_file_name = None

    def get_dict(self):

        file_properties = {
            NAME : self.file_name,
            TYPE : self.file_type,
            COMMENT : self.comment,
            DIMENSION : self.dimension,
            GTSP_SETS : self.gtsp_sets,
            EDGE_WEIGHT_TYPE : self.edge_weight_type,
            EDGE_WEIGHT_FORMAT : self.edge_weight_format
        }

        return file_properties

class GTSPFileGenerator:

    def __init__(self, gtsp_prop, start_pose=None):
        self.gtsp_prop = gtsp_prop
        self.start_pose = start_pose

    def create_obs_pushing_nodes(self, obs_to_push, obs_pushing_paths):
        nodeset = GTSPNodeSet()

        for i in range(len(obs_to_push)):
            nodeset.add_pushing_paths_to_nodeset(obs_pushing_paths[i])

        # with open('gtsp_node_mapping.json', 'w') as file:
        #     file.write(json.dumps(all_nodes)) # use `json.loads` to do the reverse

        return nodeset

    def create_artificial_node(self, pose, nodeset=GTSPNodeSet()):
        path_to_node_lookup = defaultdict(int)

        node = nodeset.add_artificial_pose_to_nodeset(pose)

        return (nodeset, path_to_node_lookup, node)


    def generate_asymmetric_cost_mat(self, all_nodes, gtsp_set_lookup, traversal_lookups):

        SELF_WEIGHT = 99999999
        
        edge_weight_matrix = []
        missed_transitions = 0
        
        for key_1, node in all_nodes.items():
            
            end_point_1 = node.traversal_route[-1]

            gtsp_set_1 = gtsp_set_lookup[key_1]
            
            row = []

            for key_2 in all_nodes.keys():
                gtsp_set_2 = gtsp_set_lookup[key_2]

                if(key_1 == key_2):
                    row.append(SELF_WEIGHT)
                elif(gtsp_set_1 == gtsp_set_2):
                    row.append(0)
                else:
                    start_point_2 = all_nodes[key_2].traversal_route[0]
                    try:
                        weight = traversal_lookups[(end_point_1, start_point_2)]
                        if(weight is None):
                            ### TODO: SOMETHING WRONG WITH THE START POINT EDGES
                            ### HACK: REPLACE WITH EUCLIDEAN EDGES
                            weight = math.sqrt((start_point_2[0] - end_point_1[0])**2 + (start_point_2[1] - end_point_1[1])**2)
                        if(weight == math.inf):
                            weight = SELF_WEIGHT
                    except:
                        # Megnath: This is happening for some reason. Need to figure out how many nodes are affected. Until then...
                        weight = SELF_WEIGHT
                        missed_transitions += 1
                        print('Missed transition')
                    row.append(weight)
            edge_weight_matrix.append(row)

        edge_weight_matrix = np.vstack(edge_weight_matrix)
        print('Missed transitions: ' + str(missed_transitions))

        # File dumps for debugging
        # np.savetxt('edge_weight_mat.csv', edge_weight_matrix, delimiter=",")
        # np.savetxt('gtsp_sets.csv', gtsp_sets, delimiter=",")

        return(edge_weight_matrix)
            
    def write_to_file(self, edge_weight_matrix, gtsp_sets):
        file_name = self.gtsp_prop.file_name

        if(file_name is None):
            raise ValueError('Invalid File Name')

        gtsp_prop_dict = self.gtsp_prop.get_dict()
        
        with open((file_name + '.gtsp'), 'w') as f:
            for key, value in gtsp_prop_dict.items():
                f.write(str(key) + ': ' + str(value) + '\n')
            
            # Edge Weight Section
            f.write(str(EDGE_WEIGHT_SECTION) + '\n')
            for edge in edge_weight_matrix:
                row = ''
                for weight in edge:
                    row += '%5i '%round(weight*100)
                f.write(row + '\n')

            # GTSP Set Section
            set_counter = 1

            f.write(str(GTSP_SET_SECTION) + '\n')
            for set in gtsp_sets:
                row = str(set_counter) + ' '
                for vertex in set:
                    row += str(vertex) + ' '
                row += '-1 '
                f.write(row + '\n')

                set_counter += 1
            
            # End file
            f.write(END_STR)

    def generate_gtsp_instance(self, obs_to_push, obs_pushing_paths, traversal_lookups, robot_pose):

        nodeset = self.create_obs_pushing_nodes(obs_to_push, obs_pushing_paths)
        
        robot_node = None
        path_to_node_lookup = dict()
        
        if(robot_pose):
            nodeset, path_to_node_lookup, robot_node = self.create_artificial_node(robot_pose, nodeset)

        self.gtsp_prop.dimension = nodeset.node_counter - 1
        self.gtsp_prop.gtsp_sets = nodeset.gtsp_set_counter

        edge_weight_matrix = self.generate_asymmetric_cost_mat(nodeset.all_nodes, nodeset.gtsp_set_lookup, traversal_lookups)
        self.write_to_file(edge_weight_matrix, nodeset.gtsp_sets)

        print('GTSP File Created!')

        return nodeset, path_to_node_lookup, robot_node
    