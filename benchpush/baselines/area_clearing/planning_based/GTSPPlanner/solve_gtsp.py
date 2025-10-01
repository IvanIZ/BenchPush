import os
from math import atan2

from benchpush.baselines.area_clearing.planning_based.GTSPPlanner.create_instance import GTSPFileProperties, GTSPFileGenerator
from benchpush.baselines.area_clearing.planning_based.GTSPPlanner.gtsp_nodeset import GTSPNodeSet

debug = False

class GTSPSolver:

    def __init__(self, glns_executable_path):
        self.gtsp_nodeset = None
        self.glns_executable_path = glns_executable_path

    def solve_GTSP_Problem(self, obs_to_push, obs_pushing_paths, transition_graph_lookup, robot_pose = None, time_limit = None):    

        gtsp_prop = GTSPFileProperties()
        gtsp_prop.file_name = 'AreaClearingGTSP'
        gtsp_prop.file_type = 'AGTSP'
        gtsp_prop.comment = 'GTSP file instance for area clearing tasks'
        gtsp_prop.edge_weight_type = 'EXPLICIT'
        gtsp_prop.edge_weight_format = 'FULL_MATRIX'

        generator = GTSPFileGenerator(gtsp_prop)

        self.gtsp_nodeset, _, robot_node = generator.generate_gtsp_instance(obs_to_push, obs_pushing_paths, transition_graph_lookup.cost_lookup, robot_pose)

        if(time_limit):
            tour, time_found = self.find_tour_with_time_limit(gtsp_prop, time_limit)
        else:
            tour, time_found = self.find_classic_tour(gtsp_prop)
        output_tuple = self.evaluate_tour(tour, transition_graph_lookup, robot_node)

        return output_tuple, time_found
    
    def find_classic_tour(self, gtsp_prop):
        gtsp_command = self.glns_executable_path + ' ' + gtsp_prop.file_name + '.gtsp -output=tour.txt'
        return GTSPSolver.run_GTSP_command(gtsp_command)

    def find_tour_with_time_limit(self, gtsp_prop, time_limit=1000):
        # Time limit unit - s
        gtsp_command = self.glns_executable_path + ' ' + gtsp_prop.file_name + '.gtsp -output=tour.txt -max_time=' + str(time_limit) + ' -trials=1000'
        return GTSPSolver.run_GTSP_command(gtsp_command)
    
    def find_tour_with_seed(self, gtsp_prop, seed_tour):
        with open(gtsp_prop.init_tour_file_name, 'w') as file:
            for node_id in seed_tour:
                file.write('%s, ' % node_id)
        
        gtsp_command = self.glns_executable_path + ' ' + gtsp_prop.file_name + '.gtsp --init_tour_file=' + gtsp_prop.init_tour_file_name + ' -output=tour.txt'
        return GTSPSolver.run_GTSP_command(gtsp_command)


    @staticmethod
    def run_GTSP_command(gtsp_command):

        stream = os.popen(gtsp_command)
        output = stream.read()
        
        # Source: Stanislav Bochkarev
        with open('tour.txt', 'r') as f:
            for i in range(7):
                f.readline()

            temp_str_tour = f.readline()
            _, str_array_tour = temp_str_tour.split(': ')
            str_array_tour = str_array_tour.strip('][\n').split(', ')
            tour = list()
            for node in str_array_tour:
                tour.append(int(node))
            # Append the first node again to get the circular tour
            tour.append(tour[0])

            temp_str_time = f.readline()
            _, str_array_time = temp_str_time.split(': ')
            time_array = str_array_time.strip('][\n').split(', ')
            time_found = time_array[0]
        
        return tour, time_found

    def evaluate_tour(self, tour, transition_graph_lookup, robot_node):

        all_gtsp_nodes = self.gtsp_nodeset.all_nodes
        
        reverse_tour = False

        if(robot_node):
            print('Robot Node:', robot_node.node_id)
            start_indices = []
            artificial_indices = []
            for i in range(len(tour)):
                cur_node = all_gtsp_nodes[tour[i]]
                if(cur_node.node_id == robot_node.node_id):
                    start_indices.append(i)
                    artificial_indices.append(i)

            print('Start indices:', start_indices)
            
            start_index = start_indices[0]
            rearranged_tour = []
            i = 0
            j = start_index

            rearranged_indices = []

            while i < len(tour):
                i += 1
                if(not(j in artificial_indices)):
                    if(len(rearranged_tour) == 0 or not(rearranged_tour[-1] == tour[j])):
                        rearranged_tour.append(tour[j])
                        rearranged_indices.append(j)
                j = (j + 1) % len(tour)
            
            # rearranged_tour.append(rearranged_tour[0])

            if(reverse_tour):
                rearranged_tour.reverse()
                rearranged_indices.reverse()

            if(debug):
                print(rearranged_indices)
        else:
            rearranged_tour = tour

        transition_paths = []
        transition_costs = []
        transition_cost = 0
        transition_length = 0
        total_angle = 0
        
        current_node = all_gtsp_nodes[rearranged_tour[0]]
        final_nodes = [current_node]

        if(reverse_tour):
            current_node.should_reverse = True

        # ADD START NODE
        if(robot_node):
            start_node_pt = robot_node.traversal_route[-1]
            pt_id = 1 if reverse_tour else 0
            current_node_pt = current_node.traversal_route[pt_id]

        for i in range(len(rearranged_tour) - 1):

            next_node = all_gtsp_nodes[rearranged_tour[i+1]]

            # Are we using any blocked edges?
            if(reverse_tour):
                # Reverse the cells 
                next_node.should_reverse = True
                
                current_node_pt = current_node.traversal_route[0]
                next_node_pt = next_node.traversal_route[-1]
            else:
                current_node_pt = current_node.traversal_route[-1]
                next_node_pt = next_node.traversal_route[0]

            try:    
                cost, length, angle = self.evaluate_transition(current_node_pt, next_node_pt, transition_graph_lookup, transition_paths)

                transition_cost += cost
                transition_length += length
                total_angle += angle

                transition_costs.append(cost)

                final_nodes.append(next_node)
                current_node = next_node
            except Exception as e:
                print('Missed transition: ' + str(current_node_pt) + ' ' + str(next_node_pt))
                print('Index:', i)
                raise(e)

        # ADD END NODE
        # if(end_node):
        #     pt_id = 0 if reverse_tour else 1
        #     current_node_pt = current_node.traversal_route[pt_id]
        #     end_node_pt = end_node.traversal_route[0]

        #     cost, length, turns, angle = self.evaluate_transition(current_node_pt, end_node_pt, transition_graph_lookup, transition_paths)

        self.gtsp_nodeset.tour = final_nodes

        return (final_nodes, transition_paths, transition_cost, transition_length, total_angle, transition_costs)

    def evaluate_transition(self, current_node_pt, next_node_pt, transition_graph_lookup, transition_paths):
        
        cost = transition_graph_lookup.cost_lookup[(current_node_pt, next_node_pt)]
        length = transition_graph_lookup.length_lookup[(current_node_pt, next_node_pt)]
        angle = transition_graph_lookup.angle_lookup[(current_node_pt, next_node_pt)]
        
        # path_x, path_y, path_yaw, mode, path_length = dubins_path_planning(
        #     node_1[1][0], node_1[1][1], angle_1,
        #     node_2[0][0], node_2[0][1], angle_2, 2/tool_width_px)
        try:
            path = transition_graph_lookup.path_lookup[(current_node_pt, next_node_pt)]
            path_x = []
            path_y = []
            for pt in path.coords:
                path_x.append(pt[0])
                path_y.append(pt[1])
            transition_paths.append((path_x, path_y))
        except Exception as e:
            print('Missed Transition' +  str(current_node_pt) + ' ' + str(next_node_pt))
            raise(e)

        return cost, length, angle