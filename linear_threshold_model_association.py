# -*- coding: utf-8 -*-
"""
@author: ludovico coletta
@author: gabriele de leonardis
"""

# import necessary librariesz
import numpy as np
import pandas as pd 
import sys
import time
from multiprocessing import Pool

def run_cascade_single_population(adj_matrix, thr, seed_node_index):
    # initialize array to track infected nodes, all set to 0 (not infected) initially
    infected_nodes = np.zeros((adj_matrix.shape[0]))
    # Compute total input to each node from all other nodes, aka calculate weighted degree for all nodes
    input_to_node = np.sum(adj_matrix, axis=0)
    # infect seed node, set to 1
    infected_nodes[seed_node_index] = 1
    # list to keep track of infected nodes per iteration
    list_of_infected_nodes_per_iter = []
    # add seed node to list of infected nodes for the first iteration
    list_of_infected_nodes_per_iter.append(np.where(infected_nodes == 1)[0].tolist())
    # counter to limit number of iterations
    counter = 0
    
    # iterate until all nodes are infected or reached a maximum of 30 iterations
    while int(np.sum(infected_nodes)) < adj_matrix.shape[0]:
        if counter > 30:
            break
        
        # get indices of currently infected nodes
        indices_of_infected_nodes = np.where(infected_nodes == 1)[0]
        # mask to isolate infected connections
        mask_array = np.zeros((adj_matrix.shape))
        # mark rows and cols corresponding to infected nodes
        mask_array[indices_of_infected_nodes, :] = 1
        mask_array[:, indices_of_infected_nodes] = 1
        # apply mask to adjacency mtrx
        infected_connections = adj_matrix.copy()
        infected_connections = infected_connections * mask_array
        # calculate input from infected nodes to each node, their weighted degree
        infected_inputs = np.sum(infected_connections, axis=0)
        # find nodes to infect based on threshold: ratio of weighted degree of infected nodes and weighted degrees of all matrx nodes should be greater than threshold
        infected_nodes_indices = np.where(infected_inputs / input_to_node > thr)[0]
        # update list
        list_of_infected_nodes_per_iter.append(infected_nodes_indices.tolist())
        # mark new nodes as infected
        infected_nodes[infected_nodes_indices] = 1
        counter += 1
        
    return list_of_infected_nodes_per_iter

          
def find_thr(adj_matrix, starting_thr):
    # list to track visited thr per node
    visited_thresholds_per_node = [None] * adj_matrix.shape[0]
    
    # iterate over each node as a seed node
    for seed_node_index in range(adj_matrix.shape[0]):
        visited_thresholds_per_node[seed_node_index] = []
        # initialize thr with starting value
        thr = starting_thr
        
        # try different thr to find the highest thr that allows to global spreading
        for dummy_thr in range(1000):
            # run single cascade for current thr
            list_of_infected_nodes_per_iter = run_cascade_single_population(adj_matrix, thr, seed_node_index)
            # if all nodes are infected, increase thr and record it
            if len(list_of_infected_nodes_per_iter[-1]) == adj_matrix.shape[0]:
                thr *= 2
                visited_thresholds_per_node[seed_node_index].append(thr)
            # if first attempt fails, reduce thr
            elif dummy_thr == 0 and len(list_of_infected_nodes_per_iter[-1]) != adj_matrix.shape[0]:
                thr /= 100
            # if cascade fails to infect all nodes, stop refining thr
            else:
                break
    
        # debug
        print(f"Node {seed_node_index}: visited thresholds = {visited_thresholds_per_node[seed_node_index]}")
    
    # find max thr for each node
    max_thresholds_per_node = np.asarray([visited_thresholds_per_node[ii][-1] for ii in range(len(visited_thresholds_per_node))])
    # identify bottleneck node (node with smallest max thr)
    bottleneck_node = np.where(max_thresholds_per_node == np.min(max_thresholds_per_node))[0]
    
    # ensure at least two thresholds before accessing them
    if len(visited_thresholds_per_node[bottleneck_node[0]]) < 2:
        raise ValueError(f"Not enough thresholds visited for bottleneck node {bottleneck_node[0]}: {visited_thresholds_per_node[bottleneck_node[0]]}")
    
    # create range of thr to refine the search
    thrs = np.linspace(visited_thresholds_per_node[bottleneck_node[0]][-2], visited_thresholds_per_node[bottleneck_node[0]][-1], 100, endpoint=True)
    visited_thresholds_of_bottleneck_node = []
    visited_thresholds_of_bottleneck_node.append(thrs[0])
    final_thr_per_node = []
    
    # further refine thr for bottleneck node
    for node in bottleneck_node:
        for final_thr in thrs:
            # run cascade process for refined thr
            list_of_infected_nodes_per_iter = run_cascade_single_population(adj_matrix, final_thr, node)
            # if all nodes are infected, record thr
            if len(list_of_infected_nodes_per_iter[-1]) == adj_matrix.shape[0]:
                visited_thresholds_of_bottleneck_node.append(final_thr)
            else:
                break
        final_thr_per_node.append(visited_thresholds_of_bottleneck_node[-1])
    
    return bottleneck_node, np.min(final_thr_per_node)


def run_cascade_multiple_populations(adj_matrix, thr, n_pop, n_sim):
    # list to store infected nodes for each population
    infected_nodes_per_run = []
    counter_sim = 0
    
    # initialize association matrix to store final output
    association_matrix = np.zeros((adj_matrix.shape[0], adj_matrix.shape[0]))
    
    # run simulations for the number of times indicated
    while len(infected_nodes_per_run) < n_sim:
        # randomly choose seed nodes for each population
        seed_node_indices = sorted(np.random.choice(adj_matrix.shape[0], size=n_pop, replace=False).tolist())
        # convert seed nodes to a nested list format
        seed_node_indices = [[ii] for ii in seed_node_indices]
        # initialize infected nodes array
        infected_nodes = np.zeros((adj_matrix.shape[0]))
        # calculate weighted degree of all nodes in the mtrx
        input_to_node = np.sum(adj_matrix, axis=0)
        # infect seed nodes
        infected_nodes[seed_node_indices] = 1
        stuck = 0
        
        # iterate until all nodes are infected 
        while int(np.sum(infected_nodes)) < adj_matrix.shape[0]:
            list_of_potential_infected_nodes_within_iter_per_pop = [[None]] * n_pop
            
            # process each population separately
            for seed_node_index, node_infected in enumerate(seed_node_indices):
                mask_array = np.zeros((adj_matrix.shape))
                # mark rows and cols corresponding to infected nodes
                mask_array[node_infected, :] = 1
                mask_array[:, node_infected] = 1
                # apply mask to adj matrix
                infected_connections = adj_matrix.copy()
                infected_connections = infected_connections * mask_array
                # calculate weighted degrees of infected nodes
                infected_inputs = np.sum(infected_connections, axis=0)
                potential_infected_nodes_indices = infected_inputs / input_to_node
                # store potential inf nodes for each population
                list_of_potential_infected_nodes_within_iter_per_pop[seed_node_index] = potential_infected_nodes_indices
            
            # combine potential inf nodes from all populations
            input_per_node = np.vstack(list_of_potential_infected_nodes_within_iter_per_pop)
            # identify nodes that can be infected based on thr
            nodes_to_check = np.where(input_per_node >= thr)[1].tolist()
            # flatten and sort the list of seed nodes
            dummy_list = [elem for sublist in seed_node_indices for elem in sublist]
            nodes_to_check = sorted(list(set([ii for ii in nodes_to_check if ii not in dummy_list])))
            
            # if no nodes can be infected, increase stuck counter and break
            if len(nodes_to_check) == 0:
                #print('I got stuck')
                stuck = stuck + 1
                break
            else:
                # solve conflicts if multiple populations can infect same node
                for node in nodes_to_check:
                    indices_of_winner = np.where(input_per_node[:, node] == np.max(input_per_node[:, node]))[0]
                    if indices_of_winner.size == 1:
                        seed_node_indices[indices_of_winner[0]].append(node)
                    else:
                        seed_node_indices[np.random.choice(indices_of_winner.size, size=1)[0]].append(node)
                
                # mark the new nodes as infected
                for node in seed_node_indices:
                    infected_nodes[node] = 1
        
        # if no nodes got stuck, add result to list
        if stuck == 0:
            infected_nodes_per_run.append(seed_node_indices)
            
            # update association matrix
            for node_set in seed_node_indices:
                for i in node_set:
                    for j in node_set:
                        if i != j:
                            association_matrix[i, j] += 1
        elif stuck == 1:
            # if no new node activated in current iteration, this removes the last entry from list
            infected_nodes_per_run.pop()
        
        counter_sim += 1
    
    # normalize association mtrix by number of simulation
    association_matrix /= n_sim
    return infected_nodes_per_run, association_matrix

def main(input_file_path, n_pop):
    # extract subject ID from the file path
    sub_id = input_file_path.split('/')[-3]

    # load adj matrix
    adj_matrix = pd.read_csv(input_file_path, delimiter=',', header=None).to_numpy().astype(float)
    
    print(f"now processing: {sub_id} with {n_pop} seeds competitive scenario")
    
    # identify zero connections and < 5 connection nodes (Seguin, Jedynak, et al., 2023)
    zero_rows = np.where(np.sum(adj_matrix, 0) == 0)[0].tolist()
    low_connection_nodes = np.where(np.sum(adj_matrix > 0, axis=0) < 5)[0].tolist()
    
    # combine together
    all_removed_nodes = sorted(set(zero_rows + low_connection_nodes))
    
    # matrix filled with ones initially
    zero_connection_nodes_matrix = np.ones_like(adj_matrix, dtype=int)
    
    # update rows and cols corresponding to zero/low-connection nodes to 0
    zero_connection_nodes_matrix[all_removed_nodes, :] = 0
    zero_connection_nodes_matrix[:, all_removed_nodes] = 0

    # remove those nodes from input matrix
    adj_matrix_clean = np.delete(adj_matrix, all_removed_nodes, axis=0)
    adj_matrix_clean = np.delete(adj_matrix_clean, all_removed_nodes, axis=1)

    #iu2 = np.triu_indices(adj_matrix_clean.shape[0], 1)
    #a = adj_matrix_clean[iu2]
    #density = np.count_nonzero(a) / a.shape[0]
    
    # define starting thr
    starting_thr = 0.0015
    start_time = time.time()
    bottl_nodes, thr = find_thr(adj_matrix_clean, starting_thr)
    print(f"Time to find threshold: {time.time() - start_time} seconds")
    
    n_steps_needed = [None] * adj_matrix_clean.shape[0]
    
    for ii in range(len(n_steps_needed)):
        n_steps_needed[ii] = len(run_cascade_single_population(adj_matrix_clean, thr, ii))
    
    start_time = time.time()
    n_sim = 10000
    
    # run main simulation
    _, association_matrix = run_cascade_multiple_populations(adj_matrix_clean, thr, n_pop, n_sim)
    print(f"Time to run competitive cascades: {time.time() - start_time} seconds")
    
    # save association matrix and indices of nodes removed
    association_matrix_filename = f"derivatives/{sub_id}/dwi/association_matrix_{sub_id}_{n_pop}seeds.csv"
    removed_nodes_filename = f"derivatives/{sub_id}/dwi/removed_nodes_{sub_id}_{n_pop}seeds.csv"

    np.savetxt(association_matrix_filename, association_matrix, delimiter=",")
    np.savetxt(removed_nodes_filename, zero_connection_nodes_matrix, delimiter=",", fmt="%d")
    
if __name__ == "__main__":
 
    # Check if script is executed with correct number of args
    if len(sys.argv) != 2:
        print("Correct syntax: cat input_file_list | python [this_script.py] [n_pop]")
        sys.exit(1)
    
    input_file_paths = [line.strip() for line in sys.stdin]
    
    n_pop = int(sys.argv[1])

    pool = Pool(processes=142)
    pool.starmap(main, [(file_path, n_pop) for file_path in input_file_paths])

########## HOW TO RUN ###########
# from terminal (bash), cd to dataset folder
# you should also have a folder called "code" in position ../code
# adjust number of parallel processes @ line 211
# choose number of populations [n_pop]
# then type these commands:
"""
path_der="derivatives/"
find "$path_der" -type f -name '*5000000mio_connectome.csv' > "$path_der/connectome_files.txt"
cat "$path_der/connectome_files.txt" | python ../code/linear-threshold-model/linear_threshold_model_association.py [n_pop] > log_LTMsimulation.txt
"""
