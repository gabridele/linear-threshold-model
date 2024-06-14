# -*- coding: utf-8 -*-
"""
@author: ludovico coletta
@author: gabriele de leonardis
"""

import numpy as np
import pandas as pd
import sys
import time

def run_cascade_single_population(adj_matrix, thr, seed_node_index):
    infected_nodes = np.zeros((adj_matrix.shape[0]))
    input_to_node = np.sum(adj_matrix, axis=0)
    infected_nodes[seed_node_index] = 1
    list_of_infected_nodes_per_iter = []
    list_of_infected_nodes_per_iter.append(np.where(infected_nodes == 1)[0].tolist())
    counter = 0
    
    while int(np.sum(infected_nodes)) < adj_matrix.shape[0]:
        if counter > 30:
            break
        
        indices_of_infected_nodes = np.where(infected_nodes == 1)[0]
        mask_array = np.zeros((adj_matrix.shape))
        mask_array[indices_of_infected_nodes, :] = 1
        mask_array[:, indices_of_infected_nodes] = 1
        infected_connections = adj_matrix.copy()
        infected_connections = infected_connections * mask_array
        infected_inputs = np.sum(infected_connections, axis=0)
        infected_nodes_indices = np.where(infected_inputs / input_to_node > thr)[0]
        list_of_infected_nodes_per_iter.append(infected_nodes_indices.tolist())
        infected_nodes[infected_nodes_indices] = 1
        counter += 1
        
    return list_of_infected_nodes_per_iter

def find_thr(adj_matrix, starting_thr):
    visited_thresholds_per_node = [None] * adj_matrix.shape[0]
    
    for seed_node_index in range(adj_matrix.shape[0]):
        visited_thresholds_per_node[seed_node_index] = []
        thr = starting_thr
        
        for dummy_thr in range(1000):
            list_of_infected_nodes_per_iter = run_cascade_single_population(adj_matrix, thr, seed_node_index)
            if len(list_of_infected_nodes_per_iter[-1]) == adj_matrix.shape[0]:
                thr *= 2
                visited_thresholds_per_node[seed_node_index].append(thr)
            elif dummy_thr == 0 and len(list_of_infected_nodes_per_iter[-1]) != adj_matrix.shape[0]:
                thr /= 100
            else:
                break
        
        # debug
        print(f"Node {seed_node_index}: visited thresholds = {visited_thresholds_per_node[seed_node_index]}")
        
    max_thresholds_per_node = np.asarray([visited_thresholds_per_node[ii][-1] if visited_thresholds_per_node[ii] else np.inf for ii in range(len(visited_thresholds_per_node))])
    bottleneck_node = np.where(max_thresholds_per_node == np.min(max_thresholds_per_node))[0]
    
    # ensure at least two thresholds before accessing them
    if len(visited_thresholds_per_node[bottleneck_node[0]]) < 2:
        raise ValueError(f"Not enough thresholds visited for bottleneck node {bottleneck_node[0]}: {visited_thresholds_per_node[bottleneck_node[0]]}")
    
    thrs = np.linspace(visited_thresholds_per_node[bottleneck_node[0]][-2], visited_thresholds_per_node[bottleneck_node[0]][-1], 100, endpoint=True)
    visited_thresholds_of_bottleneck_node = []
    visited_thresholds_of_bottleneck_node.append(thrs[0])
    final_thr_per_node = []
    
    for node in bottleneck_node:
        for final_thr in thrs:
            list_of_infected_nodes_per_iter = run_cascade_single_population(adj_matrix, final_thr, node)
            if len(list_of_infected_nodes_per_iter[-1]) == adj_matrix.shape[0]:
                visited_thresholds_of_bottleneck_node.append(final_thr)
            else:
                break
        final_thr_per_node.append(visited_thresholds_of_bottleneck_node[-1])
    
    return bottleneck_node, np.min(final_thr_per_node)


def run_cascade_multiple_populations(adj_matrix, thr, n_pop, n_sim):
    infected_nodes_per_run = []
    counter_sim = 0
    
    association_matrix = np.zeros((adj_matrix.shape[0], adj_matrix.shape[0]))
    
    while len(infected_nodes_per_run) < n_sim:
        seed_node_indices = sorted(np.random.choice(adj_matrix.shape[0], size=n_pop, replace=False).tolist())
        print("seeds:", seed_node_indices)
        seed_node_indices = [[ii] for ii in seed_node_indices]
        infected_nodes = np.zeros((adj_matrix.shape[0]))
        input_to_node = np.sum(adj_matrix, axis=0)
        infected_nodes[seed_node_indices] = 1
        stuck = 0
        
        while int(np.sum(infected_nodes)) < adj_matrix.shape[0]:
            list_of_potential_infected_nodes_within_iter_per_pop = [[None]] * n_pop
            
            for seed_node_index, node_infected in enumerate(seed_node_indices):
                mask_array = np.zeros((adj_matrix.shape))
                mask_array[node_infected, :] = 1
                mask_array[:, node_infected] = 1
                infected_connections = adj_matrix.copy()
                infected_connections = infected_connections * mask_array
                infected_inputs = np.sum(infected_connections, axis=0)
                potential_infected_nodes_indices = infected_inputs / input_to_node
                list_of_potential_infected_nodes_within_iter_per_pop[seed_node_index] = potential_infected_nodes_indices
            
            input_per_node = np.vstack(list_of_potential_infected_nodes_within_iter_per_pop)
            nodes_to_check = np.where(input_per_node >= thr)[1].tolist()
            dummy_list = [elem for sublist in seed_node_indices for elem in sublist]
            nodes_to_check = sorted(list(set([ii for ii in nodes_to_check if ii not in dummy_list])))
            
            if len(nodes_to_check) == 0:
                #print('I got stuck')
                stuck = stuck + 1
                break
            else:
                for node in nodes_to_check:
                    indices_of_winner = np.where(input_per_node[:, node] == np.max(input_per_node[:, node]))[0]
                    if indices_of_winner.size == 1:
                        seed_node_indices[indices_of_winner[0]].append(node)
                    else:
                        seed_node_indices[np.random.choice(indices_of_winner.size, size=1)[0]].append(node)
                
                for node in seed_node_indices:
                    infected_nodes[node] = 1
        
        if stuck == 0:
            infected_nodes_per_run.append(seed_node_indices)
            
            for node_set in seed_node_indices:
                for i in node_set:
                    for j in node_set:
                        if i != j:
                            association_matrix[i, j] += 1
        elif stuck == 1: 
            infected_nodes_per_run.pop()
        
        counter_sim += 1
    
    association_matrix /= n_sim
    return infected_nodes_per_run, association_matrix

def main(input_file_path):
    
    # extract subject ID from the file path
    sub_id = input_file_path.split('/')[-3]

    adj_matrix = pd.read_csv(input_file_path, delimiter=',', header=None).to_numpy().astype(float)
    #adj_matrix = pd.read_csv('dummy_matrix_2.csv', header=None).to_numpy().astype(float)
    
    print(f"now processing: {sub_id}")
    
    zero_rows = np.where(np.sum(adj_matrix, 0) == 0)[0].tolist()

    # matrix filled with ones initially
    zero_connection_nodes_matrix = np.ones_like(adj_matrix)

    # Update rows and cols corresponding to zero-connection nodes to 0
    zero_connection_nodes_matrix[zero_rows, :] = 0
    zero_connection_nodes_matrix[:, zero_rows] = 0

    adj_matrix_clean = np.delete(adj_matrix, zero_rows, axis=0)
    adj_matrix_clean = np.delete(adj_matrix_clean, zero_rows, axis=1)

    iu2 = np.triu_indices(adj_matrix_clean.shape[0], 1)
    a = adj_matrix_clean[iu2]
    density = np.count_nonzero(a) / a.shape[0]
    
    starting_thr = 0.0015
    start_time = time.time()
    bottl_nodes, thr = find_thr(adj_matrix_clean, starting_thr)
    
    print(f"Time to find threshold: {time.time() - start_time} seconds")
    #print("threshold found:", thr)
    
    n_steps_needed = [None] * adj_matrix_clean.shape[0]
    
    for ii in range(len(n_steps_needed)):
        n_steps_needed[ii] = len(run_cascade_single_population(adj_matrix_clean, thr, ii))
    
    start_time = time.time()
    _, association_matrix = run_cascade_multiple_populations(adj_matrix_clean, thr, 2, 10000)
    
    #print("infected nodes per run:", _)
    
    print(f"Time to run competitive cascades: {time.time() - start_time} seconds")
    
    association_matrix_filename = f"derivatives/{sub_id}/dwi/association_matrix_{sub_id}.csv"
    zero_connection_nodes_filename = f"derivatives/{sub_id}/dwi/zero_connection_nodes_{sub_id}.csv"

    np.savetxt(association_matrix_filename, association_matrix, delimiter=",")
    np.savetxt(zero_connection_nodes_filename, zero_connection_nodes_matrix, delimiter=",")

    
if __name__ == "__main__":
    # check if number of arguments is correct
    if len(sys.argv) != 2:
        print("Correct syntax: python this_script.py input_file_path")
        sys.exit(1)
    
    # get input file path from the command-line arguments
    input_file_path = sys.argv[1]
    
    main(input_file_path)
