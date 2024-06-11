import numpy as np
import pandas as pd
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
        
    max_thresholds_per_node = np.asarray([visited_thresholds_per_node[ii][-1] for ii in range(len(visited_thresholds_per_node))])
    bottleneck_node = np.where(max_thresholds_per_node == np.min(max_thresholds_per_node))[0]
    thrs = np.linspace(visited_thresholds_per_node[bottleneck_node[0]][-2], visited_thresholds_per_node[bottleneck_node[0]][-1], 100, endpoint=True)
    visited_thresholds_of_bottleneck_node = [thrs[0]]
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
        
        counter_sim += 1
    
    association_matrix /= n_sim
    return infected_nodes_per_run, association_matrix

def main():
    adj_matrix = pd.read_csv('/Users/gabrieledele/Desktop/symmetric_matrix.csv', header=None).to_numpy().astype(float)
    zero_rows = np.where(np.sum(adj_matrix, 0) == 0)[0].tolist()
    adj_matrix_clean = np.delete(adj_matrix, zero_rows, axis=0)
    adj_matrix_clean = np.delete(adj_matrix_clean, zero_rows, axis=1)
    
    iu2 = np.triu_indices(adj_matrix_clean.shape[0], 1)
    a = adj_matrix_clean[iu2]
    density = np.count_nonzero(a) / a.shape[0]
    
    starting_thr = 0.0015
    start_time = time.time()
    bottl_nodes, thr = find_thr(adj_matrix_clean, starting_thr)
    print(f"Time to find threshold: {time.time() - start_time} seconds")
    
    n_steps_needed = [None] * adj_matrix_clean.shape[0]
    
    for ii in range(len(n_steps_needed)):
        n_steps_needed[ii] = len(run_cascade_single_population(adj_matrix_clean, thr, ii))
    
    start_time = time.time()
    n_sim_run, association_matrix = run_cascade_multiple_populations(adj_matrix_clean, thr, 2, 10000)
    print(f"Time to run competitive cascades: {time.time() - start_time} seconds")
    
    np.savetxt("association_matrix.csv", association_matrix, delimiter=",")
    
if __name__ == "__main__":
    main()
