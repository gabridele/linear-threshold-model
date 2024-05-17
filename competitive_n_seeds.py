"""
@author: ludovico coletta
@author: gabriele de leonardis
"""
import numpy as np
import pandas as pd
import argparse

def run_cascade_single_population(adj_matrix, thr, seed_node_index, rival=None):
    
    infected_nodes = np.zeros((adj_matrix.shape[0]))
    input_to_node = np.sum(adj_matrix, axis=0)
    activation_strengths = np.zeros((adj_matrix.shape[0]))  # Store activation strengths
    
    infected_nodes[seed_node_index] = 1
    activation_strengths[seed_node_index] = 1  # Initialize activation strength for seed node
    
    list_of_infected_nodes_per_iter = []
    list_of_activation_strengths_per_iter = []  # Store activation strengths per iteration
    
    list_of_infected_nodes_per_iter.append(np.where(infected_nodes == 1)[0].tolist())
    list_of_activation_strengths_per_iter.append(np.where(activation_strengths == 1)[0].tolist())  # Store initial activation strengths
    
    counter = 0
    
    while int(np.sum(infected_nodes)) < adj_matrix.shape[0]:
        
        indices_of_infected_nodes = np.where(infected_nodes == 1)[0]

        mask_array = np.zeros((adj_matrix.shape))
        mask_array[indices_of_infected_nodes, :] = 1
        
        infected_connections = adj_matrix.copy()
        infected_connections = infected_connections * mask_array

        infected_inputs = np.sum(infected_connections, axis=0)

        if rival is not None:
            for rival_index in rival:
                infected_inputs[rival_index] = 0
        
        infected_nodes_indices = np.where(infected_inputs / input_to_node > thr)[0]
        
        # Update activation strengths for newly infected nodes
        activation_strengths[infected_nodes_indices] += 1
        
        list_of_infected_nodes_per_iter.append(infected_nodes_indices.tolist())
        list_of_activation_strengths_per_iter.append(activation_strengths.tolist())  # Store activation strengths
        
        infected_nodes[infected_nodes_indices] = 1

        if rival is not None and np.sum(infected_nodes) >= adj_matrix.shape[0] - 1:
            break  # Exit the loop if all nodes except rival are infected
        elif rival is None and np.sum(infected_nodes) == adj_matrix.shape[0]:
            break  # Exit the loop if all nodes are infected (no rival specified)
        
        counter += 1
        if counter > 30:
            break

        
    return list_of_infected_nodes_per_iter, list_of_activation_strengths_per_iter  # Return activation strengths

def find_thr(adj_matrix, starting_thr):

    visited_thresholds_per_node = [[] for _ in range(adj_matrix.shape[0])]  # Initialize empty lists

    for seed_node_index in range(0, adj_matrix.shape[0]):
    
        visited_thresholds_per_node[seed_node_index] = []
        thr = starting_thr

        for dummy_thr in range(0,1000):
            #print('current theta:', thr)
            list_of_infected_nodes_per_iter, _ = run_cascade_single_population(adj_matrix, thr, seed_node_index)
            #print('list of inf nodes per iter:', list_of_infected_nodes_per_iter)
            final_infected_nodes = list_of_infected_nodes_per_iter[-1]
            #print('final inf nodes:', final_infected_nodes)
            if len(final_infected_nodes) == adj_matrix.shape[0]:  # Check if all nodes are infected
                thr *= 2  # Increase the threshold
                visited_thresholds_per_node[seed_node_index].append(thr)
            elif (dummy_thr==0) and (len(final_infected_nodes) != adj_matrix.shape[0]):
                thr /= 100  # Decrease the threshold for the next iteration
                #visited_thresholds_per_node[seed_node_index].append(thr)
            else:
                break  # Exit the loop if no more improvement is observed
        
    max_thresholds_per_node=np.asarray([visited_thresholds_per_node[ii][-1] for ii in range(0,len(visited_thresholds_per_node))])
    
    bottleneck_node = np.where(max_thresholds_per_node==np.min(max_thresholds_per_node))[0]

    thrs=np.linspace(visited_thresholds_per_node[bottleneck_node[0]][-2],visited_thresholds_per_node[bottleneck_node[0]][-1],100,endpoint=True)

    visited_thresholds_of_bottleneck_node = []
    visited_thresholds_of_bottleneck_node.append(thrs[0])

    for final_thr in thrs:
        
            list_of_infected_nodes_per_iter, _ = run_cascade_single_population(adj_matrix, final_thr,  bottleneck_node[0])

            if len(list_of_infected_nodes_per_iter[-1]) == adj_matrix.shape[0]:                
                visited_thresholds_of_bottleneck_node.append(final_thr)
                #print('final thr:', visited_thresholds_of_bottleneck_node[-1])
            else:    
                print('Broke the cycle')            
                break
                    
    return visited_thresholds_of_bottleneck_node[-1]
    

def main(num_seeds):
    adj_matrix = pd.read_csv('/Users/gabrieledele/Desktop/dummy_matrix.csv', header=None).to_numpy().astype(float)
    zero_rows = np.where(np.sum(adj_matrix, 0) == 0)[0].tolist()
    adj_matrix_clean = np.delete(adj_matrix, zero_rows, axis=0)
    adj_matrix_clean = np.delete(adj_matrix_clean, zero_rows, axis=1)

    starting_thr = 0.0015
    thr = find_thr(adj_matrix_clean, starting_thr)
    print('thr:', thr)

    seed_node_indices = np.random.choice(range(adj_matrix_clean.shape[0]), size=num_seeds, replace=False)

    inf_nodes_list = []
    activation_strengths_list = []

    for seed_node_index in seed_node_indices:
        print('seed_node_index:', seed_node_index)
        inf_nodes, activation_strengths = run_cascade_single_population(adj_matrix_clean, thr, seed_node_index, rival=seed_node_indices)
        inf_nodes_list.append(inf_nodes)
        activation_strengths_list.append(activation_strengths)

        print('inf_nodes:', inf_nodes)
        print('activation_strengths:', activation_strengths)

    total_strengths = [np.sum(activation_strengths[-1]) for activation_strengths in activation_strengths_list]

    for seed_index, total_strength in zip(seed_node_indices, total_strengths):
        print(f'Strength of seed {seed_index}: {total_strength}')

    max_strength_index = np.argmax(total_strengths)
    max_strength_seed = seed_node_indices[max_strength_index]
    max_strength = total_strengths[max_strength_index]

    print(f"Seed node {max_strength_seed} wins the competition with total strength {max_strength}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cascade simulation with variable number of seeds.')
    parser.add_argument('--num_seeds', type=int, help='Number of seeds to use in the simulation', required=True)
    args = parser.parse_args()
    main(args.num_seeds)