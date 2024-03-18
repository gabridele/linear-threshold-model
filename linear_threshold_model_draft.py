# -*- coding: utf-8 -*-
"""
@author: ludovico coletta
"""

import numpy as np
import pandas as pd
import time

def run_cascade_single_population(adj_matrix, thr, seed_node_index):

    infected_nodes=np.zeros((adj_matrix.shape[0]))
    input_to_node=np.sum(adj_matrix,axis=0)
            
    infected_nodes[seed_node_index]=1
    
    list_of_infected_nodes_per_iter=[]
    list_of_infected_nodes_per_iter.append(np.where(infected_nodes==1)[0].tolist()) # list of lists
    counter=0
    
    while int(np.sum(infected_nodes))<adj_matrix.shape[0]:
        
        if counter>30: # Realistically, we should converge in max 10 steps (see Misic et al Neuron)
            
            #print('I got stuck, threshold of '+str(thr)+ ' was too high')
            break
        
        indices_of_infected_nodes=np.where(infected_nodes==1)[0]
        
        mask_array=np.zeros((adj_matrix.shape))
        mask_array[indices_of_infected_nodes,:]=1
        
        infected_connections=adj_matrix.copy()
        infected_connections=infected_connections*mask_array
        infected_inputs=np.sum(infected_connections,axis=0)
        infected_nodes_indices=np.where(infected_inputs/input_to_node>thr)[0]
        list_of_infected_nodes_per_iter.append(infected_nodes_indices.tolist())
        infected_nodes[infected_nodes_indices]=1
        counter=counter+1
        
    return list_of_infected_nodes_per_iter
        
def find_thr(adj_matrix,starting_thr):
                      
    visited_thresholds_per_node=[None]*adj_matrix.shape[0]
          
    for seed_node_index in range(0,adj_matrix.shape[0]):
        #start_time=time.time()     
        visited_thresholds_per_node[seed_node_index]=[]
        thr=starting_thr
        
        for dummy_thr in range(0,1000): # In doing so we guarantee to converge in less than 1000 steps, but do we?
            
            list_of_infected_nodes_per_iter=run_cascade_single_population(adj_matrix, thr, seed_node_index)
            
            if len(list_of_infected_nodes_per_iter[-1])==adj_matrix.shape[0]:
                thr=thr*2
                visited_thresholds_per_node[seed_node_index].append(thr)
            elif (dummy_thr==0) and (len(list_of_infected_nodes_per_iter[-1])!=adj_matrix.shape[0]): 
                # if the first threshold is already too high, divide the initial step by 10. 
                # If the script crashes, increasing the number
                thr=thr/100.
            else:                
                break
        #print(time.time()-start_time)
        
    max_thresholds_per_node=np.asarray([visited_thresholds_per_node[ii][-1] for ii in range(0,len(visited_thresholds_per_node))])
    
    bottleneck_node=np.where(max_thresholds_per_node==np.min(max_thresholds_per_node))[0]
    
    thrs=np.linspace(visited_thresholds_per_node[bottleneck_node[0]][-2],visited_thresholds_per_node[bottleneck_node[0]][-1],100,endpoint=True)
    
    visited_thresholds_of_bottleneck_node=[]
    visited_thresholds_of_bottleneck_node.append(thrs[0])
    
    for final_thr in thrs:
        
            list_of_infected_nodes_per_iter=run_cascade_single_population(adj_matrix, final_thr,  bottleneck_node[0])
            
            if len(list_of_infected_nodes_per_iter[-1])==adj_matrix.shape[0]:                
                visited_thresholds_of_bottleneck_node.append(final_thr)
            else:                
                break
    return visited_thresholds_of_bottleneck_node[-1]
            
def main():
    
    adj_matrix=pd.read_csv('connectome_sub-100206.csv',header=None).to_numpy().astype(float)
    zero_rows=np.where(np.sum(adj_matrix,0)==0)[0].tolist()
    adj_matrix_clean=np.delete(adj_matrix, zero_rows,axis=0)
    adj_matrix_clean=np.delete(adj_matrix_clean, zero_rows,axis=1)
     
    starting_thr=0.0015
    start_time=time.time()
    thr=find_thr(adj_matrix_clean,starting_thr)
    print(time.time()-start_time)
    seed_node_index=np.random.randint(0,adj_matrix.shape[0])
    list_of_infected_nodes_per_iter=run_cascade_single_population(adj_matrix, thr, seed_node_index)
    
if __name__ == "__main__":
    main()

