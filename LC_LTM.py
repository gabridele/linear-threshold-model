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
        mask_array[:, indices_of_infected_nodes]=1 # just for symmetric matrices
        
        infected_connections=adj_matrix.copy()
        infected_connections=infected_connections*mask_array
        infected_inputs=np.sum(infected_connections,axis=0) # just for symmetric matrices
        infected_nodes_indices=np.where(infected_inputs/input_to_node>thr)[0]
        list_of_infected_nodes_per_iter.append(infected_nodes_indices.tolist())
        infected_nodes[infected_nodes_indices]=1
        counter=counter+1
        
    return list_of_infected_nodes_per_iter
        
def find_thr(adj_matrix,starting_thr):
    
    # Now we return one single threshold, but every node has its own!
                  
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
                # If the script crashes, increase the number
                thr=thr/100.
            else:                
                break
        #print(time.time()-start_time)
        
    max_thresholds_per_node=np.asarray([visited_thresholds_per_node[ii][-1] for ii in range(0,len(visited_thresholds_per_node))])
    
    bottleneck_node=np.where(max_thresholds_per_node==np.min(max_thresholds_per_node))[0]
    
    thrs=np.linspace(visited_thresholds_per_node[bottleneck_node[0]][-2],visited_thresholds_per_node[bottleneck_node[0]][-1],100,endpoint=True)
    
    visited_thresholds_of_bottleneck_node=[]
    visited_thresholds_of_bottleneck_node.append(thrs[0])
    final_thr_per_node=[]
    
    for node in bottleneck_node:
        
        for final_thr in thrs:
            
                list_of_infected_nodes_per_iter=run_cascade_single_population(adj_matrix, final_thr, node)
                
                if len(list_of_infected_nodes_per_iter[-1])==adj_matrix.shape[0]:                
                    visited_thresholds_of_bottleneck_node.append(final_thr)
                else:                
                    break
                
        final_thr_per_node.append(visited_thresholds_of_bottleneck_node[-1])
            
    return bottleneck_node,np.min(final_thr_per_node)
            
def run_cascade_multiple_populations(adj_matrix, thr, n_pop, n_sim):
    
    #infected_nodes_per_run=[[None]]*n_sim
    infected_nodes_per_run=[]
    counter_sim=0
    
    #while counter_sim<n_sim:
    while len(infected_nodes_per_run)<n_sim:
        
        seed_node_indices=sorted(np.random.choice(adj_matrix.shape[0], size=n_pop, replace=False).tolist())
        #print(seed_node_indices)
        # index 0 = population 1, index 1 = population 2 and so on
        
        # we need list of list to keep track
        seed_node_indices=[[ii] for ii in seed_node_indices]
        
        infected_nodes=np.zeros((adj_matrix.shape[0]))
        input_to_node=np.sum(adj_matrix,axis=0) # just for symmetric matrices       
        infected_nodes[seed_node_indices]=1        
        stucked=0
        
        while (int(np.sum(infected_nodes))<adj_matrix.shape[0]):
                                    
            list_of_potential_infected_nodes_within_iter_per_pop=[[None]]*n_pop
            
            for seed_node_index,node_infected in enumerate(seed_node_indices):
                
                mask_array=np.zeros((adj_matrix.shape))
                mask_array[node_infected,:]=1
                mask_array[:,node_infected]=1 # just for symmetric matrices
                
                infected_connections=adj_matrix.copy()
                infected_connections=infected_connections*mask_array
                infected_inputs=np.sum(infected_connections,axis=0) # just for symmetric matrices
                potential_infected_nodes_indices=infected_inputs/input_to_node
                list_of_potential_infected_nodes_within_iter_per_pop[seed_node_index]=potential_infected_nodes_indices
                #inner_counter=inner_counter+1
            
            input_per_node=np.vstack(list_of_potential_infected_nodes_within_iter_per_pop) #n_popXNregions matrix
            nodes_to_check=np.where(input_per_node>=thr)[1].tolist()
                
            #recreate a simple list
            dummy_list=[elem for sublist in seed_node_indices for elem in sublist]
            #dummy_list=[elem for sublist in dummy_list for elem in sublist]
            nodes_to_check=sorted(list(set([ii for ii in nodes_to_check if ii not in dummy_list]))) # we do not want to check already infected nodes
                
            if len(nodes_to_check)==0:
                    
                ### Attention ###
                
                # Sometimes we got stuck in a situation where infected nodes could infect only nodes that were already infected
                
                print('I got stuck')
                stucked=stucked+1
                break
            
            else:
                    
                for node in nodes_to_check:
                    indices_of_winner=np.where(input_per_node[:,node]==np.max(input_per_node[:,node]))[0]
                    
                    if indices_of_winner.size==1:
                        seed_node_indices[indices_of_winner[0]].append(node)
                    else:
                        seed_node_indices[np.random.choice(indices_of_winner.size, size=1)[0]].append(node)
                
                for node in seed_node_indices:
                    infected_nodes[node]=1
                    
        infected_nodes_per_run.append(seed_node_indices)
        counter_sim=counter_sim+1
                
        if stucked==1: 
                infected_nodes_per_run.pop()
                   
    return infected_nodes_per_run
                                       
def main():
    
    #adj_matrix=pd.read_csv('sub-10227_Schaefer2018_400Parcels_Tian_Subcortex_S4_1mm_5000000mio_connectome.csv',delimiter=',',header=None).to_numpy().astype(float)
    adj_matrix=pd.read_csv('dummy_matrix_2.csv',header=None).to_numpy().astype(float)
    zero_rows=np.where(np.sum(adj_matrix,0)==0)[0].tolist()
    adj_matrix_clean=np.delete(adj_matrix, zero_rows,axis=0)
    adj_matrix_clean=np.delete(adj_matrix_clean, zero_rows,axis=1)
     
    ## density
    
    iu2 = np.triu_indices(adj_matrix_clean.shape[0], 1)
    a=adj_matrix_clean[iu2]
    density=np.count_nonzero(a)/a.shape[0]
    
    ##
    
    starting_thr=0.0015
    start_time=time.time()
    bottl_nodes,thr=find_thr(adj_matrix_clean,starting_thr)
    print(time.time()-start_time)
    
    """# single node scenario
    n_steps_needed=[None]*adj_matrix_clean.shape[0]
    
    for ii in range(0,len(n_steps_needed)):
        n_steps_needed[ii]=len(run_cascade_single_population(adj_matrix_clean, thr, ii))"""
    
    # competitive scenario
    start_time=time.time()
    competitive_n_2=run_cascade_multiple_populations(adj_matrix_clean, thr, 2, 10000)
    print(competitive_n_2[2])
    print("time elapsed:", time.time()-start_time)
    
    for ii in range(0,len(competitive_n_2)):
        if (len(competitive_n_2[ii][0])+len(competitive_n_2[ii][1])) < adj_matrix_clean.shape[0]:
            print(ii)
            
if __name__ == "__main__":
    main()

