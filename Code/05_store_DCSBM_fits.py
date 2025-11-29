import igraph as ig
import graph_tool.all as gt
import numpy as np
import networkx as nx
from collections import Counter
import pickle
import os
import time
import multiprocessing
from bidirectional_bfs import bidirectional_bfs_distance_networkx, bidirectional_bfs_distance_igraph

from scipy import stats
import copy
import warnings
import math
import random

def get_simplified_igraph_network(iG):
    if 'weight' in set(iG.es.attributes()):
        del iG.es['weight']
    iG.to_undirected(combine_edges=None)
    iG.simplify(multiple=True, loops=True, combine_edges=None)
    return iG


def save_DCSBM_fits(graphname, domain):
    t1_final = time.time()
    
    iG = ig.Graph.Read_Ncol("../network_repository_edgelists/" + domain + "/"+ graphname, names=True, directed=False)
    simplified_iG = get_simplified_igraph_network(iG)
    
    numnodes = simplified_iG.vcount()
    numedges = simplified_iG.ecount()

    Results = [graphname, numnodes, numedges]
    
    G = simplified_iG.to_networkx()
    G = nx.convert_node_labels_to_integers(G)
    
    # Create graph-tool graph gt_G.
    gt_G = gt.Graph(directed=False)
    vlist = gt_G.add_vertex(G.number_of_nodes())
    for edge in G.edges():
        e = gt_G.add_edge(edge[0], edge[1])
        

    fits = []
    time_list = []
    MDL_min = np.inf

    for i in range(100):
        last_partition_t1 = time.time()
        
        state = gt.minimize_blockmodel_dl(gt_G)
        
        b = gt.contiguous_map(state.get_blocks())
        s2 = state.copy(b=b) # These 2 lines^ make sure that the community numbers are mapped to continuous values.
        fits.append((s2.entropy(), s2.b.a, gt.adjacency(s2.get_bg(), s2.get_ers()).T))
        # s2.entropy() stores the description length of the fit/state s2
        # s2.b.a stores the group membership for each node of the state s2
        # gt.adjacency(s2.get_bg(), s2.get_ers()).T) stores the matrix of edge counts between groups
        # Therefore, for each state we store the MDL, group-membership and edge-counts between groups corresponding to the fit.
        
        if s2.entropy() < MDL_min:
            MDL_min = s2.entropy()

    
    indices_to_remove = []
    for i in range(len(fits)):
        if fits[i][0] - MDL_min >= 1300: # description-length of this fit is too big, so we can safely discard this.
            indices_to_remove.append(i)
    
    truncated_fits = []
    for i in range(len(fits)):
        if i not in indices_to_remove:
            truncated_fits.append(fits[i])
            
    fits_initial_length = len(fits)
    truncated_fits_length = len(truncated_fits)
    del fits
                
    fits_second_length = len(truncated_fits)
    
    MDL_max = -999 # Among the not-too-big description lengths, we first identify the max description length.
    for fit in truncated_fits:
        if fit[0] > MDL_max:
            MDL_max = fit[0]
            
    
    chosen_X = -700 + MDL_max # We choose this X to avoid underflow issue
    
    total_denominator_sum = 0
    for fit in truncated_fits:
        item = math.exp(chosen_X - fit[0])
        total_denominator_sum += item
        
    truncated_fits_entropy_list = []
    for index in range(len(truncated_fits)):
        truncated_fits_entropy_list.append((index, truncated_fits[index][0]))
    
    truncated_fits_entropy_list.sort(key=lambda y: y[1]) # sort list of entropies in ascending order (since smaller the better)
    
    cumulative_post_prob = 0
    top_chosen_indices = []
    
    for pair in truncated_fits_entropy_list:
        index = pair[0]
        post_prob_num = math.exp(chosen_X - truncated_fits[index][0])
        post_prob_denom = total_denominator_sum
        post_prob = post_prob_num/post_prob_denom*100 # posterior probability
        cumulative_post_prob += post_prob
        top_chosen_indices.append(index) # Keep saving the indices of good fits till we cover 99.9% probability.
        if cumulative_post_prob > 99.9:
            break

    i = 0
    counter = 1
    top_chosen_fit_lengths = []
    while i < len(top_chosen_indices):
        top_chosen_fit = []
        while i < len(top_chosen_indices) and len(top_chosen_fit) < 50:
            top_chosen_fit.append(truncated_fits[top_chosen_indices[i]])
            i+=1

        foldername = "../DCSBM_fits/" + domain + "/" + graphname[:-4] # ../DCSBM_fits/{domain} folder should exist beforehand.
        if os.path.isdir(foldername) == False:
            os.mkdir(foldername)
            
        pickleFile = "../DCSBM_fits/" + domain + "/" + graphname[:-4] + "/" + graphname[:-4] + "_" + str(counter) + ".pkl"
        pickle_out = open(pickleFile,"wb") 
        pickle.dump(top_chosen_fit, pickle_out)
        pickle_out.close()
        counter += 1
        top_chosen_fit_lengths.append(len(top_chosen_fit))
                
        if i == len(top_chosen_indices):
            break    
    
    
    t2_final = time.time()
    time_taken = (t2_final - t1_final)/3600
        
    # Store global properties about this network's sampling procedure.
    Results.append(time_taken)
    Results.append(time_list)
    Results.append((MDL_min, MDL_max, chosen_X))
    Results.append((fits_initial_length, truncated_fits_length, len(truncated_fits_entropy_list), len(top_chosen_indices), top_chosen_fit_lengths))
    
    pickleFile = "../DCSBM_fits/" + domain + "/" + graphname[:-4] + "/" + graphname[:-4] + ".pkl"
    pickle_out = open(pickleFile,"wb") 
    pickle.dump(Results, pickle_out)
    pickle_out.close()
    

def run(index, domain):
    AllFiles = os.listdir("../network_repository_edgelists/" + domain + "/")
        
    filename = AllFiles[index]
    
    if filename == ".ipynb_checkpoints" or filename == ".DS_Store":
        exit() 
        
    txtfileName = filename.strip()
    
    t1 = time.time()
    save_DCSBM_fits(txtfileName, domain)
    t2 = time.time()
    
    print("Total time taken = ", (t2-t1)/3600, "hours.")    


import sys

# Command Line Arguments.
index = sys.argv[1] # index of the network to run this file for.
domain = sys.argv[2] # domain of the chosen network
run(int(index), domain)