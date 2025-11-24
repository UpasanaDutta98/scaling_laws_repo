import igraph as ig
import graph_tool.all as gt
import numpy as np
from numpy import random
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

    
def threshold_sampler_igraph(g, threshold=0.1, batch_size=1000):
    """
    :param G: igraph.Graph
    :param threshold: Threshold value
    :param batch_size: Number of samples to take before re-evaluating
    :return: List of samples
    """
    s1 = []
    s2 = []
    half_batch = int(batch_size/2)
    threshold_met = False
    while not threshold_met:
        s1.extend(sampler_no_rejection_igraph(g, half_batch))
        s2.extend(sampler_no_rejection_igraph(g, half_batch))
        if np.abs(np.mean(s1) - np.mean(s2)) < threshold:
            threshold_met = True
    return s1 + s2


def sampler_no_rejection_igraph(g, num_samples):
    """
    igraph version of "no-rejection" sampler for pairwise distances.
    Chooses connected component i with probability proportional to n_i^2.
    Samples pairwise distance within selected component.
    :param g: igraph.Graph
    :param num_samples: Number of samples
    :return: List of pairwise distances
    """
    # Return early if graph too small
    if g.vcount() == 0 or g.vcount() == 1:
        return

    tracker = []
    components, probabilities = _component_probability_generator_igraph(g)
    num_components = len(components)
    for x in range(num_samples):
        subgraph_index = np.random.choice(num_components, p=probabilities)
        i, j = np.random.choice(components[subgraph_index], 2, replace=True)
        tracker.append(bidirectional_bfs_distance_igraph(g, i, j))

    return tracker


def _component_probability_generator_igraph(g):
    """
    Provides connected component list and probabilities for each component
    Probability of component i is proportional to n_i^2.
    :param g: igraph.Graph
    :return: list of components, list of probabilities
    """
    # Get list of connected component subgraphs
    components = g.components()
    num_components = len(components)
    component_sizes = [s for s in components.sizes()]

    # Make probabilities proportional to n_i ** 2
    tmp = [s ** 2 for s in component_sizes]
    probabilities = [n / sum(tmp) for n in tmp]
    return components, probabilities


def get_simplified_igraph_network(iG):
    if 'weight' in set(iG.es.attributes()):
        del iG.es['weight']
    iG.to_undirected(combine_edges=None)
    iG.simplify(multiple=True, loops=True, combine_edges=None)
    return iG
            
            
def save_DCSBM_Simplified_netstats(graphname, domain):
    t1_final = time.time()
    
    iG = ig.Graph.Read_Ncol("../network_repository_edgelists/" + domain + "/"+ graphname, names=True, directed=False)
    simplified_iG = get_simplified_igraph_network(iG)
    
    numnodes = simplified_iG.vcount() 
    numedges = simplified_iG.ecount()
    
    Results = [graphname, numnodes, numedges]
    cc_list = []
    r_list = []
    mgd_list = []
        
    All_AdjList_Files = os.listdir("../DCSBM_Simplify/" + domain + "/" + graphname[:-4] + "/")
    length = len(All_AdjList_Files)
    
    for index, filename in enumerate(All_AdjList_Files): # for each network, we compute the network statistics.
        if ((filename[-6] == "_" and filename[-9] == "_") or (filename[-7] == "_" and filename[-10] == "_")) and (filename[-4:] == ".txt"):
            adjList = open("../DCSBM_Simplify/" + domain + "/" + graphname[:-4] + "/" + filename,'r')
            
            all_lines = adjList.readlines()
            num_nodes = len(all_lines)
            edgeList = []
            i = 0
            for row in all_lines:
                row = row.strip()
                row_elements = list(set(row.split(" "))) # multi-edges removed
                for element in row_elements:
                    if element != '' and int(element) > i: # self-loops removed
                        edgeList.append((i, int(element)))
                i+=1                  
            
            # Build an igraph
            iG_null = ig.Graph(directed=False)
            iG_null.add_vertices(num_nodes)
            iG_null.add_edges(edgeList)
            
            clustering_coeff = iG_null.transitivity_undirected()

            degree_assort = iG_null.assortativity_degree(directed=False)

            distance_list = threshold_sampler_igraph(iG_null, threshold=0.1, batch_size=1000)
            avg_path_length = np.mean(distance_list)

            cc_list.append(clustering_coeff)
            r_list.append(degree_assort)
            mgd_list.append(avg_path_length)

    t2_final = time.time()
    time_taken = (t2_final-t1_final)/60
    
    Results.append(cc_list)
    Results.append(r_list)
    Results.append(mgd_list)
    Results.append(length-1) 
    Results.append(time_taken)
    
    pickleFile = "../Results/DCSBM_netstats/" + domain + "/" + graphname[:-4] + ".pkl"
    pickle_out = open(pickleFile,"wb") 
    pickle.dump(Results, pickle_out)
    pickle_out.close()
    
    
def save_DCSBM_netstats(graphname, domain):
    save_DCSBM_Simplified_netstats(graphname, domain)   

def run(index, domain):
    AllFiles = os.listdir("../network_repository_edgelists/" + domain + "/")
        
    filename = AllFiles[index]
    
    if filename == ".ipynb_checkpoints" or filename == ".DS_Store":
        exit() 
        
    txtfileName = filename.strip()
    
    t1 = time.time()
    save_DCSBM_netstats(txtfileName, domain)
    t2 = time.time()
    
    print("Total time taken = ", (t2-t1)/3600, "hours.")  

    
import sys

# Command Line Arguments.
index = sys.argv[1] # index of the network to run this file for.
domain = sys.argv[2] # domain of the chosen network
run(int(index), domain)