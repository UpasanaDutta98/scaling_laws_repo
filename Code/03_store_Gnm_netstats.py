import igraph as ig
import numpy as np
import networkx as nx
from collections import Counter
import pickle
import os
import time
import multiprocessing
from bidirectional_bfs import bidirectional_bfs_distance_networkx, bidirectional_bfs_distance_igraph

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

def get_degree_assortativity(G):
    G_degree = list(nx.degree(G))
    m = G.number_of_edges()
    n = G.number_of_nodes()
    S1 = 2*m
    S2 = 0
    S3 = 0
    for i in range(n):
        S2 += (G_degree[i][1])**2
        S3 += (G_degree[i][1])**3
        
    denominator = S1*S3 - (S2**2)
    
    SL = 0
    for e in G.edges():
        SL += 2*G_degree[e[0]][1]*G_degree[e[1]][1]
    numerator = S1*SL - (S2**2)
    r = float(numerator)/denominator
    return r


def get_simplified_igraph_network(iG):
    if 'weight' in set(iG.es.attributes()):
        del iG.es['weight']
    iG.to_undirected(combine_edges=None)
    iG.simplify(multiple=True, loops=True, combine_edges=None)
    return iG


def save_Gnm_netstats(graphname, domain):
    iG = ig.Graph.Read_Ncol("../network_repository_edgelists/" + domain + "/"+ graphname, names=True, directed=False)
    simplified_iG = get_simplified_igraph_network(iG)
    
    numnodes = simplified_iG.vcount()
    numedges = simplified_iG.ecount()
    
    Results = [graphname, numnodes, numedges]
    cc_list = []
    r_list = []
    mgd_list = []
    
    t1 = time.time()
    for i in range(50):
        g_null = ig.Graph.Erdos_Renyi(n=numnodes, m=numedges, directed=False, loops=False) # Generate 50 G_nm graphs.
        
        clustering_coeff = g_null.transitivity_undirected()
        
        degree_assort = g_null.assortativity_degree(directed=False)
    
        distance_list = threshold_sampler_igraph(g_null, threshold=0.1, batch_size=1000)
        avg_path_length = np.mean(distance_list)
        
        cc_list.append(clustering_coeff)
        r_list.append(degree_assort)
        mgd_list.append(avg_path_length)
        
        
    t2 = time.time()
    
    time_taken = (t2-t1)/60
    
    Results.append(cc_list)
    Results.append(r_list)
    Results.append(mgd_list)
    Results.append(time_taken)
    
    
    pickleFile = "../Results/Gnm_netstats/" + domain + "/"+graphname[:-4] + ".pkl"
    pickle_out = open(pickleFile,"wb") 
    pickle.dump(Results, pickle_out)
    pickle_out.close()
    

def parallel(list_of_names, domain):
    manager = multiprocessing.Manager()
    jobs = []
    for i in range(len(list_of_names)):
        graphname = list_of_names[i]
        p = multiprocessing.Process(target=save_Gnm_netstats, args=(graphname, domain))
        jobs.append(p)
        p.start()
    print("len of jobs = ", len(jobs))
    for proc in jobs:
        proc.join()

        
def run(index1, index2, domain):
    AllFiles = os.listdir("../network_repository_edgelists/" + domain + "/")
    
    list_of_names = []
    for eachline in AllFiles[index1:index2+1]:
        if eachline == ".ipynb_checkpoints" or eachline == ".DS_Store":
            continue
        txtfileName = eachline.strip()
        list_of_names.append(txtfileName)
        
    print("len(list_of_names) = ", len(list_of_names))
    t1 = time.time()
    parallel(list_of_names, domain)
    t2 = time.time()
    print("Total time taken = ", (t2-t1)/3600, "hours.")    

import sys
index1 = sys.argv[1]
index2 = sys.argv[2]
domain = sys.argv[3]
run(int(index1), int(index2), domain)