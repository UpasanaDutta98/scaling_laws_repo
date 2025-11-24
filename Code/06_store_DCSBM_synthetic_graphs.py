import igraph as ig
import random
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
import graph_tool

    
def get_simplified_igraph_network(iG):
    if 'weight' in set(iG.es.attributes()):
        del iG.es['weight']
    iG.to_undirected(combine_edges=None)
    iG.simplify(multiple=True, loops=True, combine_edges=None)
    return iG
        
    
def generate_DCSBM_Microcanonical(graphname, domain, chosen_fits, chosen_indices, degree_list, n, m, degs):
    # if degs == 1, degree-corrected version of the model is used where the degrees of nodes will be given exactly by the degree_list parameter. When degs == 1, micro_ers is automatically True, meaning not only the degrees are specified but the exact number of edges between communities are also specified.

    t1_null = time.time()

    foldername = "../DCSBM_SyntheticNetworks/" + domain + "/" + graphname[:-4] # Note: DCSBM_SyntheticNetworks/{domain} should exist from beforehand. Create it if the folder for each domain does not already exist.
    os.mkdir(foldername)    
    
    i = 0   
    
    # Save the chosen_indices list for downstream runs.
    pickleFile = foldername + "/" + graphname[:-4] + ".pkl"
    pickle_out = open(pickleFile,"wb") 
    pickle.dump(chosen_indices, pickle_out)
    pickle_out.close()
    
    
    for fit_tuple in chosen_fits:
        i+=1
        synthetic_network = gt.generate_sbm(fit_tuple[1], fit_tuple[2], out_degs=degree_list, micro_degs=True, directed=False) # this is degree-corrected version of SBM

        # Creating adjacency list of each synthetic network.
        Adj_list = [[] for j in range(synthetic_network.num_vertices())]
        for e in synthetic_network.iter_edges():
            if e[0] == e[1]: # If this edge is a self-loop
                Adj_list[e[0]].append(e[1])
            else:
                Adj_list[e[0]].append(e[1])
                Adj_list[e[1]].append(e[0])

        file2 = open(foldername + "/" + graphname[:-4] + "_" + str(i) + ".txt", "a")
        
        for row in Adj_list:
            string = ''
            for element in row:
                string = string + str(element)
                string = string + " "
            string = string + "\n"
            file2.write(string)
        file2.close()
    
    
def generate_DCSBM_networks(graphname, domain):    
    iG = ig.Graph.Read_Ncol("../network_repository_edgelists/" + domain + "/"+ graphname, names=True, directed=False)
    simplified_iG = get_simplified_igraph_network(iG)
    n, m = simplified_iG.vcount(), simplified_iG.ecount()
 
    degree_list = []
    count = 0
    for i in range(simplified_iG.vcount()):
        deg = simplified_iG.vs[i].degree()
        degree_list.append(deg)
        
    saved_pkl_files = os.listdir("../DCSBM_fits/" + domain + "/" + graphname[:-4] + "/") # Read all pickle files from folder.
    
    pickle_in = open("../DCSBM_fits/" + domain + "/" + graphname[:-4] + "/" + graphname[:-4] + ".pkl","rb")
    Results = pickle.load(pickle_in) # global properties about the network's sampling procedure.
    chosen_X = Results[5][-1]
    
    num_saved_fit_lists = len(saved_pkl_files) - 1

    # First, compute sum_denom for all the fits stored.
    sum_denom = 0
    counter = 1
    while counter <= num_saved_fit_lists:
        pickle_in = open("../DCSBM_fits/" + domain + "/" + graphname[:-4] + "/" + graphname[:-4] + "_" + str(counter) + ".pkl","rb")
        counter += 1
        fit_list = pickle.load(pickle_in)
        for fit in fit_list:
            entropy = fit[0]
            item = math.exp(chosen_X - entropy)
            sum_denom += item 
        
    counter = 1
    all_fits = []
    all_probabilities = [] # for each fit, we store the probability of sampling the fit relative to other fits
    while counter <= num_saved_fit_lists:
        pickle_in = open("../DCSBM_fits/" + domain + "/" + graphname[:-4] + "/" + graphname[:-4] + "_" + str(counter) + ".pkl","rb")
        counter += 1
        fit_list = pickle.load(pickle_in)
        for fit in fit_list:
            entropy = fit[0]
            all_fits.append(fit)
            numerator = math.exp(chosen_X - entropy)
            prob = numerator/sum_denom
            all_probabilities.append(prob)
    
    # Choose 50 fits proportionately, w/ replacement
    index_list = [i for i in range(len(all_fits))]
    random.seed(0)
    chosen_indices = random.choice(index_list, p = all_probabilities, size = 50)
    
    # Store the fit information in a list.
    chosen_fits = []
    for index in chosen_indices:
        chosen_fits.append(all_fits[index])
        
    generate_DCSBM_Microcanonical(graphname, domain, chosen_fits, chosen_indices, degree_list, n, m, degs = 1)
    

def run(index, domain):
    AllFiles = os.listdir("../network_repository_edgelists/" + domain + "/")
        
    filename = AllFiles[index]
    
    if filename == ".ipynb_checkpoints" or filename == ".DS_Store":
        exit() 
        
    txtfileName = filename.strip()
    
    t1 = time.time()
    generate_DCSBM_networks(txtfileName, domain)
    t2 = time.time()
    
    print("Total time taken = ", (t2-t1)/3600, "hours.")  
    
import sys

# Command Line Arguments.
index = sys.argv[1] # index of the network to run this file for.
domain = sys.argv[2] # domain of the chosen network
run(int(index), domain)