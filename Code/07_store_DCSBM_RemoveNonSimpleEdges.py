import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import time
import random
import igraph as ig
import pickle
import os

from collections import defaultdict

def get_simplified_igraph_network(iG):
    if 'weight' in set(iG.es.attributes()):
        del iG.es['weight']
    iG.to_undirected(combine_edges=None)
    iG.simplify(multiple=True, loops=True, combine_edges=None)
    return iG


def check_for_second_edge_case1(a, b, comm1, comm2, shared_edge_matrix, Neighbor_dicts):
    # Here c-d should be chosen from comm1-comm2, since a-b is from comm1-comm2, however comm1 is not necessarily <= comm2 here, so need to check accordingly
        
    comm1_lessThanEq_comm2 = True if comm1 <= comm2 else False

    if comm1_lessThanEq_comm2 == True:
        shuffled_choices = shared_edge_matrix[comm1][comm2] # the edges had been randomly shuffled before.
    else:
        shuffled_choices = shared_edge_matrix[comm2][comm1] # the edges had been randomly shuffled before.
    
    found_second_edge_flag = 0
    
    for p2, second_edge in enumerate(shuffled_choices):  
        if second_edge[2] == 1: # If the deleted_flag for this node is 1, skip this edge because it does not exist.
            continue  
        if comm1_lessThanEq_comm2 == True:
            c = second_edge[0] # note that c belongs to comm1 like node a does
            d = second_edge[1] # note that d belongs to comm2, which may or may not be the same as node b's community
        else:
            d = second_edge[0] # since edges in shared_edge_matrix[comm-1][comm-2] are always stored in a way that first end-point's community is comm-1, and second's comm-2. Since we want c-d in this function to particularly from comm1-comm2, we have to search d-c in shared_edge_matrix[comm2][comm1] essentially.
            c = second_edge[1]
            
        # Note that we know here that node a and node c belong to the same community, comm1, and node b and node d belong to the same community, comm2.
        # Hence node a can only be newly connected to d and not c, so that the community structure remains preserved, i.e. out of the 2 kinds of edge-swaps possible as per Fosdick et al, we can do only one of them (the cross one, not horizontal)
        if len(set([a, b, c, d])) == 1 or len(set([a, b, c, d])) == 2: # We have NOT found the second edge c-d yet. 
            found_second_edge_flag = 0
        elif (d in Neighbor_dicts[a]) or (c in Neighbor_dicts[b]) or (a == d) or (c == b): # this swap will introduce a multi-edge since a-d or c-b or both are already connected from before.
            found_second_edge_flag = 0
        else: # at least 3 distinct nodes exists between a, b, c, d AND the swap will not introduce non-simple edge.
            found_second_edge_flag = 1
            return found_second_edge_flag, c, d, p2
    
    return found_second_edge_flag, None, None, None


def check_for_second_edge_case2(a, b, comm1, comm2, shared_edge_matrix, Neighbor_dicts):
    # Here c-d should be from comm1-comm2, since a-b is from comm1-comm2, however comm1 is not necessarily <= comm2 here.
    
    found_second_edge_flag = 0
            
    # We iterate over each neighboring community of a's community, and check if a valid second-edge can be found.
    for adj_community in range(len(shared_edge_matrix[comm1])): # shared_edge_matrix is a cxc matrix where c is number of communities in the network.
        if adj_community == comm2:
            continue # we skip since we have already checked b's community in case 1
        
        # for all communities other than b's, we check if we can find a valid second-edge
                
        if len(shared_edge_matrix[comm1][adj_community]) != 0: # We fix comm1 and iterate over all neighboring communities of comm1 to find c-d such that c belongs to a'd community (comm1) but d doesn't have to belong to b's community
            found_second_edge_flag, c, d, p2 = check_for_second_edge_case1(a, b, comm1, adj_community, shared_edge_matrix, Neighbor_dicts) # comm1 is fixed throughout the loop
            if found_second_edge_flag == 1:
                return found_second_edge_flag, c, d, p2            
            
    # If execution has reached here, it means found_second_edge_flag is still 0 and edge c-d was not found.
    # In the case, we will redo the search but this time instead of iterating over neighboring communities of node a's community, we will iterate over neighboring communities of node b's community
    for adj_community in range(len(shared_edge_matrix[comm2])): # We fix comm2 and iterate over all neighboring communities of comm2 to find c-d such that d definitely belongs to b's community (comm2) but c doesn't have to belong to a's community
        if adj_community == comm1:
            continue # we skip since we have already checked b's community in case 1
            
        # for all communities other than a's, we check if we can find a valid second-edge
        if len(shared_edge_matrix[adj_community][comm2]) != 0:
            found_second_edge_flag, c, d, p2 = check_for_second_edge_case1(a, b, adj_community, comm2, shared_edge_matrix, Neighbor_dicts) # comm2 is fixed throughout the loop
            if found_second_edge_flag == 1:
                return found_second_edge_flag, c, d, p2
    
    # Note that when p2 is returned above, it corresponds to the index of edge (c-d) or (d-c) in shared_edge_matrix[c1][c2] or shared_edge_matrix[c2][c1] respectively, depending on whether community_labels[c] <= community_labels[d] or not. So this needs to be checked inside the update function.
    return found_second_edge_flag, None, None, None
       
def update_data_structures(a, b, c, d, p1, p2, case3_flag, community_labels, shared_edge_matrix, Neighbor_dicts):
    # Update data structures shared_edge_matrix and Neighbor_dicts because the swap added and deleted edges.
    a_comm = community_labels[a]
    b_comm = community_labels[b]
    
    if case3_flag == 0:
        c_comm = community_labels[c]
        d_comm = community_labels[d]

        # print("a_comm, c_comm, b_comm, d_comm =", a_comm, c_comm, b_comm, d_comm)
    
        assert a_comm == c_comm or b_comm == d_comm # At least one of the end points must be from the same community

        if a_comm == c_comm:
            a_c_comm_same = True
        else:
            a_c_comm_same = False
        if b_comm == d_comm:
            b_d_comm_same = True
        else:
            b_d_comm_same = False
            
    # First update "shared_edge_matrix" to add new edges a-d and b-c.
    if case3_flag == 0: # Check case3_flag since we won't add edges a-d/d-a and b-c/c-b when case-3 is true.
        # This means the second edge was successfully found and no edge would be simply deleted.
        # Deleting existing edge and adding new edge is equivalent to overwriting the new edge at the index of the existing edge
        # Between a_comm and b_comm, edge a_b is removed and edge c_b is added, so c_b replaces a_b between these 2 communities
        # Similarly, between c_comm (which is same as a_comm) and d_comm, edge c_d is removed and edge a_d is added, so a_d replaces c_d
#         if p1 >= len(shared_edge_matrix[a_comm][b_comm]):
#             print("len(shared_edge_matrix[a_comm][b_comm])=", len(shared_edge_matrix[a_comm][b_comm]), "a_comm, b_comm =", a_comm, b_comm, "community_labels[a], community_labels[b] =", community_labels[a], community_labels[b])
        
        # Now perform the O(1) update which instantaneously does deletion+addition of edges by simply overwriting on the existing edge
        
        # We know a_comm is always <= b_comm because that is how non-simple edges were stored.
        if a_c_comm_same:
            shared_edge_matrix[a_comm][b_comm][p1] = [c, b, 0] 
            if c_comm <= d_comm: # then p2 corresponds to the index of edge (c-d) in shared_edge_matrix[c_comm][d_comm]
                shared_edge_matrix[c_comm][d_comm][p2] = [a, d, 0]
            else:
                shared_edge_matrix[d_comm][c_comm][p2] = [d, a, 0]
        
        elif b_d_comm_same:
            shared_edge_matrix[a_comm][b_comm][p1] = [a, d, 0] 
            if c_comm <= d_comm: # then p2 corresponds to the index of edge (c-d) in shared_edge_matrix[c_comm][d_comm]
                shared_edge_matrix[c_comm][d_comm][p2] = [c, b, 0]
            else:
                shared_edge_matrix[d_comm][c_comm][p2] = [b, c, 0]
    
    else: # Case-3 is true, so simply remove a-b edge by setting deleted_flag inside shared_edge_matrix to 1, so that this edge is never chosen later as edge c-d to swap a non-simple edge with.
        shared_edge_matrix[a_comm][b_comm][p1][2] = 1 # change the deleted_flag for the edge (a, b)

    # Now update "Neighbor_dicts". Adding and Removing in Neighbor_dicts are always O(1) time, so no issues.
    
    # Adding neighbor sets (We know a!=d and b!=c because we add only new simple edges to the network) for cases 1 and 2. For case 3, no new edges, hence this should not be executed.
    if case3_flag == 0:
        Neighbor_dicts[a][d] += 1
        Neighbor_dicts[d][a] += 1
        Neighbor_dicts[b][c] += 1
        Neighbor_dicts[c][b] += 1

    # Deleting neighbor sets for edge a-b, should be executed irrespective of cases 1/2/3.
    if a == b:
        Neighbor_dicts[a][b] -= 1
        if Neighbor_dicts[a][b] == 0: # Recall that Neighbor_dicts[x] is a defaultdict.
            del(Neighbor_dicts[a][b])
    else:
        Neighbor_dicts[a][b] -= 1
        if Neighbor_dicts[a][b] == 0:
            del(Neighbor_dicts[a][b])
        Neighbor_dicts[b][a] -= 1
        if Neighbor_dicts[b][a] == 0:
            del(Neighbor_dicts[b][a])

    # Deleting neighbor sets for edge c-d, should only be executed for cases 1 and 2.
    if case3_flag == 0:
        if c == d: 
            Neighbor_dicts[c][d] -= 1
            if Neighbor_dicts[c][d] == 0:
                del(Neighbor_dicts[c][d])
        else:
            Neighbor_dicts[c][d] -= 1
            if Neighbor_dicts[c][d] == 0:
                del(Neighbor_dicts[c][d])
            Neighbor_dicts[d][c] -= 1
            if Neighbor_dicts[d][c] == 0:
                del(Neighbor_dicts[d][c])

def check_if_ers_and_deg_preserved(number_of_communities, community_labels, e_rs, Adj_list):
    final_ers_matrix = [[0 for i in range(number_of_communities)] for j in range(number_of_communities)]
    for node in range(len(Adj_list)):
        for neighbor in Adj_list[node]:
            final_ers_matrix[community_labels[node]][community_labels[neighbor]] += 1
    
    # print("Final e_rs of synthetic network", final_ers_matrix)
    
    for i in range(len(final_ers_matrix)):
        for j in range(len(final_ers_matrix[i])):
            if final_ers_matrix[i][j] != e_rs[i, j]:
                # print(i, j, "not preserved")
                return 0
            
    # print("Final e_rs of synthetic network fully preserved.")
    return 1

def sanity_check_selfLoops_multiEdges(Adj_list):
    selfLoop_found_flag = 0
    multiEdge_found_flag = 0
    
    neighborSet = [set() for node in range(len(Adj_list))]
    
    for node in range(len(Adj_list)):
        for neighbor in Adj_list[node]:
            if node == neighbor:
                selfLoop_found_flag = 1
            if neighbor in neighborSet[node]:
                multiEdge_found_flag = 1
            
            neighborSet[node].add(neighbor)
           
    return selfLoop_found_flag, multiEdge_found_flag


def simplify_DCSBM_networks(graphname, domain):
    iG = ig.Graph.Read_Ncol("../network_repository_edgelists/" + domain + "/"+ graphname, names=True, directed=False)
    simplified_iG = get_simplified_igraph_network(iG)
    numnodes, m = simplified_iG.vcount(), simplified_iG.ecount()
    
    
    foldername = "../DCSBM_Simplify/" + domain + "/" + graphname[:-4] # Note: DCSBM_Simplify/{domain} should exist from before.
    os.mkdir(foldername)
    
    saved_pkl_files = os.listdir("../DCSBM_fits/" + domain + "/" + graphname[:-4] + "/")
    num_saved_fit_lists = len(saved_pkl_files) - 1
    
    counter = 1
    all_fits = []
    while counter <= num_saved_fit_lists:
        pickle_in = open("../DCSBM_fits/" + domain + "/" + graphname[:-4] + "/" + graphname[:-4] + "_" + str(counter) + ".pkl","rb")
        fit_list = pickle.load(pickle_in)
        for fit in fit_list:
            all_fits.append(fit)
        counter+=1

    foldername = "../DCSBM_SyntheticNetworks/" + domain + "/" + graphname[:-4]

    pickle_in = open(foldername + "/" + graphname[:-4] + ".pkl","rb")
    chosen_fit_indices = pickle.load(pickle_in)
    Results = [graphname, numnodes, m]
    tmain1 = time.time()
    choices = []
        
    for i in range(0, len(chosen_fit_indices)):
        tstart = time.time()
        Each_network_result = [graphname, numnodes, m]
            
        chosen_fit_index = chosen_fit_indices[i]

        chosen_fit = all_fits[chosen_fit_index]

        community_labels = chosen_fit[1] # membership of each node to a community.
        number_of_communities = len(set(community_labels))
        community_nodecount = [0 for comm in range(number_of_communities)]
        for node in range(numnodes):
            community_nodecount[community_labels[node]] += 1
        
        e_rs = chosen_fit[2].todense() # e_rs matrix

        # Read the generated null graph adj list.
        filename = foldername + "/" + graphname[:-4] + "_" + str(i+1) + ".txt"
        f = open(filename, "r")
        contents = f.readlines()

        non_simple_edges = []
        shared_edge_matrix = [[[] for comm1 in range(number_of_communities)] for comm2 in range(number_of_communities)]
        Neighbor_dicts = [defaultdict(int) for node in range(len(contents))]
        edgeList = []
                
        
        for node in range(numnodes):
            neighbor_list = contents[node].strip().split(" ")
            for neighbor in neighbor_list:
                if neighbor != '':
                    neighbor = int(neighbor)
                    if neighbor >= node:
                        edgeList.append((node, neighbor))
                        
        random.shuffle(edgeList)
        uniqueEdges_set = set()
        
        for edge in edgeList:
            node, neighbor = edge
            comm1 = community_labels[node]
            comm2 = community_labels[neighbor]
            deleted_flag = 0
            
            if (node, neighbor) not in uniqueEdges_set and (node != neighbor): # This edge is a simple edge
                uniqueEdges_set.add((node, neighbor))
                if comm2 >= comm1: 
                    shared_edge_matrix[comm1][comm2].append([node, neighbor, deleted_flag])
                else:
                    shared_edge_matrix[comm2][comm1].append([neighbor, node, deleted_flag])

            else: # This is either a multi-edge or a self-loop or both
                if comm2 >= comm1: 
                    index_in_shared_edge_matrix = len(shared_edge_matrix[comm1][comm2])
                    shared_edge_matrix[comm1][comm2].append([node, neighbor, deleted_flag])
                    non_simple_edges.append([node, neighbor, index_in_shared_edge_matrix])      
                else:
                    index_in_shared_edge_matrix = len(shared_edge_matrix[comm2][comm1])
                    shared_edge_matrix[comm2][comm1].append([neighbor, node, deleted_flag])
                    non_simple_edges.append([neighbor, node, index_in_shared_edge_matrix])  
                    
            Neighbor_dicts[node][neighbor] += 1
            if node != neighbor:
                Neighbor_dicts[neighbor][node] += 1
                  
        Each_network_result.append(number_of_communities)
        Each_network_result.append(len(non_simple_edges))
                
        case2_count = 0
        case3_count = 0
        
        edge_t1 = time.time()
        for index, edge in enumerate(non_simple_edges): 
            case2_flag = 0
            case3_flag = 0
            
            a = edge[0]
            b = edge[1]
            index_in_shared_edge_matrix = edge[2] # Index of edge a-b in shared_edge_matrix[comm1][comm2]
            comm1 = community_labels[edge[0]]
            comm2 = community_labels[edge[1]]

            if [a, b, 0] not in shared_edge_matrix[comm1][comm2]:
                continue

            # Case-1
            found_second_edge_flag, c, d, p2 = check_for_second_edge_case1(a, b, comm1, comm2, shared_edge_matrix, Neighbor_dicts) 
            
            if found_second_edge_flag == 0: 
                case2_flag = 1
                found_second_edge_flag, c, d, p2 = check_for_second_edge_case2(a, b, comm1, comm2, shared_edge_matrix, Neighbor_dicts) 
                if found_second_edge_flag == 0:
                    case3_flag = 1
                    
            case2_count += case2_flag
            case3_count += case3_flag

            p1 = index_in_shared_edge_matrix
            update_data_structures(a, b, c, d, p1, p2, case3_flag, community_labels, shared_edge_matrix, Neighbor_dicts)

        
        # Once all swaps are done, we create adjacency list for each synthetic network.
        Adj_list = [[] for n in range(numnodes)]
        node = 0
        for neighborList in Neighbor_dicts:
            for neighbor in neighborList:
                if neighbor > node:
                    Adj_list[node].append(neighbor)
                    Adj_list[neighbor].append(node)
            node+=1

        preserved_flag = check_if_ers_and_deg_preserved(number_of_communities, community_labels, e_rs, Adj_list)
        selfLoop_found_flag, multiEdge_found_flag = sanity_check_selfLoops_multiEdges(Adj_list)

        # Store the Adjacency List of the Simplified Network.
        file2 = open("../DCSBM_Simplify/" + domain + "/" + graphname[:-4] + "/" + graphname[:-4] + "_" + str(i+1) + ".txt", "a")

        for row in Adj_list:
            string = ''
            for element in row:
                string = string + str(element)
                string = string + " "
            string = string + "\n"
            file2.write(string)
        file2.close()

        tend = time.time()
        time_taken = (tend - tstart)
        
        Each_network_result.extend([case2_count, case3_count, preserved_flag, selfLoop_found_flag, multiEdge_found_flag])   
        Each_network_result.append(time_taken)
        
        # pickle files with case information stored for each simplified network.
        pickleFile = "../DCSBM_Simplify/" + domain + "/" + graphname[:-4] + "/" + graphname[:-4] + "_" + str(i+1) + ".pkl"
        pickle_out = open(pickleFile,"wb") 
        pickle.dump(Each_network_result, pickle_out)
        pickle_out.close()    

def run(index, domain):
    AllFiles = os.listdir("../network_repository_edgelists/" + domain + "/")
        
    filename = AllFiles[index]
    
    if filename == ".ipynb_checkpoints" or filename == ".DS_Store":
        exit() 
        
    txtfileName = filename.strip()
    
    t1 = time.time()
    simplify_DCSBM_networks(txtfileName, domain)
    t2 = time.time()
    
    print("Total time taken = ", (t2-t1)/3600, "hours.")  

import sys

# Command Line Arguments.
index = sys.argv[1] # index of the network to run this file for.
domain = sys.argv[2] # domain of the chosen network
run(int(index), domain)