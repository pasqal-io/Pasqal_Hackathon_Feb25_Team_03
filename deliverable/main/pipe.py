import os 
import glob
import pandas as pd 
import numpy as np
from .tree.linkageTree import linkageCut
from .tsp.TSP_Formulation_Methods import ( 
    create_QUBO_matrix,
    solve_qubo_with_Dwave,
    check_solution,
    check_solution_return,
    load_lambda_means,
    draw_solution_graph,
    brute_force_finding,
    calculate_distances_cost_of_bidireccional_routes,
)
from .vqaa.vqaa_tools import ( 
    heuristical_embedding, 
    atoms_register,
    atoms_list, 
    generate_grid,
    run_vqaa,
    plot_distribution,
)
from .tree.utils import ( 
    view_linkage_on_map, 
    draw_centers_on_map,
    map_draw_line,
    convert_bitstring_to_matrix,
    assemble_line,
    string_to_bitstring,
)

def give_line(amenities_data, nclusters, p, startNode, endNode, emulator):
    # Create a hierarchical clustering of amenities
    hierarchical_cluster = linkageCut(amenities_data)
    # Set a specific number of clusters per levels. Max 9 in this POC
    levels = 2
    labels = hierarchical_cluster.top_down_view_recur(nclusters=nclusters, levels=levels)
    # Visualize for debugging purposes.
    centers = hierarchical_cluster.give_centers_level(0)

    
    # ------ Q level 0
    np.random.seed(21)
    # Fetch the distance from the centers of the first level
    distances = hierarchical_cluster.dist_matrix_level(0, return_labels=False)
    # Set initial global parameters
    N = distances.shape[0]
    node_options = set(np.arange(nclusters) + 1)

    # Process Parameters
    p = min(p, N-1)
    startNode = min(startNode, N)
    endNode = min(endNode, N)

    reduced_distances = distances/np.max(distances)
    maxDistance = np.max(reduced_distances)
    # NOTE: temporary double pardir while we decide the new structure for the lambdas
    lambda_paths = glob.glob(os.path.join(os.path.pardir, 'data', 'lamdasOptimized', '*'))
    mean_lambdas = load_lambda_means(lambda_paths)
    
    Q_matrix_initial,_ = create_QUBO_matrix(reduced_distances, p, startNode - 1, endNode - 1, mean_lambdas)   

    # Creating register and solving VQAA
    coords = heuristical_embedding(atoms_list(len(Q_matrix_initial)), generate_grid(50, 50,1), Q_matrix_initial)
    register = atoms_register(coords, show=False)
    # Emulator options: "qutip" (pulser), "mps", and "sv"
    print("----- Solving level 0 ------\n")
    C_0, x = run_vqaa(Q_matrix_initial, register, emulator)
    
     
    def get_best_solution(proposed_sols, distances, s_node, e_node, p):
        # Discard solutions that do not fulfill our constraints
        constrained = [check_solution_return(string_to_bitstring(sol), N, p, s_node - 1, e_node - 1) for sol in proposed_sols]
        # Select the one(s) with lower distance associated cost
        distance_costs = np.array([calculate_distances_cost_of_bidireccional_routes(np.array(string_to_bitstring(sol)), distances, p) for sol in proposed_sols])
        if np.sum(constrained) == 0:
            print('No valid solution, consider increasing the number of proposed solutions or VQAA didnt do its job')
        return proposed_sols[constrained][np.argmin(distance_costs[constrained])]
    
    number_props = min(int(len(C_0.keys())/2), 300)
    C_ = dict(sorted(C_0.items(), key=lambda item: item[1], reverse=True))
    proposed_sols = np.array(list(C_.keys() ) )[:number_props]
    level0 = get_best_solution(proposed_sols, distances, startNode, endNode, p) # symmetric matrix means that we are counting distances twice
    level0 = np.array(string_to_bitstring(level0))
    adjacency = convert_bitstring_to_matrix(level0, N=N, p=p)

    
    # Formulation with initial lambdas
    level1 = {} # Dict that will hold the bitstring, connected level-0 clusters and corresponding start-end nodes 
    all_indices = set(np.arange(nclusters - 1) + 1)
    for i in range(1, nclusters+1):
        print("----- Solving level-1:", i, "------\n")
    
        connections = np.concatenate([adjacency[:,i-1].nonzero()[0], adjacency[i-1, :].nonzero()[0]], axis=0) + 1
        if len(connections) > 0: #Selected
            # Fetch the centers of the first level
            distances, closest, _ = hierarchical_cluster.dist_matrix_label_down(
            i,
            connections=connections,
            )
        
            startNode_ = None
            if len(closest) >= 1:
                startNode_ = closest[0]
                choices = all_indices - set([startNode_])
            if len(closest) == 2:
                endNode_ = closest[1]
            else:
                endNode_ = np.random.choice(list(choices)) # POC criterion, better heuristic should be chosen
            Q_matrix_initial,_ = create_QUBO_matrix(reduced_distances, p, startNode_ - 1, endNode_ - 1, mean_lambdas)
        
            # Creating register and solving VQAA
            coords = heuristical_embedding(atoms_list(len(Q_matrix_initial)), generate_grid(50, 50,1), Q_matrix_initial)
            register = atoms_register(coords, show=False)
            C_1, x = run_vqaa(Q_matrix_initial, register, emulator)
        
        
            C_ = dict(sorted(C_1.items(), key=lambda item: item[1], reverse=True))
            number_props = min(int(len(C_1.keys())/2), 300)

            proposed_sols = np.array(list(C_.keys() ) )[:number_props]
            sol_ =  get_best_solution(proposed_sols, distances, startNode_, endNode_, p) # symmetric matrix means that we are counting distances twice
            sol_ = string_to_bitstring(sol_)
            level1[i] = [sol_, closest]
        else:
            level1[i] = (np.zeros((nclusters*(p + 1))), [])
    return level0, level1