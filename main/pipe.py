from __future__ import annotations

import numpy as np

from .tree.linkageTree import linkageCut
from .tree.utils import convert_bitstring_to_matrix
from .tree.utils import string_to_bitstring
from .tsp.TSP_Formulation_Methods import calculate_distances_cost_of_bidireccional_routes
from .tsp.TSP_Formulation_Methods import check_solution_return
from .tsp.TSP_Formulation_Methods import compute_general_lambdas
from .tsp.TSP_Formulation_Methods import create_QUBO_matrix
from .tsp.TSP_Formulation_Methods import solve_qubo_with_Dwave
from .vqaa.vqaa_tools import atoms_list
from .vqaa.vqaa_tools import atoms_register
from .vqaa.vqaa_tools import generate_grid
from .vqaa.vqaa_tools import heuristical_embedding
from .vqaa.vqaa_tools import run_vqaa


def get_best_solution(proposed_sols, N, p, distances, s_node, e_node):
    """
    Filter from the proposed set of solutions the valid and best scoring one.

    :param proposed_sols: Proposed solutions
    :param N: Number of nodes
    :param p: Number of stops
    :param distances: Distance matrix for scoring
    :param s_node: Start node of the level
    :param e_node: End node of the level
    """
    # Discard solutions that do not fulfill our constraints
    constrained = [
        check_solution_return(string_to_bitstring(sol), N, p, s_node - 1, e_node - 1) for sol in proposed_sols
    ]
    # Select the one(s) with lower distance associated cost
    distance_costs = np.array(
        [
            calculate_distances_cost_of_bidireccional_routes(
                np.array(string_to_bitstring(sol)),
                distances,
                p,
            )
            for sol in proposed_sols
        ],
    )
    if np.sum(constrained) == 0:
        raise ValueError(
            "No valid solution, consider increasing the number of proposed solutions or VQAA didnt do its job",
        )
    return proposed_sols[constrained][np.argmin(distance_costs[constrained])]


def execute_step(
    reduced_distances,
    N,
    p,
    startNode,
    endNode,
    lambdas,
    emulator: str = "sv",
    nchecks: int = 500,
    level: int = 0,
    classical: bool = False,
):
    """
    Executes a single step of the algorithm given the input parameters

    :param reduced_distances: Normalized distances
    :param N: Number of nodes
    :param p: Number of desired stops
    :param startNode: Entrance node of the level
    :param endNode: Exit node of the level
    :param lambdas: Weights of the constraints
    :param emulator: Vqaa emulator choice
    :param nchecks: Filter number of proposed solutions
    :param level: For verbose
    """
    Q_matrix_initial, _ = create_QUBO_matrix(
        reduced_distances,
        p,
        startNode - 1,
        endNode - 1,
        lambdas,
    )

    if not classical:
        # Creating register and solving VQAA
        coords = heuristical_embedding(
            atoms_list(len(Q_matrix_initial)),
            generate_grid(50, 50, 1),
            Q_matrix_initial,
        )
        register = atoms_register(coords, show=False)
        # Emulator options: "qutip" (pulser), "mps", and "sv"
        print(f"----- Solving level {level} ------\n")
        C_0, _ = run_vqaa(Q_matrix_initial, register, emulator)
        number_props = min(int(len(C_0.keys())), nchecks)
        C_ = dict(sorted(C_0.items(), key=lambda item: item[1], reverse=True))
        proposed_sols = np.array(list(C_.keys()))[:number_props]
        level = get_best_solution(
            proposed_sols,
            N,
            p,
            reduced_distances,
            startNode,
            endNode,
        )  # symmetric matrix means that we are counting distances twice
        level = np.array(string_to_bitstring(level))
    else:
        level,_ = solve_qubo_with_Dwave(Q_matrix_initial, num_reads=1000)
    
    return level


def give_line(
    amenities_data,
    nclusters,
    p,
    startNode,
    endNode,
    emulator,
    nchecks: int = 1024,
    classical: bool = False,
):
    """
    Executes the whole pipeline and outputs the results

    :param amenities_data: Overpy dataframe with lon/lat values.
    :param nclusters: Number of clusters per level
    :param p: Number of desired stops per level
    :param startNode: Start node of the first level
    :param endNode: End node of the first level
    :param emulator: Which emulator to use for the vqaa
    """
    # Create a hierarchical clustering of amenities
    hierarchical_cluster = linkageCut(amenities_data)
    # Set a specific number of clusters per levels. Max 9 in this POC
    levels = 2
    _ = hierarchical_cluster.top_down_view_recur(nclusters=nclusters, levels=levels)
    # ------ Q level 0
    np.random.seed(21)
    # Fetch the distance from the centers of the first level
    distances = hierarchical_cluster.dist_matrix_level(0, return_labels=False)
    # Set initial global parameters
    N = distances.shape[0]

    # Process Parameters
    p = min(p, N - 1)
    startNode = min(startNode, N)
    endNode = min(endNode, N)
    reduced_distances = distances / np.max(distances)

    lambdas = compute_general_lambdas(
        reduced_distances,
        max_N=3,
    )
    # Emulator options: "qutip" (pulser), "mps", and "sv"
    level0 = execute_step(
        reduced_distances,
        N,
        p,
        startNode,
        endNode,
        lambdas,
        emulator,
        nchecks,
        level=0,
        classical=classical
    )
    adjacency = convert_bitstring_to_matrix(level0, N=N, p=p)

    # Formulation with initial lambdas
    level1 = {}  # Dict that will hold the bitstring, connected level-0 clusters and corresponding start-end nodes
    all_indices = set(np.arange(nclusters - 1) + 1)
    for i in range(1, nclusters + 1):
        print("----- Solving level-1:", i, "------\n")

        connections = (
            np.concatenate(
                [adjacency[:, i - 1].nonzero()[0], adjacency[i - 1, :].nonzero()[0]],
                axis=0,
            )
            + 1
        )
        if len(connections) > 0:  # Selected
            # Fetch the centers of the first level
            distances, closest, _ = hierarchical_cluster.dist_matrix_label_down(
                i,
                connections=connections,
            )
            startNode = None
            if len(closest) >= 1:
                startNode = closest[0]
                choices = all_indices - {startNode}
            if len(closest) == 2:
                endNode = closest[1]
            else:
                endNode = np.random.choice(
                    list(choices),
                )  # POC criterion, better heuristic should be chosen
            reduced_distances = distances / np.max(distances)
            sol = execute_step(
                reduced_distances,
                N,
                p,
                startNode,
                endNode,
                lambdas,
                emulator,
                nchecks,
                level=i,
                classical=classical
            )
            level1[i] = [sol, closest]
        else:
            level1[i] = (np.zeros(nclusters * (p + 1)), [])
    return level0, level1
