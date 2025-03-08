from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler

"""
Methods to create the QUBO matrix for using it in Pulser with the Traveling Salesman Problem formulation.
The representation is called TS Bus Line Problem

- The distances are assumed to be symmetric. The cost matrix is symmetrized before creating the QUBO matrix.
- N is the number of stops. p is the number of travels in the route. R = N(p+1) is the number of binary variables.
- We consider just one line which has an start (stop i), and end (stop j) and p travels in the route.
- To introtuce more lines, we can use the methods iteratively and check gloval constraints.

There are two levels of representation:

- QUBO variables: y_k, k=0,1,...,R-1 and matrices with shape (R,R).
  The distances are stored in an N x N matrix which is used several times.
- Stops representation: n_i, i=0,1,...,N-1. Graph representation of the stops.

"""


# QUBO matrix creation


def create_QUBO_matrix(
    distances,
    p,
    startNode: int | None = 0,
    endNode: int | None = None,
    list_of_lambdas: list[float] | None = None,
    constraints_activations: list[bool] | None = None,
) -> tuple[np.array, list[float]]:
    """
    Creation of Q matrix for the bus lines problem using TS Bus Line Problem representation

    Parameters
    ----------
    distances : np.array
        Matrix with the distances between the stops. It is not symmetric. Is has shape (N,N).
    p : int
        Number of travels in the route.
    startNode : int
        Index of the start stop.
    endNode : int
        Index of the end stop.
    list_of_lambdas : list
        List of lambdas (weights) for the constraints. Default is None.
    constraints_activations : list
        List of booleans to activate or deactivate the constraints. Default is None.

    Returns
    -------
    Q : np.array
        QUBO matrix with shape (R,R).
    """

    N = distances.shape[0]
    R = N * (p + 1)
    Q = np.zeros((R, R))

    if endNode is None:
        endNode = N - 1

    if list_of_lambdas is None:
        list_of_lambdas = [1.0 for i in range(5)]

    if constraints_activations is None:
        constraints_activations = [True for i in range(5)]

    # Cost funtion for the route: distances symmetrizated

    distances_symmetric = (distances + distances.T) / 2
    Q += convertCostMatrixToQUBORepresentation(distances_symmetric, p)

    Q = Q + Q.T

    # Constraint 1: just one stop in each position of the route

    if constraints_activations[0]:
        Q += list_of_lambdas[0] * create_matrix_constraint_1(N, p)

    # Constraint 2: at least one stop in each position of the route

    if constraints_activations[1]:
        Q += list_of_lambdas[1] * create_matrix_constraint_2(N, p)

    # Constaint 3: just one visit to each node
    if constraints_activations[2]:
        Q += list_of_lambdas[2] * create_matrix_constraint_3(N, p)

    # Constaint 4: start node

    if constraints_activations[3]:
        Q += list_of_lambdas[3] * create_matrix_constraint_4(N, p, startNode)

    # Constaint 5: end node
    if constraints_activations[4]:
        Q += list_of_lambdas[4] * create_matrix_constraint_5(N, p, endNode)

    return Q, list_of_lambdas


def create_matrix_constraint_1(N, p) -> np.array:
    """
    Create the matrix for the constraint 1.
    """

    R = N * (p + 1)
    Q = np.zeros((R, R))

    for l in range(p + 1):
        for i in range(N):
            for j in range(N):
                if i != j:
                    Q[i + N * l, j + N * l] += 1

    return Q


def create_matrix_constraint_2(N, p) -> np.array:
    """
    Create the matrix for the constraint 2.
    """

    R = N * (p + 1)
    Q = np.zeros((R, R))

    for k in range(R):
        Q[k, k] += 1 - 2 * (p + 1)
        for l in range(0, R):
            if l != k:
                Q[k, l] += 1

    return Q


def create_matrix_constraint_3(N, p) -> np.array:
    """
    Create the matrix for the constraint 3.
    """

    R = N * (p + 1)
    Q = np.zeros((R, R))

    for k in range(R):
        for f in range(k + N, N * (p + 1), N):
            Q[k, f] += 1
            Q[f, k] += 1

    return Q


def create_matrix_constraint_4(N, p, startNode) -> np.array:
    """
    Create the matrix for the constraint 4.
    """

    R = N * (p + 1)
    Q = np.zeros((R, R))
    Q[startNode, startNode] += -1

    return Q


def create_matrix_constraint_5(N, p, endNode) -> np.array:
    """
    Create the matrix for the constraint 5.
    """

    R = N * (p + 1)
    Q = np.zeros((R, R))
    Q[N * p + endNode, N * p + endNode] += -1

    return Q


def is_symmetric(matrix, tol=1e-8):
    """
    Check if a matrix is symmetric.

    Parameters
    ----------
    matrix : np.array
        Matrix to check.
    tol : float
        Tolerance for the comparison.

    Returns
    -------
    bool
        True if the matrix is symmetric.
    """
    return np.allclose(matrix, matrix.T, atol=tol)


# Checks


def convertCostMatrixToQUBORepresentation(distances: np.ndarray, p: int) -> np.ndarray:
    """
    Convert the cost matrix (given as an adjacency matrix) to the QUBO representation.

    Parameters
    ----------
    distances : np.array
        Matrix with the distances between the stops. Is has shape (N,N).
    p : int
        Number of travels in the route.

    Returns
    -------
    Q : np.array
        QUBO matrix with shape (R,R).
    """

    N = distances.shape[0]
    R = N * (p + 1)
    costMatrix = np.zeros((R, R))

    for k in range(p):
        for i in range(N):
            for j in range(N):
                costMatrix[i + N * k, j + N * (k + 1)] = distances[i, j]

    return costMatrix


def calculate_cost(solution_array, Q_matrix):
    """
    Calculate the cost of a solution.
    It assumes that the Q_matrix and the array are in the QUBO representation.

    If the input is the distances matrix in the QUBO representation, it will return the distances cost.
    """

    return solution_array.T @ Q_matrix @ solution_array


def calculate_distances_cost(solution_array, distances, p):
    """
    Calculate the distances cost of a solution.
    The distances matrix is not in the QUBO representation.
    The cost is calculated as a one-way route.
    """

    distances_QUBO = convertCostMatrixToQUBORepresentation(distances, p)
    return calculate_cost(solution_array, distances_QUBO)


def invert_solution_direction(solution_array, N, p):
    """
    Invert the direction of the route in a solution.

    Parameters
    ----------
    solution_array : np.array
        Solution in the QUBO representation.
    N : int
        Number of stops.
    p : int
        Number of travels in the route.

    Returns
    -------
    inverted_solution : np.array
        New solution with the route inverted.
    """
    inverted_solution = np.zeros_like(solution_array)
    original_route = [np.argmax(solution_array[i * N : (i + 1) * N]) for i in range(p + 1)]
    inverted_route = original_route[::-1]
    for i, nodo in enumerate(inverted_route):
        inverted_solution[i * N + nodo] = 1

    return inverted_solution


def calculate_distances_cost_of_bidireccional_routes(solution_array, distances, p):
    """
    Calculate the distances cost of a solution considering bidireccional routes
    The distances matrix is not in the QUBO representation.
    """
    N = distances.shape[0]
    distances_QUBO = convertCostMatrixToQUBORepresentation(distances, p)

    originalRouteCost = calculate_cost(solution_array, distances_QUBO)
    invertedRoute = invert_solution_direction(solution_array, N, p)
    invertedRouteCost = calculate_cost(invertedRoute, distances_QUBO)

    return originalRouteCost + invertedRouteCost


def check_solution(solution_array, N, p, startNode, endNode) -> None:
    """
    Check the constraints of the solution and print the violated constraints.
    """

    endNode = N * p + endNode

    if not check_constraint_1(solution_array, N, p):
        print("Constraint 1 not fulfilled. Some stops are activated at the same time.")
    if not check_constraint_2(solution_array, N, p):
        print("Constraint 2 not fulfilled. At some point there is no stop activated.")
    if not check_constraint_3(solution_array, N, p):
        print("Constraint 3 not fulfilled. Some stops are visited more than once.")
    if not check_constraint_4(solution_array, startNode):
        print("Constraint 4 not fulfilled. The start node is not activated first.")
    if not check_constraint_5(solution_array, endNode):
        print("Constraint 5 not fulfilled. The end node is not activated last.")


def check_solution_return(solution_array, N, p, startNode, endNode) -> bool:
    """
    Check the constraints of the solution and return True if all constraints are fulfilled.
    """
    endNode = N * p + endNode

    if not check_constraint_1(solution_array, N, p):
        return False
    if not check_constraint_2(solution_array, N, p):
        return False
    if not check_constraint_3(solution_array, N, p):
        return False
    if not check_constraint_4(solution_array, startNode):
        return False
    if not check_constraint_5(solution_array, endNode):
        return False

    return True


def check_constraint_1(solution_array, N, p):
    for l in range(p + 1):
        if sum(solution_array[l * N : (l + 1) * N]) > 1:
            return False
    return True


def check_constraint_2(solution_array, N, p):
    for l in range(p + 1):
        if sum(solution_array[l * N : (l + 1) * N]) < 1:
            return False
    return True


def check_constraint_3(solution_array, N, p):
    for k in range(N):
        sum = 0
        for f in range(k, N * (p + 1), N):
            sum += solution_array[f]
        if sum > 1:
            return False
    return True


def check_constraint_4(solution_array, startNode):
    if solution_array[startNode] == 0:
        return False
    return True


def check_constraint_5(solution_array, endNode):
    if solution_array[endNode] == 0:
        return False
    return True


# Solve QUBO


def generate_valid_initial_solution(N, p, startNode, endNode):
    """
    Generate a valid initial solution.
    """
    solution = np.zeros(N * (p + 1), dtype=int)
    solution[startNode] = 1
    solution[N * p + endNode] = 1

    shift = 0
    for k in range(p - 1):
        if k + shift == startNode or k + shift == endNode:
            shift += 1
        solution[N * (k + 1) + (k + shift)] = 1

    return solution


def solve_qubo_with_Dwave(Q, num_reads=100):
    """
    Solve the QUBO problem using D-Wave's Simulated Annealing Sampler.
    """
    num_vars = Q.shape[0]
    Q_dict = {(i, j): Q[i, j] for i in range(num_vars) for j in range(num_vars)}

    bqm = BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)

    best_solution = np.array(list(response.first.sample.values()))
    best_cost = response.first.energy

    return best_solution, best_cost


def brute_force_finding(Q_matrix, distances_matrix, p, bidirectional=True):
    """
    Brute force method to find the optimal solution.
    It returns the bitstring, the cost with constraints and the distances.
    """

    N = distances_matrix.shape[0]
    R = N * (p + 1)
    bitstrings = [np.binary_repr(i, R) for i in range(2**R)]
    costs = []
    # this takes exponential time with the dimension of the QUBO
    for b in bitstrings:
        z = np.array(list(b), dtype=int)
        cost = calculate_cost(z, Q_matrix)
        costs.append(cost)

    distances = []
    for b in bitstrings:
        b_array = np.array(list(b), dtype=int)
        if bidirectional:
            distances.append(calculate_distances_cost_of_bidireccional_routes(b_array, distances_matrix, p))
        else:
            distances.append(calculate_distances_cost(b_array, distances_matrix, p))

    zipped = zip(bitstrings, costs, distances)
    sort_zipped = sorted(zipped, key=lambda x: x[1])
    return sort_zipped


# Optimization


def calculate_upper_bound_distances(distances, p):
    """
    For a solution with all the stops activated in all time steps,
    it calculates the cost function as an upper bound for the constraints.
    """
    distances_QUBO = convertCostMatrixToQUBORepresentation(distances, p)
    distances_QUBO_t = distances_QUBO.T + distances_QUBO
    maxArray = np.ones(distances_QUBO_t.shape[0])
    return maxArray.T @ distances_QUBO_t @ maxArray


def optimize_lambdas(distances, p, startNode, endNode, iterations=10, best_solution_number=30):
    """
    For a given distances matrix, given the number of travels p, the start stop and the end stop,
    it optimizes the lambdas. The algotithm starts with an stimated lambdas, computes the QUBO
    matrix and find all the bitstrings combinations.
    For the first 'best_solution_number' combinations it calculates which constraints are violated
    and updates the lambdas proportionally.

    Parameters
    ----------
    distances : np.array
        Matrix with the distances between the stops. Is has shape (N,N).
    p : int
        Number of travels in the route.
    startNode : int
        Index of the start stop.
    endNode : int
        Index of the end stop.
    iterations : int
        Number of iterations.
    best_solution_number : int
        Number of best solutions to consider.

    Returns
    -------
    Q : np.array
        QUBO matrix.
    lambdas : np.array
        Array with the optimized lambdas.
    solutions_zipped : list
        List with the bitstrings, costs and distances of all combinations.
    """

    initial_lambdas = np.array([0.01 * calculate_upper_bound_distances(distances, p) for i in range(5)])
    for i in range(iterations):
        Q, lambdas = create_QUBO_matrix(distances, p, startNode, endNode, initial_lambdas)
        solutions_zipped = brute_force_finding(Q, distances, p)
        violation_of_constraints = count_most_violated_constraints(
            solutions_zipped,
            distances.shape[0],
            p,
            startNode,
            endNode,
            best_solution_number,
        )
        if i < iterations - 1:
            initial_lambdas = [lambdas[i] + violation_of_constraints[i] for i in range(5)]

    return Q, initial_lambdas, solutions_zipped


def count_most_violated_constraints(solutions_zipped, N, p, startNode, endNode, plotRange=20):
    """
    Count the most violated constraints in a list of solutions.
    """
    constraints = [0, 0, 0, 0, 0]
    for i in range(len(solutions_zipped[:plotRange])):
        solution = np.array(list(solutions_zipped[i][0]), dtype=int)
        if not check_constraint_1(solution, N, p):
            constraints[0] += 1
        if not check_constraint_2(solution, N, p):
            constraints[1] += 1
        if not check_constraint_3(solution, N, p):
            constraints[2] += 1
        if not check_constraint_4(solution, startNode):
            constraints[3] += 1
        if not check_constraint_5(solution, endNode):
            constraints[4] += 1
    for j in range(len(constraints)):
        constraints[j] = constraints[j] / plotRange
    return constraints


def load_lambda_means(files):
    """
    Compute the mean of the lambdas from a list of files.
    """
    all_weights = [np.loadtxt(file) for file in files]
    if not all_weights:
        return []  # Retorna lista vacía si no hay datos

    all_weights = np.array(all_weights)  # Shape: (num_files, 5)

    return np.mean(all_weights, axis=0).tolist()


def show_statistics_lambdas(solutions_analysis_array):
    """
    Show, for a given set of combinations of solutions, the statistics of the lambdas.
    """

    longitude = solutions_analysis_array.shape[0]

    # Sum over all the solutions that are correct
    correct_solutions = np.sum(solutions_analysis_array[:, 2])

    # Relative percentage of correct solutions
    percentage_correct_solutions = correct_solutions / longitude

    print(f"Percentage of correct solutions: {percentage_correct_solutions:.2f}\n")

    for i in range(5):
        correct_solutions_start_i = np.sum(
            solutions_analysis_array[np.where(solutions_analysis_array[:, 0] == i)][:, 2],
        )
        total_solutions_start_i = len(solutions_analysis_array[np.where(solutions_analysis_array[:, 0] == i)])
        percentage_correct_solutions_start_i = correct_solutions_start_i / total_solutions_start_i

        print(f"Percentage of correct solutions starting at node {i}: {percentage_correct_solutions_start_i:.2f}")

        correct_solutions_end_i = np.sum(solutions_analysis_array[np.where(solutions_analysis_array[:, 1] == i)][:, 2])
        total_solutions_end_i = len(solutions_analysis_array[np.where(solutions_analysis_array[:, 1] == i)])
        percentage_correct_solutions_end_i = correct_solutions_end_i / total_solutions_end_i

        print(f"Percentage of correct solutions ending at node {i}: {percentage_correct_solutions_end_i:.2f}\n")


# Presentation

# Drawing


def draw_solution_graph(solution_array, distances, p, startNode, endNode):
    """
    Draw the graph of the solution for a single line
    """

    N = distances.shape[0]
    # Create the graph and add a node for each stop

    G = nx.Graph()
    for i in range(N):
        G.add_node(i)

    # Add edges based on the solution

    for k in range(p):
        for i in range(N):
            for j in range(N):
                if solution_array[i + N * k] == 1 and solution_array[j + N * (k + 1)] == 1:
                    G.add_edge(i, j)

    # Is the node is a start node, it is colored in red. If it is the end node it is colored in green.

    color_map = []
    for node in G:
        if node == startNode:
            color_map.append("red")
        elif node == endNode:
            color_map.append("green")
        else:
            color_map.append("blue")

    # Draw the graph

    # If the edje i,j is in the solution, the nodes will be drawn at a distance distances[i,j]

    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=color_map, with_labels=True)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def show_parameters_of_solution(solution, distances, N, p, startNode, endNode, lambdas=None):
    """
    Show the parameters of the solution.

    Returns
    -------

    Q_matrix : np.array
        QUBO matrix.
    solution_cost : float
        Cost of the solution.
    solution_total_cost : float
        Total cost of the solution as considered bidirectional.

    """

    solution_cost = calculate_distances_cost(solution, distances, p)
    solution_total_cost = calculate_distances_cost_of_bidireccional_routes(solution, distances, p)

    print("\nOptimized solution:")
    print(solution)
    print("\nOptimized solution cost:")
    print(solution_cost)
    print("\nOptimized solution total cost:")
    print(solution_total_cost)

    if lambdas is not None:
        print("\nLambdas:")
        print(lambdas)

    print("\nValidity of the solution:")
    print(check_solution_return(solution, N, p, startNode, endNode))

    Q_matrix, _ = create_QUBO_matrix(distances, p, startNode, endNode, lambdas)

    return Q_matrix, solution_cost, solution_total_cost


def plot_brute_force_minimums(solutions_zipped, N, p, startNode, endNode, rangePlot=20):
    """
    Plot the minimums found with the brute force method.
    - The left Y-axis shows the total cost.
    - The right Y-axis shows the distance cost.
    - Solutions are colored by validity (yellow = valid, blue = invalid).
    """

    validity = []
    costs = []
    distance_costs = []

    for i in range(len(solutions_zipped[:rangePlot])):
        solution = np.array(list(solutions_zipped[i][0]), dtype=int)
        costs.append(solutions_zipped[i][1])  # Total cost including constraints
        distance_costs.append(solutions_zipped[i][2])  # Distance cost
        validity.append(check_solution_return(solution, N, p, startNode, endNode))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Primer eje Y (coste total)
    ax1.set_xlabel("Solution index (ordered by total cost)")
    ax1.set_ylabel("Total cost", color="blue")
    _ = ax1.scatter(range(len(costs)), costs, c=validity, cmap="coolwarm", alpha=0.7, label="Total cost")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Segundo eje Y (coste en distancia)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Distance cost", color="black")
    for i in range(len(costs)):
        if validity[i]:  # Solo para soluciones válidas
            ax2.scatter(i, distance_costs[i], color="black", marker="x", s=80, label="Distance cost" if i == 0 else "")
    ax2.tick_params(axis="y", labelcolor="black")

    fig.tight_layout()
    plt.title("Total cost vs Distance cost")


# Multiline Methods


def eliminate_stop_from_distance_matrix(distances, stop):
    """
    Eliminate a stop from the (NxN) distance matrix.
    """
    distances_copy = distances.copy()
    distances_copy = np.delete(distances, stop, axis=0)
    distances_copy = np.delete(distances, stop, axis=1)
    return distances_copy


def add_independent_stop_to_solution(solution, N, p, stop):
    """
    Add an independent, non connected stop to the solution in the exact position.
    """
    solution_copy = solution.copy()
    for k in range(p + 1):
        solution_copy = np.insert(solution_copy, k * N + stop, 0)

    return solution_copy


def generate_solutions_for_multiple_lines(distances, p, startNodes, endNodes, num_lines, lambdas, num_reads):
    """
    Generate solutions for multiple lines.
    When the first line is computed, the stops are removed from the distance matrix.
    """
    N = distances.shape[0]
    solutions = []
    distances_copy = distances.copy()
    for l in range(num_lines):
        if l == 0:
            Q, _ = create_QUBO_matrix(distances_copy, p, startNodes[l], endNodes[l], lambdas[l])
            solution, _ = solve_qubo_with_Dwave(Q, num_reads)
        else:
            for j in range(l):
                distances_copy = eliminate_stop_from_distance_matrix(distances_copy, startNodes[j])
                distances_copy = eliminate_stop_from_distance_matrix(distances_copy, endNodes[j])
            Q, _ = create_QUBO_matrix(distances, p, startNodes[l], endNodes[l], lambdas[l])
            solution, _ = solve_qubo_with_Dwave(Q, num_reads)

            for j in range(l):
                solution = add_independent_stop_to_solution(solution, N, p, startNodes[j])
                solution = add_independent_stop_to_solution(solution, N, p, endNodes[j])

        solutions.append(solution)
        if l < num_lines - 1:
            if startNodes[l] < startNodes[l + 1]:
                startNodes[l + 1] -= 1
            if endNodes[l] < endNodes[l + 1]:
                endNodes[l + 1] -= 1

    return solutions


def generate_solutions_for_multiple_lines_uninformed(distances, p, startNodes, endNodes, num_lines, lambdas, num_reads):
    """
    Generate solutions for multiple lines. The stops are not removed from the distance matrix.
    """
    solutions = []
    for l in range(num_lines):
        Q, _ = create_QUBO_matrix(distances, p, startNodes[l], endNodes[l], lambdas[l])
        solution, _ = solve_qubo_with_Dwave(Q, num_reads)
        solutions.append(solution)

    return solutions


def check_multiline_validity(list_of_solutions, N, p, startNodes, endNodes, num_lines, returnFormat=False):
    """
    Check the validity of the multiline solutions.
    """

    for l in range(num_lines):
        if not check_solution_return(list_of_solutions[l], N, p, startNodes[l], endNodes[l]):
            if returnFormat:
                return False
            else:
                print(f"Local constraint: solution {l} is not valid.")
                return None

    # Global constraint: All stops are visited
    visited_stops = np.zeros(N)
    for l in range(num_lines):
        for k in range(N):
            for f in range(k, N * (p + 1), N):
                visited_stops[k] += list_of_solutions[l][f]

    for i in range(N):
        if visited_stops[i] < 1:
            if returnFormat:
                return False
            else:
                print(f"Global Constraint: stop {i} is not visited.")
                return None

    if returnFormat:
        return True
    else:
        print("All solutions are valid.")
        return None


def distance_cost_of_multilines(list_of_solutions, distances, p, bidirectional=True):
    """
    Calculate the distance cost of the multiline solutions.
    """

    total_cost = 0
    for l in range(len(list_of_solutions)):
        if bidirectional:
            total_cost += calculate_distances_cost_of_bidireccional_routes(list_of_solutions[l], distances, p)
        else:
            total_cost += calculate_distances_cost(list_of_solutions[l], distances, p)

    return total_cost


def generate_all_start_end_combinations(N, L, symmetric=True):
    """
    Generates all possible combinations of startNodes and endNodes satisfying:
    - startNodes[i] ≠ startNodes[j] (all distinct)
    - endNodes[i] ≠ endNodes[j] (all distinct)
    - startNodes[i] ≠ endNodes[j] (cannot be repeated in both)
    - If symmetric = True: Duplicates where (startNodes, endNodes) are equivalent to (endNodes, startNodes) are removed.

    Parameters:
    - N: Total number of available nodes (values between 0 and N-1).
    - L: Number of lines (length of the arrays).

    Returns:
    - List of tuples (startNodes, endNodes) with all valid combinations for the given number of lines.
    """
    nodes = list(range(N))
    valid_combinations = []

    # Generate all possible combinations of L distinct nodes for startNodes (ORDER DOES NOT MATTER)
    for start_comb in itertools.combinations(nodes, L):
        remaining_nodes = set(nodes) - set(start_comb)

        # Generate all permutations of L nodes for endNodes (ORDER MATTERS)
        for end_perm in itertools.permutations(remaining_nodes, L):
            if not symmetric:
                valid_combinations.append((np.array(start_comb), np.array(end_perm)))
            else:
                for i in range(L):
                    if start_comb[i] >= end_perm[i]:
                        break
                    else:
                        valid_combinations.append((np.array(start_comb), np.array(end_perm)))

    return valid_combinations


def draw_multiple_solutions_graph(solutions_list, distances, p, startNodes, endNodes):
    """
    Draws a graph with multiple bus lines given a list of solutions.

    - Each solution represents a bus line.
    - Each bus line is drawn with a different color.
    - If multiple lines share an edge, multiple edges of different colors will be drawn.
    """
    N = distances.shape[0]
    G = nx.Graph()

    for i in range(N):
        G.add_node(i)

    color_palette = itertools.cycle(plt.cm.tab10.colors)

    edge_colors = {}

    for solution_array, startNode, endNode, color in zip(solutions_list, startNodes, endNodes, color_palette):
        for k in range(p):
            for i in range(N):
                for j in range(N):
                    if solution_array[i + N * k] == 1 and solution_array[j + N * (k + 1)] == 1:
                        edge = (i, j) if i < j else (j, i)
                        if edge not in edge_colors:
                            edge_colors[edge] = []
                        edge_colors[edge].append(color)

    color_map = []
    for node in G.nodes():
        if node in startNodes:
            color_map.append("red")
        elif node in endNodes:
            color_map.append("green")
        else:
            color_map.append("blue")

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=500, edge_color="gray", alpha=0.5)

    for edge, colors in edge_colors.items():
        i, j = edge
        for idx, color in enumerate(colors):
            offset = 0.05 * (idx - len(colors) / 2)
            edge_pos = {i: (pos[i][0], pos[i][1] + offset), j: (pos[j][0], pos[j][1] + offset)}
            nx.draw_networkx_edges(G, edge_pos, edgelist=[edge], edge_color=[color], width=2)

    plt.title("Complete bus route lines")
