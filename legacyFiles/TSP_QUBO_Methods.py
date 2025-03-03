import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from scipy.optimize import minimize
from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

"""
Methods to create the QUBO matrix for using it in Pulser with the Travelling Salesman Problem representation.

- The distances are no symmetric, we use a non symmetric cost matrix.
- N is the number of stops. p is the number of travels in the route. R = N(p+1) is the number of variables.
- We consider just one line which has an start (stop i), and end (stop j) and p travels in the route.

Thera are two levels of representation:

- QUBO variables: y_k, k=0,1,...,R-1 and matrices with shape (R,R). The distances are stored in an N x N matrix which is used several times.
- Stops representation: n_i, i=0,1,...,N-1. Graph representation of the stops.

"""



# QUBO matrix creation

def createQUBOmatrix(distances, p, startNode: Optional[int] = 0, endNode: Optional[int] = None, 
                     list_of_lambdas: Optional[List[float]] = None, constraints_activations: Optional[List[bool]]=None) -> Tuple[np.array, List[float]]:
    """
    Creation of Q matrix for the bus lines problem using TSP representation.

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

    Returns
    -------
    Q : np.array
        QUBO matrix with shape (R,R).
    """

    N = distances.shape[0]
    R = N*(p+1)
    Q = np.zeros((R,R))

    if endNode is None:
        endNode = N-1

    if list_of_lambdas is None:
        list_of_lambdas = [1.0 for i in range(5)]

    if constraints_activations is None:
        constraints_activations = [True for i in range(5)]


    # Cost funtion for the route: distances

    for k in range(p):
        for i in range(N):
            for j in range(N):
                Q[i + N*k, j + N*(k+1)] += distances[i,j]

    Q = Q + Q.T


    # Constraint 1: just one stop in each position of the route

    if constraints_activations[0]:
        for l in range(p+1):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        Q[i + N*l, j + N*l] += list_of_lambdas[0]

    # Constraint 2: at least one stop in each position of the route

    if constraints_activations[1]:
        for k in range(R):
            Q[k, k] += list_of_lambdas[1] * (1 - 2 * (p+1))
            for l in range(0, R):
                if l != k:
                    Q[k, l] += list_of_lambdas[1]


    # Constaint 3: just one visit to each node
    if constraints_activations[2]:
        for k in range(R):
            for f in range(k+N, N*(p+1), N):
                Q[k, f] += list_of_lambdas[2]
                Q[f,k] += list_of_lambdas[2]

    # Constaint 4: start node

    if constraints_activations[3]:
        Q[startNode, startNode] += -list_of_lambdas[3]

    # Constaint 5: end node
    if constraints_activations[4]:
        Q[N*p+endNode, N*p+endNode] += -list_of_lambdas[4]


    return Q, list_of_lambdas


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
    Convert the cost matrix to the QUBO representation.

    Parameters
    ----------
    distances : np.array
        Matrix with the distances between the stops. It is not symmetric. Is has shape (N,N).
    p : int
        Number of travels in the route.

    Returns
    -------
    Q : np.array
        QUBO matrix with shape (R,R).
    """

    N = distances.shape[0]
    R = N*(p+1)
    costMatrix = np.zeros((R,R))

    for k in range(p):
        for i in range(N):
            for j in range(N):
                costMatrix[i + N*k, j + N*(k+1)] = distances[i,j]
            
    return costMatrix

def calculate_cost(solution_array, Q_matrix):
    """
    Calculate the cost of a solution.
    It assumes that the Q_matrix and the array are in the QUBO representation.

    If the input is the distances matrix in the QUBO representation, it will return the distances cost.
    """

    return solution_array.T @ Q_matrix @ solution_array

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

    # Extraer la ruta original en índices
    original_route = [np.argmax(solution_array[i*N:(i+1)*N]) for i in range(p+1)]
    
    # Invertir la ruta
    inverted_route = original_route[::-1]
    
    # Mapear la nueva ruta al formato QUBO
    for i, nodo in enumerate(inverted_route):
        inverted_solution[i * N + nodo] = 1

    return inverted_solution

def calculate_distances_cost_of_bidireccional_routes(solution_array, distances_QUBO, N, p):
    """
    Calculate the distances cost of a solution considering bidireccional routes
    It assumes that the distances_QUBO matrix is in the QUBO representation.
    """

    originalRouteCost = calculate_cost(solution_array, distances_QUBO)
    invertedRoute = invert_solution_direction(solution_array, N, p)
    invertedRouteCost = calculate_cost(invertedRoute, distances_QUBO)

    return originalRouteCost + invertedRouteCost




def check_solution(solution_array, N, p, startNode, endNode) -> None:
    """
    Check the constraints of the solution.
    """

    endNode = N*p + endNode

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
    endNode = N*p + endNode

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
    for l in range(p+1):
        if sum(solution_array[l*N:(l+1)*N]) > 1:
            return False
    return True

def check_constraint_2(solution_array, N, p):
    for l in range(p+1):
        if sum(solution_array[l*N:(l+1)*N]) < 1:
            return False
    return True

def check_constraint_3(solution_array, N, p):
    for k in range(N*(p+1)):
        sum = 0
        for f in range(k+N, N*(p+1), N):
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
    


# Drawing


def draw_solution_graph(solution_array, distances, p, startNode, endNode):
    """
    Draw the graph of the solution.
    """

    N = distances.shape[0]
    R = N*(p+1)

    # Create the graph and add a node for each stop

    G = nx.Graph()
    for i in range(N):
        G.add_node(i)

    # Add edges based on the solution

    for k in range(p):
        for i in range(N):
            for j in range(N):
                if solution_array[i + N*k] == 1 and solution_array[j + N*(k+1)] == 1:
                    G.add_edge(i,j)

    # Is the node is a start node, it is colored in red. If it is the end node it is colored in green.

    color_map = []
    for node in G:
        if node == startNode:
            color_map.append('red')
        elif node == endNode:
            color_map.append('green')
        else:
            color_map.append('blue')

    # Draw the graph

    # If the edje i,j is in the solution, the nodes will be drawn at a distance distances[i,j]

    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=color_map, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)



# Optimization of lambdas

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

def optimize_lambdas(distances, p, startNode=0, endNode=None, method="L-BFGS-B", initial_solution= None, initial_lambdas=None, bidireccionality=False):
    """
    Optimize the lambdas for the QUBO matrix with adaptive balancing between cost and constraints.

    Parameters
    ----------
    distances : np.array
        Matrix with the distances between the stops. It is not symmetric. Has shape (N,N).
    p : int
        Number of travels in the route.
    startNode : int
        Index of the start stop.
    endNode : int
        Index of the end stop.
    method : str
        Optimization method from scipy.optimize.minimize. Default is "L-BFGS-B".
    initial_solution : np.array
        Initial solution to use for the optimization.
    initial_lambdas : np.array
        Initial values for the lambdas.
    bidireccionality : bool
        If True, the cost function considers bidirectional routes.

    Returns
    -------
    lambdas : np.array
        Optimized lambdas for the initial tentative solution.
    """
    N = distances.shape[0]
    if endNode is None:
        endNode = N - 1

    def cost_function(lambdas):
        
        # Create the QUBO matrix
        Q, _ = createQUBOmatrix(distances, p, startNode, endNode, lambdas)
        
        # Generate a valid initial solution
        if initial_solution is not None:
            solution_guess = initial_solution
        else:
            solution_guess = generate_valid_initial_solution(N, p, startNode, endNode)
        if bidireccionality:
            cost = calculate_distances_cost_of_bidireccional_routes(solution_guess, Q, N, p)
        else:
            cost = calculate_cost(solution_guess, Q)
        
        # Adaptive penalty based on the violation level
        penalty = 0
        if not check_solution_return(solution_guess, N, p, startNode, endNode):
            penalty = 100 * abs(cost)  # Adaptive penalty proportional to cost
        
        return cost + penalty
    
    if initial_lambdas is not None:
        lambdas_init = initial_lambdas
    else:
        lambdas_init = np.ones(5) * 0.1  # Start with small lambda values
    res = minimize(cost_function, lambdas_init, method=method, bounds=[(0, 10)] * 5)
    
    return res.x


# Solve QUBO 

def solve_qubo_with_Dwave(Q):
    """
    Solve the QUBO problem using D-Wave's Simulated Annealing Sampler.
    """
    num_vars = Q.shape[0]
    Q_dict = {(i, j): Q[i, j] for i in range(num_vars) for j in range(i, num_vars)}
    
    bqm = BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=100)
    
    best_solution = np.array(list(response.first.sample.values()))
    best_cost = response.first.energy
    
    return best_solution, best_cost

def brute_force_finding(Q_matrix, distances_matrix, p):
    """
    Brute force method to find the optimal solution.
    It returns the bitstring, the cost with constraints and the distances.
    """

    N = distances_matrix.shape[0]
    R = N*(p+1)
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
        distances_matrix_converted = convertCostMatrixToQUBORepresentation(distances_matrix, p)
        distances.append(calculate_cost(b_array, distances_matrix_converted))

    zipped = zip(bitstrings, costs, distances)
    sort_zipped = sorted(zipped, key=lambda x: x[1])
    return sort_zipped


def contruct_complete_solution(distances, p, startNode, endNode, method, iterations, bidireccionality):
    N = distances.shape[0]
    initial_solution = generate_valid_initial_solution(N, p, startNode, endNode)
    lambdas = [1.0 for _ in range(5)]
    Q_matrix = None

    for i in range(iterations):
        lambdas = optimize_lambdas(distances, p, startNode, endNode, method, initial_solution=initial_solution, initial_lambdas = lambdas, bidireccionality = bidireccionality)
        Q_matrix, _ = createQUBOmatrix(distances, p, startNode, endNode, lambdas)
        initial_solution, _ = solve_qubo_with_Dwave(Q_matrix)

    best_solution = initial_solution

    distances_QUBO = convertCostMatrixToQUBORepresentation(distances, p)
    best_solution_cost = calculate_cost(best_solution, distances_QUBO)
    best_solution_total_cost = calculate_distances_cost_of_bidireccional_routes(best_solution, distances_QUBO, N, p)

    resultsDict = {
        "lambdas": lambdas,
        "best_solution": best_solution,
        "best_solution_cost": best_solution_cost,
        "best_solution_total_cost": best_solution_total_cost,
        "solution_validity": check_solution_return(best_solution, N, p, startNode, endNode),
        "Q_matrix": Q_matrix
    }

    return resultsDict



# Multi line analysis

def compute_multiple_lines_solution(distances, p_list, startNodes, endNodes, method, iterations, bidireccionality):
    """
    Compute the solution for multiple bus lines.
    """
    L = len(p_list)
    Q_matrices = []
    lambdas_list = []
    best_solutions = []
    best_costs = []
    best_total_costs = []
    solution_validity = []

    for l in range(L):
        resultsSingleLine = contruct_complete_solution(distances, p_list[l], startNodes[l], endNodes[l], method, iterations, bidireccionality)

        Q_matrices.append(resultsSingleLine["Q_matrix"])
        lambdas_list.append(resultsSingleLine["lambdas"])
        best_solutions.append(resultsSingleLine["best_solution"])
        best_costs.append(resultsSingleLine["best_solution_cost"])
        best_total_costs.append(resultsSingleLine["best_solution_total_cost"])
        solution_validity.append(resultsSingleLine["solution_validity"])

    resultsDict = {
        "lambdas": lambdas_list,
        "best_solutions": best_solutions,
        "best_costs": best_costs,
        "best_total_costs": best_total_costs,
        "solutions_validity": solution_validity,
        "Q_matrices": Q_matrices
    }

    return resultsDict

def joint_solution(resultsDict, p_list, N):
    """
    Join the results 'best_solutions' of the multiple lines analysis in a larger solution_array.
    """

    best_solutions = resultsDict["best_solutions"]
    L = len(p_list)
    R = sum([N*(p+1) for p in p_list])
    solution_array = np.zeros(R)

    offset = 0
    for l in range(L):
        p = p_list[l]
        solution = best_solutions[l]
        solution_array[offset:offset + N * (p + 1)] = solution
        offset += N * (p + 1)
    
    return solution_array

def joint_distance_cost(resultsDict):
    """
    Calculate the distances cost of the joint solution. It assumes bidirectional routes.
    """

    return sum(resultsDict["best_total_costs"])

def draw_multiple_solutions(solution_array, distances, p_list, startNodes, endNodes):
    """
    Draw the graph of the solution for multiple bus lines.
    """
    N = distances.shape[0]
    L = len(p_list)
    G = nx.Graph()
    
    # Add nodes
    for i in range(N):
        G.add_node(i)
    
    # Colors for different lines
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    edge_colors = []
    
    offset = 0
    for l in range(L):
        p = p_list[l]
        
        # Add edges based on solution
        for k in range(p):
            for i in range(N):
                for j in range(N):
                    if solution_array[i + N*k + offset] == 1 and solution_array[j + N*(k+1) + offset] == 1:
                        G.add_edge(i, j)
                        edge_colors.append(colors[l % len(colors)])
        
        offset += N * (p + 1)
    
    # Assign colors to nodes
    node_colors = ['gray' for _ in range(N)]
    for l in range(L):
        node_colors[startNodes[l]] = 'red'
        node_colors[endNodes[l]] = 'green'
    
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def check_solution_global_constraint(solution_array, N, p_list):
    """
    Check the global constraint: all stops must be visited at least once.
    """
    visited = np.zeros(N)
    offset = 0
    for l in range(len(p_list)):
        for k in range(p_list[l] + 1):
            for i in range(N):
                if solution_array[i + N*k + offset] == 1:
                    visited[i] = 1  # Mark as visited
        offset += N * (p_list[l] + 1)
    
    return np.all(visited == 1)


def create_valid_complete_graph(distances, N, L, startNodes, endNodes, longitude_of_lines, iterations):
    """
    Creates a complete set of routes for L lines which visit all stops.
    """

    p_list = [min(longitude_of_lines, N-1) for _ in range(L)]
    resultsDict = compute_multiple_lines_solution(distances, p_list, startNodes, endNodes, "L-BFGS-B", iterations, False)
    validity_flag = True
    for solution in resultsDict["solutions_validity"]:
        if not solution:
            validity_flag = False
            break
    completeness_flag = check_solution_global_constraint(joint_solution(resultsDict, p_list, N), N, p_list)

    return resultsDict, p_list, validity_flag, completeness_flag