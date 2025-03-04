import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from scipy.optimize import minimize
from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

"""
Methods to create the QUBO matrix for using it in Pulser with the Traveling Salesman Problem representation.

- The distances are no symmetric, we use a non symmetric cost matrix.
- N is the number of stops. p is the number of travels in the route. R = N(p+1) is the number of variables.
- We consider just one line which has an start (stop i), and end (stop j) and p travels in the route.

Thera are two levels of representation:

- QUBO variables: y_k, k=0,1,...,R-1 and matrices with shape (R,R). The distances are stored in an N x N matrix which is used several times.
- Stops representation: n_i, i=0,1,...,N-1. Graph representation of the stops.

"""


# QUBO matrix creation

def create_QUBO_matrix(distances, p, startNode: Optional[int] = 0, endNode: Optional[int] = None, 
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

    Q += convertCostMatrixToQUBORepresentation(distances, p)

    Q = Q + Q.T


    # Constraint 1: just one stop in each position of the route

    if constraints_activations[0]:
        Q += list_of_lambdas[0]*create_matrix_constraint_1(N, p)

    # Constraint 2: at least one stop in each position of the route

    if constraints_activations[1]:
        Q += list_of_lambdas[1]*create_matrix_constraint_2(N, p)


    # Constaint 3: just one visit to each node
    if constraints_activations[2]:
        Q += list_of_lambdas[2]*create_matrix_constraint_3(N, p)

    # Constaint 4: start node

    if constraints_activations[3]:
        Q+= list_of_lambdas[3]*create_matrix_constraint_4(N, p,startNode)

    # Constaint 5: end node
    if constraints_activations[4]:
        Q += list_of_lambdas[4]*create_matrix_constraint_5(N, p, endNode)


    return Q, list_of_lambdas


def create_matrix_constraint_1(N, p) -> np.array:
    """
    Create the matrix for the constraint 1.
    """

    R = N*(p+1)
    Q = np.zeros((R,R))

    for l in range(p+1):
        for i in range(N):
            for j in range(N):
                if i != j:
                    Q[i + N*l, j + N*l] += 1

    return Q

def create_matrix_constraint_2(N, p) -> np.array:
    """
    Create the matrix for the constraint 2.
    """

    R = N*(p+1)
    Q = np.zeros((R,R))

    for k in range(R):
        Q[k, k] += 1 - 2 * (p+1)
        for l in range(0, R):
            if l != k:
                Q[k, l] += 1

    return Q

def create_matrix_constraint_3(N, p) -> np.array:
    """
    Create the matrix for the constraint 3.
    """

    R = N*(p+1)
    Q = np.zeros((R,R))

    for k in range(R):
        for f in range(k+N, N*(p+1), N):
            Q[k, f] += 1
            Q[f,k] += 1

    return Q

def create_matrix_constraint_4(N, p, startNode) -> np.array:
    """
    Create the matrix for the constraint 4.
    """

    R = N*(p+1)
    Q = np.zeros((R,R))
    Q[startNode, startNode] += -1

    return Q

def create_matrix_constraint_5(N, p, endNode) -> np.array:
    """
    Create the matrix for the constraint 5.
    """

    R = N*(p+1)
    Q = np.zeros((R,R))
    Q[N*p+endNode, N*p+endNode] += -1

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
    original_route = [np.argmax(solution_array[i*N:(i+1)*N]) for i in range(p+1)]
    inverted_route = original_route[::-1]
    for i, nodo in enumerate(inverted_route):
        inverted_solution[i * N + nodo] = 1

    return inverted_solution

def calculate_distances_cost(solution_array, distances, p):
    """
    Calculate the distances cost of a solution.
    """

    distances_QUBO = convertCostMatrixToQUBORepresentation(distances, p)
    return calculate_cost(solution_array, distances_QUBO)

def calculate_distances_cost_of_bidireccional_routes(solution_array, distances, p):
    """
    Calculate the distances cost of a solution considering bidireccional routes
    It assumes that the distances_QUBO matrix is in the QUBO representation.
    """
    N = distances.shape[0]
    distances_QUBO = convertCostMatrixToQUBORepresentation(distances, p)

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
    for k in range(N):
        sum = 0
        for f in range(k, N*(p+1), N):
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
    Q_dict = {(i, j): Q[i, j] for i in range(num_vars) for j in range(i, num_vars)}
    
    bqm = BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    
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
        distances.append(calculate_distances_cost(b_array, distances_matrix, p))

    zipped = zip(bitstrings, costs, distances)
    sort_zipped = sorted(zipped, key=lambda x: x[1])
    return sort_zipped


# Optimization


def optimize_lambdas_given_solution(distances, p, startNode=0, endNode=None, methodIndex=2, initial_solution= None, initial_lambdas=None):
    """
    Optimize the lambdas for the QUBO matrix with adaptive balancing between cost and constraints. It reduces the cost of a given solution.

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

    methods = [
    "Nelder-Mead",  # Algoritmo simplex (no requiere derivadas)
    "Powell",       # Algoritmo de búsqueda direccional (sin derivadas)
    "L-BFGS-B",     # Variante limitada de BFGS (acepta restricciones de caja)
    "TNC",          # Algoritmo de Newton truncado (adecuado para problemas grandes)
    "COBYLA",       # Optimización secuencial por aproximaciones cuadráticas
    "SLSQP"         # Programación cuadrática secuencial
    ]

    method = methods[methodIndex]

    def cost_function(lambdas):
        
        # Create the QUBO matrix
        Q, _ = create_QUBO_matrix(distances, p, startNode, endNode, lambdas)
        
        # Generate a valid initial solution
        if initial_solution is not None:
            solution_guess = initial_solution
        else:
            solution_guess = generate_valid_initial_solution(N, p, startNode, endNode)
        cost = calculate_cost(solution_guess, Q)
        
        # Adaptive penalty based on the violation level
        penalty = 0
        if not check_solution_return(solution_guess, N, p, startNode, endNode):
            penalty = 10 * abs(cost)  # Adaptive penalty proportional to cost
        
        return cost + penalty
    
    if initial_lambdas is not None:
        lambdas_init = initial_lambdas
    else:
        lambdas_init = np.ones(5) * 0.1*max(distances)  # Start with small lambda values
    res = minimize(cost_function, lambdas_init, method=method, bounds=[(0, 10)] * 5)
    
    return res.x

def calculate_upper_bound_distances(distances, p):
    """
    For a solution with all the stops activated in all time steps, it calculates the cost function as an upper bound for the constraints.
    """
    distances_QUBO = convertCostMatrixToQUBORepresentation(distances, p)
    distances_QUBO_t = distances_QUBO.T + distances_QUBO
    maxArray = np.ones(distances_QUBO_t.shape[0])
    return maxArray.T @ distances_QUBO_t @ maxArray

def update_lambdas(lambdas, scaling_factor, solution, N, p, startNode, endNode):
    """
    For a given solution, it scales the lambdas based on the constraints that are not fulfilled.

    Parameters
    ----------

    lambdas : np.array
        Array with the lambdas.
    scaling_factor : float
        Scaling factor for the lambdas.
    solution : np.array 
        Solution in the QUBO representation.
    N : int
        Number of stops.
    p : int
        Number of travels in the route.

    Returns
    -------
    lambdas_updated : np.array
        Updated lambdas.
    """
    lambdas_updated = []
    endNode = N*p + endNode
    for i in range(len(lambdas)):
        sum = lambdas[i]
        if i == 0:
            if not check_constraint_1(solution,N, p):
                sum = sum + scaling_factor
        elif i == 1:
            if not check_constraint_2(solution,N, p):
                sum = sum + scaling_factor
        elif i == 2:
            if not check_constraint_3(solution,N, p):
                sum = sum + scaling_factor
        elif i == 3:
            if not check_constraint_4(solution, startNode):
                sum = sum + scaling_factor
        elif i == 4:
            if not check_constraint_5(solution, endNode):
                sum = sum + scaling_factor
        lambdas_updated.append(sum)

    return lambdas_updated


def find_optimized_solution(distances, p, N, startNode, endNode, scaling_factor, max_iterations, num_reads_solver, initial_lambdas = None):
    """
    For given distances, travels, start and end nodes, it finds the optimized solution based on the iteration over the funcion update lambdas
    and solving the QUBO with DWave annealer.

    Parameters
    ----------
    distances : np.array
        Matrix (NxN) with the distances between the stops. It is not symmetric.
    p : int
        Number of travels in the route.
    N : int
        Number of stops.
    startNode : int
        Index of the start stop.
    endNode : int
        Index of the end stop.
    scaling_factor : float
        Scaling factor for the lambdas.
    max_iterations : int
        Maximum number of iterations.
    num_reads_solver : int
        Number of reads for the DWave annealer.
    
    Returns
    -------
    solution : np.array
        Optimized solution in the QUBO representation.
    """
    if initial_lambdas is not None:
        lambdas = initial_lambdas
    else:
        lambdas = [0.01*calculate_upper_bound_distances(distances, p) for _ in range(5)] # Start and end nodes are not penalized in the beggining as the have negative penalties
    Q_matrix,_ = create_QUBO_matrix(distances, p, startNode, endNode, lambdas)
    solution, _ = solve_qubo_with_Dwave(Q_matrix, num_reads=num_reads_solver)
    if check_solution_return(solution, N, p, startNode, endNode):
        return solution, lambdas
    else:
        for i in range(max_iterations):
            lambdas = update_lambdas(lambdas, scaling_factor, solution, N, p, startNode, endNode)
            Q_matrix,_ = create_QUBO_matrix(distances, p, startNode, endNode, lambdas)
            solution, _ = solve_qubo_with_Dwave(Q_matrix, num_reads=num_reads_solver)
            if check_solution_return(solution, N, p, startNode, endNode):
                return solution, lambdas
        print("No solution found")
        return solution, lambdas
    
def count_most_violated_constraints(solutions_zipped, N, p, startNode, endNode, plotRange = 20):
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


# Presentation

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


def show_parameters_of_solution(solution, distances, N, p, startNode, endNode, lambdas = None):
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
    scatter = ax1.scatter(range(len(costs)), costs, c=validity, cmap="coolwarm", alpha=0.7, label="Total cost")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Segundo eje Y (coste en distancia)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Distance cost", color="black")
    for i in range(len(costs)):
        if validity[i]:  # Solo para soluciones válidas
            ax2.scatter(i, distance_costs[i], color="black", marker="x", s=80, label="Distance cost" if i == 0 else "")
    ax2.tick_params(axis='y', labelcolor="black")

    fig.tight_layout()
    plt.title("Total cost vs Distance cost")